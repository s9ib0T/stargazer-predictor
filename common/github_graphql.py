import os
import time
import logging

import requests
from dotenv import load_dotenv

load_dotenv() # read .env into env
GRAPHQL_URL = "https://api.github.com/graphql"
REST_API = "https://api.github.com"
log = logging.getLogger(__name__)


# one search page = up to 40 repos + their counts in a single call
# the nested connections only ask for totalCount so they cost ~0 points, a page is basically 1 point -> way cheaper than the old 4 REST calls/repo
# first:40 not 100: github times out building a big page for this query and returns 502. measured 502 rate: 100/75 always fail, 50 ~1 in 12, 40 and below clean. the 5xx retry in _post covers the rare miss
SEARCH_QUERY = """
query($q: String!, $cursor: String)
{
    search(query: $q, type: REPOSITORY, first: 40, after: $cursor)
    {
        repositoryCount
        pageInfo { hasNextPage endCursor }
        nodes
        {
            ... on Repository 
            {
                nameWithOwner
                stargazerCount
                forkCount
                watchers { totalCount }
                issues(states: OPEN) { totalCount }
                pullRequests(states: OPEN) { totalCount }
                releases { totalCount }
                repositoryTopics { totalCount }
                defaultBranchRef { target { ... on Commit { history { totalCount } } } }
                primaryLanguage { name }
                licenseInfo { key }
                createdAt updatedAt pushedAt
                isArchived isFork isDisabled isMirror
                hasIssuesEnabled hasWikiEnabled hasProjectsEnabled hasDiscussionsEnabled
                diskUsage
                description
                owner { __typename }
            }
        }
    }
}
"""

# cheap count, first:1 because search needs first/last (0 not allowed)
COUNT_QUERY = """
query($q: String!)
{
    search(query: $q, type: REPOSITORY, first: 1) { repositoryCount }
}
"""

# single repo lookup for the predictor (no search). same fields as above
REPO_QUERY = """
query($owner: String!, $name: String!)
{
    repository(owner: $owner, name: $name)
    {
        nameWithOwner
        stargazerCount
        forkCount
        watchers { totalCount }
        issues(states: OPEN) { totalCount }
        pullRequests(states: OPEN) { totalCount }
        releases { totalCount }
        repositoryTopics { totalCount }
        defaultBranchRef { target { ... on Commit { history { totalCount } } } }
        primaryLanguage { name }
        licenseInfo { key }
        createdAt updatedAt pushedAt
        isArchived isFork isDisabled isMirror
        hasIssuesEnabled hasWikiEnabled hasProjectsEnabled hasDiscussionsEnabled
        diskUsage
        description
        owner { __typename }
    }
}
"""


def flatten(node):
    # graphql node -> flat dict that features.make_features understands
    commit_count = 0
    ref = node.get("defaultBranchRef")
    if ref and ref.get("target") and ref["target"].get("history"):
        commit_count = ref["target"]["history"]["totalCount"]
    lang = (node.get("primaryLanguage") or {}).get("name")
    lic = (node.get("licenseInfo") or {}).get("key")
    pushed = node.get("pushedAt") or node.get("updatedAt") or node["createdAt"]
    return {
        "full_name": node["nameWithOwner"],
        "stargazers_count": node["stargazerCount"],  # target
        "forks_count": node["forkCount"],
        "watchers": node["watchers"]["totalCount"],
        "open_issues_count": node["issues"]["totalCount"],
        "open_pulls_count": node["pullRequests"]["totalCount"],
        "release_count": node["releases"]["totalCount"],
        "topics_count": node["repositoryTopics"]["totalCount"],
        "commit_count": commit_count,
        "disk_usage": node.get("diskUsage") or 0,
        "language": lang,
        "license_key": lic,
        "created_at": node["createdAt"],
        "updated_at": node["updatedAt"],
        "pushed_at": pushed,
        "is_archived": node["isArchived"],
        "is_fork": node["isFork"],
        "is_disabled": node["isDisabled"],
        "is_mirror": node["isMirror"],
        "has_issues": node["hasIssuesEnabled"],
        "has_wiki": node["hasWikiEnabled"],
        "has_projects": node["hasProjectsEnabled"],
        "has_discussions": node["hasDiscussionsEnabled"],
        "description": node.get("description") or "",
        "owner_type": node["owner"]["__typename"],
    }
    
class GitHubGraphQL:
    def __init__(self, token=None):
        # token from arg first, then env
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN missing (put it in .env)")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json"
            }
        )
        
    def _post(self, query, variables, max_retries=4):
        # one graphql request. sleeps through rate limits, retries 5xx
        for attempt in range(max_retries + 1):
            r = self.session.post(GRAPHQL_URL, json={"query": query, "variables": variables})
            
            # primary rate limit -> wait for reset
            if r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
                reset = int(r.headers.get("x-ratelimit-reset", time.time() + 60))
                wait = max(reset - int(time.time()), 0) + 2
                log.warning(f"rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            
            # secondary/abuse limit -> retry-after
            if r.status_code in (403, 429) and "retry-after" in r.headers:
                wait = int(r.headers["retry-after"]) + 1
                log.warning(f"secondary limit, sleeping {wait}s")
                time.sleep(wait)
                continue
            
            if r.status_code >= 500 and attempt < max_retries:
                backoff = 2 ** attempt
                log.warning(f"{r.status_code} from graphql, retry in {backoff}s")
                time.sleep(backoff)
                continue
            
            r.raise_for_status()
            body = r.json()
            
            # graphql can hand back 200 + an errors array
            if body.get("errors"):
                msg = body["errors"][0].get("message", "")
                if "rate limit" in msg.lower() and attempt < max_retries:
                    log.warning("graphql rate limit error, sleeping 60s")
                    time.sleep(60)
                    continue
                raise RuntimeError(f"graphql error: {body['errors']}")
            return body["data"]
        
        raise RuntimeError("giving up on graphql after retries")
    
    def count_repos(self, q):
        # repositoryCount without pulling the nodes
        return self._post(COUNT_QUERY, {"q": q})["search"]["repositoryCount"]
    
    def search_repos(self, q, max_repos=None):
        # walk a search query page by page, yield flat repo dicts
        cursor = None
        seen = 0
        while True:
            search = self._post(SEARCH_QUERY, {"q": q, "cursor": cursor})["search"]
            for node in search["nodes"]:
                if not node: # null nodes happen, skip
                    continue
                yield flatten(node)
                seen += 1
                if max_repos and seen >= max_repos:
                    return
            page = search["pageInfo"]
            if not page["hasNextPage"]:
                return
            cursor = page["endCursor"]
    
    def get_repo(self, full_name):
        # single repo for the predictor. full_name like "owner/name"
        owner, name = full_name.split("/", 1)
        node = self._post(REPO_QUERY, {"owner": owner, "name": name})["repository"]
        if node is None:
            raise RuntimeError(f"repo not found: {full_name}")
        return flatten(node)
    
    def contributor_count(self, full_name):
        # optional extra. graphql cant do contributors, so REST Link header trick
        url = f"{REST_API}/repos/{full_name}/contributors"
        r = self.session.get(url, params={"per_page": 1, "anon": "true"})
        r.raise_for_status()
        last = r.links.get("last", {}).get("url")
        if last is None:
            return len(r.json())
        return int(last.rsplit("page=", 1)[1].split("&", 1)[0])