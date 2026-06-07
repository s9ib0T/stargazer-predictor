import math
from datetime import datetime, timezone

# numerics copied straight over (missing/None -> 0)
NUM_FIELDS = [
    "forks_count",
    "watchers",
    "open_issues_count",
    "total_issues_count",
    "open_pulls_count",
    "release_count",
    "commit_count",
    "topics_count",
    "disk_usage",
    "description_length",
    "name_length",
    "contributor_count",  # optional, 0 unless crawled with --with-contributors
]

# bool-ish -> 0/1
BOOL_FIELDS = [
    "is_archived",
    "is_fork",
    "is_disabled",
    "is_mirror",
    "has_issues",
    "has_wiki",
    "has_projects",
    "has_discussions",
    "has_license",
    "owner_is_org",
]

# most skewed counts also get a log1p column so ridge/linear cope
LOG_FIELDS = ["forks_count", "commit_count", "open_issues_count", "total_issues_count", "disk_usage"]


def days_since(iso):
    # whole days between a github timestamp and now (utc)
    dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).days


def make_features(repo):
    # one flat repo dict -> flat dict of features
    repo = dict(repo) 
    # derived bits first
    repo["description_length"] = len(repo.get("description") or "")
    repo["name_length"] = len(repo.get("full_name") or "")
    repo["has_license"] = 1 if repo.get("license_key") else 0
    repo["owner_is_org"] = 1 if repo.get("owner_type") == "Organization" else 0
    
    f = {}
    for k in NUM_FIELDS:
        f[k] = repo.get(k) or 0
    for k in BOOL_FIELDS:
        f[k] = int(bool(repo.get(k)))
    
    # dates -> days since
    age = days_since(repo["created_at"])
    f["age_days"] = age
    f["days_since_push"] = days_since(repo.get("pushed_at") or repo["created_at"])
    f["days_since_update"] = days_since(repo.get("updated_at") or repo["created_at"])
    
    # rates, guard divide-by-zero on new repos
    years = max(age / 365.0, 1 / 365.0)
    f["commits_per_day"] = f["commit_count"] / max(age, 1)
    f["commits_per_year"] = f["commit_count"] / years
    # total issues ever filed / age = real open-rate
    f["issues_per_day"] = f["total_issues_count"] / max(age, 1)
    # share of all issues still open: high = backlog piling up / unmaintained
    f["open_issue_ratio"] = f["open_issues_count"] / max(f["total_issues_count"], 1)
    f["forks_per_day"] = f["forks_count"] / max(age, 1)
    f["releases_per_year"] = f["release_count"] / years

    # log versions of the skewed counts
    for k in LOG_FIELDS:
        f[f"log_{k}"] = math.log1p(f[k])

    # categoricals stay raw, one-hot later in build_dataset
    f["language"] = repo.get("language") or "unknown"
    f["license_key"] = repo.get("license_key") or "none"
    return f