"""
make a small synthetic data/raw/repos.jsonl so the whole pipeline (build_dataset -> train -> select_best -> predictor) can be tested offline, no github token, no network.
features are correlated with stars on purpose so the models actually learn something.
"""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(0)
OUT = Path("data/raw/repos.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

langs = ["Python", "JavaScript", "Go", "Rust", "C++", "Java", "TypeScript", None]
lics = ["mit", "apache-2.0", "gpl-3.0", "bsd-3-clause", None]
owners = ["Organization", "User"]


def iso(days_ago):
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


N = 600
with OUT.open("w") as f:
    for i in range(N):
        # log-uniform stars from ~50 to ~300k, full range like the real crawl
        stars = int(10 ** random.uniform(1.7, 5.5))
        scale = stars ** 0.6  # forks/commits/etc loosely track stars
        forks = max(0, int(scale * random.uniform(0.05, 0.3)))
        watchers = max(0, int(scale * random.uniform(0.02, 0.1)))
        commits = max(1, int(scale * random.uniform(0.5, 5) + random.uniform(0, 500)))
        issues = max(0, int(scale * random.uniform(0.01, 0.2)))
        # open + closed, always >= open. the issue-rate features divide by this
        total_issues = issues + int(scale * random.uniform(0.1, 1.0))
        age = random.randint(60, 4000)
        repo = {
            "full_name": f"user{i}/repo{i}",
            "stargazers_count": stars,
            "forks_count": forks,
            "watchers": watchers,
            "open_issues_count": issues,
            "total_issues_count": total_issues,
            "open_pulls_count": random.randint(0, 50),
            "release_count": random.randint(0, 60),
            "topics_count": random.randint(0, 10),
            "commit_count": commits,
            "disk_usage": random.randint(100, 500000),
            "language": random.choice(langs),
            "license_key": random.choice(lics),
            "created_at": iso(age),
            "updated_at": iso(random.randint(0, 60)),
            "pushed_at": iso(random.randint(0, 120)),
            "is_archived": random.random() < 0.05,
            "is_fork": False,
            "is_disabled": False,
            "is_mirror": False,
            "has_issues": True,
            "has_wiki": random.random() < 0.7,
            "has_projects": random.random() < 0.5,
            "has_discussions": random.random() < 0.3,
            "description": "x" * random.randint(0, 120),
            "owner_type": random.choice(owners),
        }
        f.write(json.dumps(repo) + "\n")

print(f"wrote {N} synthetic repos to {OUT}")