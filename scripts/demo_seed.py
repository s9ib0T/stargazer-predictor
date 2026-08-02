"""
make a small synthetic data/raw/repos.jsonl so the whole pipeline (build_dataset -> train -> select_best -> predictor) can be tested offline, no github token, no network
features are correlated with stars on purpose so the models actually learn something

writes to the same path the crawler uses, so it refuses to run if that file
holds a real crawl. every synthetic record carries a marker key to tell them apart
"""

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

MARKER = "_synthetic"  # set on every generated record, a real crawl never has it

langs = ["Python", "JavaScript", "Go", "Rust", "C++", "Java", "TypeScript", None]
lics = ["mit", "apache-2.0", "gpl-3.0", "bsd-3-clause", None]
owners = ["Organization", "User"]


def parse_args():
    p = argparse.ArgumentParser(description="write a synthetic repos.jsonl for offline testing")
    p.add_argument("--out", type=Path, default=Path("data/raw/repos.jsonl"))
    p.add_argument("-n", "--count", type=int, default=600)
    p.add_argument("--force", action="store_true", help="overwrite even if the file holds a real crawl")
    return p.parse_args()


def holds_real_crawl(path):
    # only the first record is checked, the file is written in one go so it is enough
    with path.open(encoding="utf-8") as f:
        first = f.readline()
    if not first.strip():
        return False  # empty file, nothing to lose
    try:
        return not json.loads(first).get(MARKER)
    except json.JSONDecodeError:
        return True  # cannot tell, assume it matters


def iso(days_ago):
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    args = parse_args()
    out = args.out
    if out.exists() and not args.force and holds_real_crawl(out):
        raise SystemExit(
            f"{out} looks like a real crawl, not overwriting\n"
            f"  write somewhere else:  --out data/raw/demo.jsonl\n"
            f"  or overwrite anyway:   --force\n"
            f"a real crawl can be rebuilt from data/raw/shards with bash scripts/crawl.sh"
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    with out.open("w") as f:
        for i in range(args.count):
            # log-uniform stars from ~50 to ~300k, full range like the real crawl
            stars = int(10 ** random.uniform(1.7, 5.5))
            scale = stars ** 0.6  # forks/commits/etc loosely track stars
            forks = max(0, int(scale * random.uniform(0.05, 0.3)))
            watchers = max(0, int(scale * random.uniform(0.02, 0.1)))
            commits = max(1, int(scale * random.uniform(0.5, 5) + random.uniform(0, 500)))
            issues = max(0, int(scale * random.uniform(0.01, 0.2)))
            # open + closed, always >= open
            # the issue-rate features divide by this
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
                MARKER: True,  # so a rerun knows this file is safe to overwrite
            }
            f.write(json.dumps(repo) + "\n")

    print(f"wrote {args.count} synthetic repos to {out}")


if __name__ == "__main__":
    main()
