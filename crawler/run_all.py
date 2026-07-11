import argparse
import logging
from pathlib import Path

from common.github_graphql import GitHubGraphQL
from crawler.star_windows import make_bands
from crawler.fetch import fetch_band, setup_logging
from crawler.merge import merge


def parse_args():
    p = argparse.ArgumentParser(description="crawl all star bands then merge")
    p.add_argument("--total", type=int, default=5000, help="rough target repo count")
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/shards"))
    p.add_argument("--merged", type=Path, default=Path("data/raw/repos.jsonl"))
    p.add_argument("--with-contributors", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    bands = make_bands()
    # split the target evenly across bands -> balanced sample over star sizes
    per_band = max(args.total // len(bands), 1)
    logging.info(f"{len(bands)} bands, ~{per_band} repos each, target ~{args.total}")

    client = GitHubGraphQL()
    for lo, hi in bands:
        fetch_band(client, lo, hi, per_band, args.out_dir, args.with_contributors)

    merge(args.out_dir, args.merged)


if __name__ == "__main__":
    main()