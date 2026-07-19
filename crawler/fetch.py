import argparse
import json
import logging
from pathlib import Path

from common.github_graphql import GitHubGraphQL
from crawler.star_windows import make_bands, band_query, band_label


def setup_logging():
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler("data/logs/crawler.log"),
            logging.StreamHandler(),
        ],
    )


def fetch_band(client, lo, hi, per_band, out_dir, with_contributors=False):
    # crawl one star band into its own shard file
    # resumable: a .done marker means the band finished, skip it on rerun
    label = band_label(lo, hi)
    out = Path(out_dir) / f"repos_{label}.jsonl"
    done = Path(str(out) + ".done")
    out.parent.mkdir(parents=True, exist_ok=True)
    if done.exists():
        logging.info(f"band {label} done, skip")
        return 0

    q = band_query(lo, hi) + " sort:stars-desc"
    written = 0
    # write fresh each run, marker at the end means complete (no half files)
    with out.open("w", encoding="utf-8") as f:
        for repo in client.search_repos(q, max_repos=per_band):
            if with_contributors:
                try:
                    repo["contributor_count"] = client.contributor_count(repo["full_name"])
                except Exception as e:
                    logging.warning(f"contributors failed {repo['full_name']}: {e}")
                    repo["contributor_count"] = 0
            f.write(json.dumps(repo) + "\n")
            written += 1
            if written % 100 == 0:
                logging.info(f"band {label}: {written}")
    done.write_text("ok\n")
    logging.info(f"band {label} wrote {written} (target {per_band})")
    return written


def parse_args():
    p = argparse.ArgumentParser(description="crawl one star band (a shard)")
    p.add_argument("--band-index", type=int, required=True,
                   help="which band from make_bands() to crawl")
    p.add_argument("--per-band", type=int, default=400)
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/shards"))
    p.add_argument("--with-contributors", action="store_true",
                   help="extra REST call/repo for contributor_count")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    bands = make_bands()
    if not 0 <= args.band_index < len(bands):
        raise SystemExit(f"band-index must be in [0,{len(bands)})")
    lo, hi = bands[args.band_index]
    client = GitHubGraphQL()
    fetch_band(client, lo, hi, args.per_band, args.out_dir, args.with_contributors)


if __name__ == "__main__":
    main()