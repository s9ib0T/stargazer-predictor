import argparse
import json
import logging
from pathlib import Path


def merge(shards_dir, out_path):
    # append every shard file into one jsonl, one file at a time, dedupe by full_name
    shards_dir = Path(shards_dir)
    files = sorted(shards_dir.glob("repos_*.jsonl"))
    if not files:
        raise SystemExit(f"no shard files in {shards_dir}")
    seen = set()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for fp in files:
            for line in fp.open(encoding="utf-8"):
                repo = json.loads(line)
                fn = repo["full_name"]
                if fn in seen:  # window overlaps / reruns can dup
                    continue
                seen.add(fn)
                out.write(json.dumps(repo) + "\n")
                n += 1
            logging.info(f"merged {fp.name}")
    logging.info(f"wrote {n} unique repos to {out_path}")
    return n


def parse_args():
    p = argparse.ArgumentParser(description="merge shard files into one jsonl")
    p.add_argument("--shards-dir", type=Path, default=Path("data/raw/shards"))
    p.add_argument("--out", type=Path, default=Path("data/raw/repos.jsonl"))
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    merge(args.shards_dir, args.out)


if __name__ == "__main__":
    main()