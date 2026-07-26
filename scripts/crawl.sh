#!/bin/bash
# crawl all star bands then merge into data/raw/repos.jsonl
# needs GITHUB_TOKEN in .env
# extra args pass through, eg --total 10000
set -e
cd "$(dirname "$0")/.."
python -m crawler.run_all "$@"
