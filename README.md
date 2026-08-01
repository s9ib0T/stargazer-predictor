# GitHub Stargazer Predictor

Predicts how many stars a GitHub repo has from its activity data (forks, watchers, commits, issues, releases, age, language, etc).
Give it a few repos and it ranks them by predicted stars.

Everything runs locally in Docker containers with hard CPU and RAM caps.

## Setup

Needs Python 3.10+ and a GitHub token.

    cp .env.example .env  # put your token in it
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Crawl data (needs a token)

Pull repos across the full star range into `data/raw/repos.jsonl`:

    python -m crawler.run_all  # ~5000 repos, all star bands

## Try it offline (no token)

Proves the pipeline on synthetic data, no GitHub token needed:

    python scripts/demo_seed.py  # make a fake repos.jsonl
    python -m trainer.build_dataset  # features + train/test split
    python -m trainer.train --num-shards 1 --jobs -2  # train the grid
    python -m trainer.select_best  # pick best -> models/production/best.joblib

## Predict

Rank repos by predicted stars (needs a token):

    python -m predictor.app facebook/react torvalds/linux tinygrad/tinygrad

For a real run from scratch: crawl, then build/train/select (same commands as the offline block, without demo_seed), then predict.

## Run in Docker

Same steps, in containers with CPU and RAM caps:

    docker compose build
    docker compose run --rm trainer python -m trainer.train --num-shards 1 --jobs 1
    docker compose run --rm trainer python -m trainer.select_best
    docker compose run --rm predictor python -m predictor.app facebook/react
