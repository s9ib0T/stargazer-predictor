# GitHub Stargazer Predictor

![ci](https://github.com/s9ib0T/stargazer-predictor/actions/workflows/ci.yml/badge.svg)

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

`demo_seed.py` writes to the same path as the crawler, so it stops if `data/raw/repos.jsonl`
holds a real crawl. Use `--out` to write elsewhere, or `--force` to overwrite anyway.
A real crawl can always be rebuilt from `data/raw/shards` with `bash scripts/crawl.sh`.

## Predict

Rank repos by predicted stars (needs a token):

    python -m predictor.app facebook/react torvalds/linux tinygrad/tinygrad

For a real run from scratch: crawl, then build/train/select (same commands as the offline block, without demo_seed), then predict. Finish with `python experiments/plots.py`, otherwise the figures still show whatever data was trained on last.

## Run in Docker

Same steps, in containers with CPU and RAM caps:

    docker compose build
    docker compose run --rm trainer python -m trainer.train --num-shards 1 --jobs 1
    docker compose run --rm trainer python -m trainer.select_best
    docker compose run --rm predictor python -m predictor.app facebook/react

## CI

Every pull request runs the full pipeline on synthetic data and builds the three images:

    .github/workflows/ci.yml

No token needed, it trains on `demo_seed.py` output. The trained model from each run is downloadable from the Actions tab.

`main` is protected: no direct pushes, everything goes through a pull request with both checks green.

## Scaling experiments

    docker build -t stargazer-trainer -f infra/docker/Dockerfile.trainer .
    bash experiments/run_horizontal.sh
    bash experiments/run_vertical.sh
    python -m trainer.select_best  # two of the four figures read its output
    python experiments/plots.py  # figures in experiments/figures/

`plots.py` makes two figures from the timing csvs and two from the trained model, so run
it last, after whichever data you want the figures to reflect. The figures are generated
output and are not committed, same as `data/` and `models/`.


## Layout

    common/       shared github client + feature builder
    crawler/      sharded graphql crawler + merge
    trainer/      dataset build, sharded training, model selection
    predictor/    cli predictor
    scripts/      demo seed + crawl/train wrappers
    infra/        dockerfiles + git hooks
    experiments/  scaling scripts + plots
