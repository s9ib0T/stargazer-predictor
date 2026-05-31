# GitHub Stargazer Predictor

Predicts how many stars a GitHub repo has from its activity data (forks, watchers, commits, issues, releases, age, language, etc).
Give it a few repos and it ranks them by predicted stars.

Everything runs locally in Docker containers with hard CPU and RAM caps.

## Setup

Needs Python 3.10+ and a GitHub token.

    cp .env.example .env      # put your token in it
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt