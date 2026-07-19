import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from common.github_graphql import GitHubGraphQL
from common.features import make_features

PROD = Path("models/production")
PROCESSED = Path("data/processed")


def load_artifacts():
    # model + the column schema it was trained with
    model_path = PROD / "best.joblib"
    schema_path = PROCESSED / "feature_schema.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model missing at {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"schema missing at {schema_path}")
    model = joblib.load(model_path)
    schema = json.loads(schema_path.read_text())
    return model, schema["feature_columns"], schema


def fetch_features(repo_names, with_contributors=False):
    # same graphql path as the crawler -> feature parity with training
    client = GitHubGraphQL()
    rows, valid = [], []
    print("fetching repo data from github...")
    for name in repo_names:
        try:
            repo = client.get_repo(name)
            if with_contributors:
                # model was trained with real contributor counts, fetch them here too
                try:
                    repo["contributor_count"] = client.contributor_count(name)
                except Exception:
                    repo["contributor_count"] = 0
            rows.append(make_features(repo))
            valid.append(name)
            print(f"  [ok]   {name}")
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    if not rows:
        raise RuntimeError("no valid repos")
    return rows, valid


def prepare(rows, cols, schema):
    # features -> dataframe, one-hot, align to the training columns
    df = pd.DataFrame(rows)
    df["language"] = df["language"].where(df["language"].isin(schema["top_langs"]), "other")
    df["license_key"] = df["license_key"].where(df["license_key"].isin(schema["top_licenses"]), "other")
    df = pd.get_dummies(df, columns=["language", "license_key"], dtype=int)
    return df.reindex(columns=cols, fill_value=0)


def predict(model, df, valid):
    # model outputs log stars, invert and sort desc
    stars = np.expm1(model.predict(df)).clip(min=0)
    out = list(zip(valid, stars))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def rank_repos(repo_names, artifacts=None):
    # whole pipeline
    if artifacts is None:
        artifacts = load_artifacts()
    model, cols, schema = artifacts
    rows, valid = fetch_features(repo_names, schema.get("uses_contributors", False))
    df = prepare(rows, cols, schema)
    return predict(model, df, valid)


def print_results(results):
    # aligned table with a small proportional bar per repo
    star_header = "predicted stars"
    star_strs = [f"{int(s):,}" for _, s in results]
    repo_w = min(max((len(r) for r, _ in results), default=10), 40)
    star_w = max(max((len(x) for x in star_strs), default=5), len(star_header))
    bar_w = 18
    top = max((s for _, s in results), default=1) or 1

    width = 10 + repo_w + star_w + bar_w
    sep = "─" * width
    print()
    print("  GitHub star prediction")
    print(sep)
    print(f"  {'#':>2}  {'repository':<{repo_w}}  {star_header:>{star_w + 5}}")
    print(sep)
    for i, ((repo, stars), sstr) in enumerate(zip(results, star_strs), 1):
        name = repo if len(repo) <= repo_w else repo[: repo_w - 1] + "…"
        bar = "█" * max(1, round(bar_w * stars / top))
        print(f"  {i:>2}  {name:<{repo_w}}  {sstr:>{star_w}}  {bar}")
    print(sep)
    print()


def main():
    if len(sys.argv) < 2:
        print("usage: python -m predictor.app <repo1> <repo2> ...")
        sys.exit(1)
    try:
        results = rank_repos(sys.argv[1:])
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)
    print_results(results)


if __name__ == "__main__":
    main()
