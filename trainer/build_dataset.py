import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common.features import make_features

RAW = Path("data/raw/repos.jsonl")
OUT = Path("data/processed")
TOP_LANGS = 20   # keep this many languages, rest -> "other"
TOP_LICENSES = 15


def load_repos(path):
    repos = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            repos.append(json.loads(line))
    return repos


def top_values(series, n):
    return list(series.value_counts().head(n).index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    repos = load_repos(RAW)

    X = pd.DataFrame([make_features(r) for r in repos])
    y = pd.Series([r["stargazers_count"] for r in repos], name="stargazers_count")
    print(f"{len(X)} repos, {X.shape[1]} raw feature cols")

    top_l = top_values(X["language"], TOP_LANGS)
    top_lic = top_values(X["license_key"], TOP_LICENSES)
    X["language"] = X["language"].where(X["language"].isin(top_l), "other")
    X["license_key"] = X["license_key"].where(X["license_key"].isin(top_lic), "other")
    X = pd.get_dummies(X, columns=["language", "license_key"])
    X = X.astype({c: int for c in X.columns if X[c].dtype == bool})  # no bool cols

    n_missing = int(X.isna().sum().sum())
    if n_missing:
        print(f"warning: {n_missing} missing, filling 0")
    X = X.fillna(0)

    # stratify on star-rank buckets so train and test both span 50 -> max
    bucket = pd.qcut(y.rank(method="first"), q=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=bucket
    )
    print(f"train={len(X_train)} test={len(X_test)}")

    X_train.to_parquet(OUT / "X_train.parquet")
    X_test.to_parquet(OUT / "X_test.parquet")
    y_train.to_frame().to_parquet(OUT / "y_train.parquet")
    y_test.to_frame().to_parquet(OUT / "y_test.parquet")

    schema = {
        "target": "stargazers_count",
        "feature_columns": list(X_train.columns),  # exact cols + order models expect
        "top_langs": top_l + ["other"],
        "top_licenses": top_lic + ["other"],
        "uses_contributors": bool((X["contributor_count"] > 0).any()),  # true if the crawl ran --with-contributors
            # predictor reads this so it fetches contributor counts too, otherwise the feature is 0 at predict time
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    print(f"wrote {len(X_train.columns)} feature cols to {OUT}")


if __name__ == "__main__":
    main()
