"""
make the figures for the report from whatever outputs exist:

experiments/horizontal.csv  ->  scaling.png (time + speedup)
experiments/vertical.csv    ->  vertical_scaling.png
models/results.json         ->  model_accuracy.png
best model + test data      ->  pred_vs_actual.png

missing inputs are just skipped
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def horizontal():
    p = HERE / "horizontal.csv"
    if not p.exists():
        return
    rows = read_csv(p)
    n = [r["workers"] for r in rows]
    secs = [r["seconds"] for r in rows]
    base = secs[0] * n[0]  # rough single-worker total
    speedup = [base / s for s in secs]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(n, secs, "o-")
    a.set_xlabel("workers")
    a.set_ylabel("seconds")
    a.set_title("training time vs workers")
    b.plot(n, speedup, "o-", label="measured")
    b.plot(n, n, "--", color="gray", label="ideal")
    b.set_xlabel("workers")
    b.set_ylabel("speedup")
    b.set_title("speedup")
    b.legend()
    fig.tight_layout()
    fig.savefig(FIG / "scaling.png", dpi=120)
    print("wrote scaling.png")


def vertical():
    p = HERE / "vertical.csv"
    if not p.exists():
        return
    rows = read_csv(p)
    c = [r["cpus"] for r in rows]
    secs = [r["seconds"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(c, secs, "o-")
    plt.xlabel("cpus")
    plt.ylabel("seconds")
    plt.title("vertical scaling (one worker)")
    plt.tight_layout()
    plt.savefig(FIG / "vertical_scaling.png", dpi=120)
    print("wrote vertical_scaling.png")


def accuracy():
    p = Path("models/results.json")
    if not p.exists():
        return
    res = json.loads(p.read_text())
    names = list(res)
    r2 = [res[n]["r2_log"] for n in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, r2)
    plt.ylabel("test R2 (log space)")
    plt.title("model accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "model_accuracy.png", dpi=120)
    print("wrote model_accuracy.png")


def pred_vs_actual():
    model_path = Path("models/production/best.joblib")
    data = Path("data/processed")
    if not model_path.exists() or not (data / "X_test.parquet").exists():
        return
    import joblib
    import pandas as pd
    model = joblib.load(model_path)
    X_test = pd.read_parquet(data / "X_test.parquet")
    y_test = pd.read_parquet(data / "y_test.parquet")["stargazers_count"]
    pred = np.expm1(model.predict(X_test)).clip(min=1)
    plt.figure(figsize=(5, 5))
    plt.loglog(y_test.clip(lower=1), pred, ".", alpha=0.4)
    lim = [1, max(y_test.max(), pred.max())]
    plt.plot(lim, lim, "--", color="gray")
    plt.xlabel("actual stars")
    plt.ylabel("predicted stars")
    plt.title("predicted vs actual")
    plt.tight_layout()
    plt.savefig(FIG / "pred_vs_actual.png", dpi=120)
    print("wrote pred_vs_actual.png")


if __name__ == "__main__":
    horizontal()
    vertical()
    accuracy()
    pred_vs_actual()
    print(f"figures in {FIG}")
