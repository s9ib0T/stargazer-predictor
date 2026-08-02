"""
shared for training: model grid, model builder, job list, data loading, metrics
train.py and select_best.py both import from here

note: models are trained on log1p(stars), not raw star count
Stars are skewed -> train in log space for balanced accuracy
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

MODELS = [
    "LinearRegression",
    "Ridge",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
]

GRIDS = {
    "LinearRegression": {},
    "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTrees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
    "HistGradientBoosting": {
        "max_iter": [200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [None, 4, 8],
    },
}


def create_model(name, cfg):
    # build the sklearn model from name + param dict
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth"),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            random_state=68, n_jobs=1,
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth"),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            random_state=68, n_jobs=1,
        )
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=cfg.get("n_estimators", 200),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth", 3),
            random_state=68,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=cfg.get("max_iter", 200),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_depth=cfg.get("max_depth"),
            random_state=68,
        )
    # linear models get scaled features first
    # disk_usage runs to 5e7 while the one-hot columns are 0/1, which makes X'X badly conditioned
    # sklearn then warns on every Ridge fit, and alpha is far too small to regularize anything
    # trees do not care about scale, so they stay as they are
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=cfg.get("alpha", 1.0)))
    return make_pipeline(StandardScaler(), LinearRegression())


def build_jobs(models=None):
    # flatten the grids into one list of configs (every combination)
    models = models or MODELS
    jobs = []
    for name in models:
        grid = GRIDS[name]
        if not grid:
            jobs.append({"model_name": name})  # no params, one job
            continue
        keys = sorted(grid)  # sorted -> same order on every worker
        for combo in itertools.product(*(grid[k] for k in keys)):
            c = {"model_name": name}
            c.update(dict(zip(keys, combo)))
            jobs.append(c)
    return jobs


def job_weight(cfg):
    # rough relative cost of one config, only used to spread work evenly over shards
    # number of trees dominates the runtime, everything else is noise next to it
    per_model = {
        "RandomForest": 1.0,
        "ExtraTrees": 0.4,
        "GradientBoosting": 0.4,
        "HistGradientBoosting": 0.05,
    }
    trees = cfg.get("n_estimators", cfg.get("max_iter", 1))
    return per_model.get(cfg["model_name"], 0.001) * trees


def config_id(cfg):
    # short stable id, used as the trial filename
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return f"{cfg['model_name']}_{h}"


def load_data(data_dir=DATA_DIR):
    # parquet from build_dataset
    # returns RAW star targets, caller logs them
    data_dir = Path(data_dir)
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")["stargazers_count"]
    y_test = pd.read_parquet(data_dir / "y_test.parquet")["stargazers_count"]
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test_raw):
    # model was fit on log target
    # invert preds for star-space numbers
    log_pred = model.predict(X_test)
    star_pred = np.expm1(log_pred).clip(min=0)
    log_true = np.log1p(y_test_raw)
    # median absolute percent error, not thrown off by the huge repos
    medape = float(np.median(np.abs(star_pred - y_test_raw) / np.maximum(y_test_raw, 1)) * 100)
    return {
        "r2_log": float(r2_score(log_true, log_pred)),
        "r2_star": float(r2_score(y_test_raw, star_pred)),
        "mae_star": float(mean_absolute_error(y_test_raw, star_pred)),
        "rmse_star": float(root_mean_squared_error(y_test_raw, star_pred)),
        "median_ape": medape,
    }
