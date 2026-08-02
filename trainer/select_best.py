"""
the reduce step, runs once after all workers are done

reads every trial file, 
picks the best config per model by CV score, 
refits it on the full train set (log target), 
scores on the test set in both log and star space, 
and copies the overall best to models/production/best.joblib
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from trainer.grid import DATA_DIR, MODELS_DIR, create_model, evaluate, load_data


def load_trials(trials_dir):
    files = sorted(Path(trials_dir).glob("trial_*.json"))
    if not files:
        raise SystemExit(f"no trials in {trials_dir}, run trainer.train first")
    return [json.loads(f.read_text()) for f in files]


def best_per_model(records):
    # keep the highest cv score per model name
    best = {}
    for r in records:
        n = r["model_name"]
        if n not in best or r["cv_r2"] > best[n]["cv_r2"]:
            best[n] = r
    return best


def parse_args():
    p = argparse.ArgumentParser(description="reduce trials, pick best model")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    prod = models_dir / "production"

    records = load_trials(models_dir / "trials")
    print(f"read {len(records)} trials")

    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    y_log = np.log1p(y_train)
    winners = best_per_model(records)

    summary = {}
    fitted = {}
    for name, r in winners.items():
        m = create_model(name, r["config"])
        m.fit(X_train, y_log)  # log target
        metrics = evaluate(m, X_test, y_test)
        fitted[name] = m
        summary[name] = {**metrics, "cv_r2": r["cv_r2"], "best_hyperparameters": r["config"]}
        print(f"{name:<22} cv_r2={r['cv_r2']:.4f} test_r2_log={metrics['r2_log']:.4f} "
              f"mae_star={metrics['mae_star']:.0f}")

    (models_dir / "results.json").write_text(json.dumps(summary, indent=2))

    # judged on the cv score
    # cv_r2 is log-space r2, fair across magnitudes
    best_name = max(summary, key=lambda n: summary[n]["cv_r2"])
    print(f"\nbest: {best_name} (cv_r2={summary[best_name]['cv_r2']:.4f}, "
          f"held-out test r2_log={summary[best_name]['r2_log']:.4f})")

    prod.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], prod / "best.joblib")

    meta = {
        "model_name": best_name,
        "log_target": True,
        "deployment_time": datetime.now(timezone.utc).isoformat(),
        "metrics": summary[best_name],
    }
    (prod / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved model + metadata to {prod}")


if __name__ == "__main__":
    main()
