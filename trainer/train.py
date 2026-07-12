"""
the training worker, runs one shard of the grid

the grid is the same fixed list on every worker (see grid.py), so each worker just takes its slice jobs[i::k] and runs it
results go one file per job into the shared trials dir, no talking between workers

examples:
    # everything on one machine, all cores
    python -m trainer.train --num-shards 1 --jobs -1

    # worker 0 of 3
    python -m trainer.train --shard-index 0 --num-shards 3 --jobs 1
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score

from trainer.grid import (
    DATA_DIR, MODELS_DIR, MODELS, build_jobs, config_id, create_model, load_data,
)


def write_json_atomic(path, payload):
    # temp file + rename so a reader never sees a half written file
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def run_one(cfg, X_train, y_log):
    # 5-fold CV on the log target, returns the score record (no model kept)
    model = create_model(cfg["model_name"], cfg)
    scores = cross_val_score(model, X_train, y_log, cv=5, scoring="r2", n_jobs=1)
    return {"model_name": cfg["model_name"], "config": cfg, "cv_r2": float(scores.mean())}


def parse_args():
    p = argparse.ArgumentParser(description="sharded grid search worker")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1, help="1 = run everything")
    p.add_argument("--model", choices=MODELS, help="only this model (default all)")
    p.add_argument("--jobs", type=int, default=1, help="cores on this worker, -1 = all")
    p.add_argument("--max-jobs", type=int, default=0, help="cap jobs (quick/vertical run)")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(MODELS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"shard-index must be in [0,{args.num_shards})")

    trials = Path(args.out_dir) / "trials"
    trials.mkdir(parents=True, exist_ok=True)

    print(f"loading data from {args.data_dir}")
    X_train, _, y_train, _ = load_data(args.data_dir)
    y_log = np.log1p(y_train)  # train in log space

    all_jobs = build_jobs([args.model] if args.model else None)
    my_jobs = all_jobs[args.shard_index::args.num_shards]  # every Nth job
    if args.max_jobs:
        my_jobs = my_jobs[:args.max_jobs]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(my_jobs)}/{len(all_jobs)} configs, jobs={args.jobs}")

    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(delayed(run_one)(c, X_train, y_log) for c in my_jobs)
    for r in res:
        write_json_atomic(trials / f"trial_{config_id(r['config'])}.json", r)
        print(f"  {r['model_name']:<22} cv_r2={r['cv_r2']:.4f}")
    elapsed = time.time() - t0

    # timing file for the scaling plot
    write_json_atomic(trials / f"timing_shard{args.shard_index}.json", {
        "host": socket.gethostname(),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_configs": len(my_jobs),
        "jobs": args.jobs,
        "seconds": round(elapsed, 2),
    })
    print(f"shard {args.shard_index} done in {elapsed:.2f}s, wrote {len(my_jobs)} trials")


if __name__ == "__main__":
    main()
