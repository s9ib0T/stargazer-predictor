#!/bin/bash
# build dataset, train the full grid on all cores, pick the best model
set -e
cd "$(dirname "$0")/.."
python -m trainer.build_dataset
python -m trainer.train --num-shards 1 --jobs -2
python -m trainer.select_best
