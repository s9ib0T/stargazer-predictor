#!/bin/bash
# horizontal scaling: same grid, more worker containers, each pinned to 1 cpu
# measures wall clock vs number of workers
#
# needs the trainer image:
#   docker build -t stargazer-trainer -f infra/docker/Dockerfile.trainer .
# and built data in data/processed/.
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

source "$root/experiments/lib.sh"

IMAGE="${TRAINER_IMAGE:-stargazer-trainer}"
check_image_fresh "$IMAGE" "$root"
WORKERS="${WORKERS:-1 2 4}"    # matches CPU_LIST in run_vertical.sh so the two plots line up
MEM="${MEM:-1g}"
MAX_JOBS="${MAX_JOBS:-30}"    # sample the grid down so a run doesnt take an hour (full grid 114 jobs)
                              # sampled before sharding, so every worker count does the same total work
                              # set MAX_JOBS=0 for the full grid
CSV="experiments/horizontal.csv"
echo "workers,seconds" > "$CSV"

for n in $WORKERS; do
  echo "==> $n worker(s), 1 cpu each"
  rm -f models/trials/timing_shard*.json
  start=$(date +%s.%N)
  pids=()
  for ((i=0; i<n; i++)); do
    docker run --rm --cpus=1 --memory="$MEM" \
      -e PYTHONUNBUFFERED=1 \
      -v "$root/data:/app/data" -v "$root/models:/app/models" \
      "$IMAGE" python -m trainer.train --shard-index "$i" --num-shards "$n" --jobs 1 --max-jobs "$MAX_JOBS" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  end=$(date +%s.%N)
  secs=$(echo "$end - $start" | bc)
  echo "$n,$secs" >> "$CSV"
  echo "    took ${secs}s"
done

echo
echo "wrote $CSV. make plots with: python experiments/plots.py"
