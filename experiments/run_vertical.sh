#!/bin/bash
# vertical scaling: one worker, more cpus each run, same chunk of the grid
# same machine + more resources -> faster
# docker --cpus does the limiting
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

source "$root/experiments/lib.sh"

IMAGE="${TRAINER_IMAGE:-stargazer-trainer}"
check_image_fresh "$IMAGE" "$root"
CPU_LIST="${CPU_LIST:-1 2 4}"
MEM="${MEM:-2g}"
MAX_JOBS="${MAX_JOBS:-30}"    # sample grid down so it doesnt take forever (full grid 114 jobs)
CSV="experiments/vertical.csv"
echo "cpus,seconds" > "$CSV"

for c in $CPU_LIST; do
  echo "==> --cpus=$c (first $MAX_JOBS jobs)"
  rm -f models/trials/timing_shard0.json
  # --jobs matches cpu cap so more cpu actually gets used
  docker run --rm --cpus="$c" --memory="$MEM" \
    -e PYTHONUNBUFFERED=1 \
    -v "$root/data:/app/data" -v "$root/models:/app/models" \
    "$IMAGE" python -m trainer.train --num-shards 1 --jobs "$c" --max-jobs "$MAX_JOBS"
  secs=$(python3 -c "import json;print(json.load(open('models/trials/timing_shard0.json'))['seconds'])")
  echo "$c,$secs" >> "$CSV"
  echo "    took ${secs}s"
done

echo
echo "wrote $CSV. make plots with: python experiments/plots.py"
