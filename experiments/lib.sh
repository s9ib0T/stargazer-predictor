# shared helpers for the experiment scripts

# the trainer code is baked into the image with COPY, so editing trainer/ or common/
# changes nothing until the image is rebuilt
# easy to miss, the run just silently uses old code
check_image_fresh() {
  local image="$1" root="$2"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "image $image not found, build it first:" >&2
    echo "  docker build -t $image -f infra/docker/Dockerfile.trainer ." >&2
    exit 1
  fi
  local img_epoch src_epoch
  img_epoch=$(date -d "$(docker image inspect "$image" --format '{{.Created}}')" +%s)
  src_epoch=$(find "$root/trainer" "$root/common" -name '*.py' -printf '%T@\n' | sort -rn | head -1 | cut -d. -f1)
  if [ "$src_epoch" -gt "$img_epoch" ]; then
    echo "image $image is $(( (src_epoch - img_epoch) / 60 )) min older than the code in trainer/ or common/" >&2
    echo "rebuild first:" >&2
    echo "  docker build -t $image -f infra/docker/Dockerfile.trainer ." >&2
    exit 1
  fi
}
