#!/usr/bin/env bash
# fetch.sh  — run on your Mac.
# Pull trained checkpoints from B2 back into the local repo.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
[ -f "$HERE/.env" ] && source "$HERE/.env"
: "${B2_KEY_ID:?set B2_KEY_ID in cloud/.env}"
: "${B2_APP_KEY:?set B2_APP_KEY in cloud/.env}"
: "${B2_BUCKET:?set B2_BUCKET in cloud/.env}"

export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="$B2_KEY_ID"
export RCLONE_CONFIG_B2_KEY="$B2_APP_KEY"

DEST="$ROOT/checkpoints/dac_adapter"
mkdir -p "$DEST"
echo "==> Pulling b2:$B2_BUCKET/checkpoints/dac_adapter -> $DEST"
rclone copy "b2:$B2_BUCKET/checkpoints/dac_adapter" "$DEST" --progress

echo
echo "Done. Local checkpoints:"
ls -lh "$DEST"
echo
echo "Remember to destroy the instance when finished:  vastai destroy instance <INSTANCE_ID>"