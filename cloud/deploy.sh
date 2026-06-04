#!/usr/bin/env bash
# deploy.sh  — run on your Mac, once the instance is 'running'.
# Bootstraps the instance: installs deps and pulls code + latents from B2.
# Usage:
#   cloud/deploy.sh ssh://root@HOST:PORT      # paste the output of `vastai ssh-url <ID>`
#   cloud/deploy.sh HOST PORT
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/.env" ] && source "$HERE/.env"
: "${B2_KEY_ID:?set B2_KEY_ID in cloud/.env}"
: "${B2_APP_KEY:?set B2_APP_KEY in cloud/.env}"
: "${B2_BUCKET:?set B2_BUCKET in cloud/.env}"

# Parse either an ssh:// url or "HOST PORT".
if [[ "${1:-}" =~ ^ssh://([^@]+)@([^:]+):([0-9]+) ]]; then
  USER="${BASH_REMATCH[1]}"; HOST="${BASH_REMATCH[2]}"; PORT="${BASH_REMATCH[3]}"
elif [ -n "${2:-}" ]; then
  USER=root; HOST="$1"; PORT="$2"
else
  echo "Usage: cloud/deploy.sh ssh://root@HOST:PORT   |   cloud/deploy.sh HOST PORT"; exit 1
fi

SSH="ssh -o StrictHostKeyChecking=accept-new -p $PORT $USER@$HOST"
echo "==> Target: $USER@$HOST:$PORT"

echo "==> Copying bootstrap script…"
scp -o StrictHostKeyChecking=accept-new -P "$PORT" "$HERE/bootstrap.sh" "$USER@$HOST:/root/bootstrap.sh"

echo "==> Running bootstrap (install deps, pull code + latents)…"
$SSH "B2_KEY_ID='$B2_KEY_ID' B2_APP_KEY='$B2_APP_KEY' B2_BUCKET='$B2_BUCKET' bash /root/bootstrap.sh"

echo
echo "Deployed. Start training:"
echo "  $SSH 'bash /root/mst/cloud/train.sh'"
echo "Resume from a checkpoint:"
echo "  $SSH 'bash /root/mst/cloud/train.sh checkpoints/dac_adapter/best_model.pt'"