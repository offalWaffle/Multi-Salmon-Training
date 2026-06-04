#!/usr/bin/env bash
# provision.sh  — run on your Mac.
# Find a cheap, reliable single-GPU offer and launch a vast.ai instance.
# Pass an explicit offer id to skip auto-selection:  cloud/provision.sh 1234567
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/.env" ] && source "$HERE/.env"

IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
DISK="${VAST_DISK:-40}"
QUERY="${VAST_QUERY:-reliability>0.98 num_gpus=1 gpu_name in [RTX_3090,RTX_4090] inet_down>200 disk_space>${DISK} rentable=true}"

command -v vastai >/dev/null || { echo "vastai CLI not found — install with: pip install vastai"; exit 1; }

echo "==> Top offers for: $QUERY"
vastai search offers "$QUERY" -o 'dph' | head -n 15
echo

# --cancel-unavail makes create fail fast when an offer can't actually be scheduled,
# instead of silently leaving a stuck "loading"/stopped instance that never runs.
create_on_offer() {
  oid="$1"
  echo "==> Trying offer $oid (image=$IMAGE disk=${DISK}GB)…"
  out="$(vastai create instance "$oid" \
    --image "$IMAGE" --disk "$DISK" --ssh --direct \
    --cancel-unavail --label mst-dac-adapter 2>&1)" || true
  echo "$out"
  echo "$out" | grep -qiE "success.*true|new_contract"
}

OFFER_ID="${1:-}"
if [ -n "$OFFER_ID" ]; then
  create_on_offer "$OFFER_ID" || { echo "Offer $OFFER_ID could not be scheduled."; exit 1; }
else
  OFFERS="$(vastai search offers "$QUERY" -o 'dph' --raw \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); print(" ".join(str(o["id"]) for o in r[:8]))')"
  [ -z "$OFFERS" ] && { echo "No offers matched. Loosen VAST_QUERY in cloud/.env."; exit 1; }
  scheduled=0
  for oid in $OFFERS; do
    if create_on_offer "$oid"; then scheduled=1; break; fi
    echo "    offer $oid unschedulable, trying next…"
  done
  [ "$scheduled" -eq 0 ] && { echo "All top offers were unschedulable right now. Re-run to refresh offers, or loosen VAST_QUERY in cloud/.env."; exit 1; }
fi

echo
echo "Instance launching. Track it:        vastai show instances"
echo "When status is 'running', deploy:    cloud/deploy.sh \$(vastai ssh-url <INSTANCE_ID>)"