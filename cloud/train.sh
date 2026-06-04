#!/usr/bin/env bash
# train.sh  — runs ON the instance.
# Launches DAC pitch-adapter training in a detached tmux session and pushes
# checkpoints to B2 every 5 min (so progress survives if the instance dies).
# Optional arg: a checkpoint path to resume from.
set -euo pipefail

cd /root/mst
[ -f /root/.b2.env ] && source /root/.b2.env

RESUME="${1:-}"
ARGS="--config config/dac_adapter_config.yaml --device cuda"
[ -n "$RESUME" ] && ARGS="$ARGS --resume $RESUME"

# Background checkpoint syncer -> B2 (best-effort; skipped if no creds).
if [ -n "${B2_BUCKET:-}" ]; then
  export RCLONE_CONFIG_B2_TYPE=b2
  export RCLONE_CONFIG_B2_ACCOUNT="$B2_KEY_ID"
  export RCLONE_CONFIG_B2_KEY="$B2_APP_KEY"
  cat > /root/sync_ckpts.sh <<'EOS'
#!/usr/bin/env bash
source /root/.b2.env
export RCLONE_CONFIG_B2_TYPE=b2 RCLONE_CONFIG_B2_ACCOUNT="$B2_KEY_ID" RCLONE_CONFIG_B2_KEY="$B2_APP_KEY"
while true; do
  rclone copy /root/mst/checkpoints/dac_adapter "b2:$B2_BUCKET/checkpoints/dac_adapter" --quiet 2>/dev/null || true
  sleep 300
done
EOS
  chmod +x /root/sync_ckpts.sh
  pkill -f sync_ckpts.sh 2>/dev/null || true
  nohup /root/sync_ckpts.sh >/root/sync_ckpts.log 2>&1 &
  echo "==> Checkpoint syncer -> b2:$B2_BUCKET/checkpoints/dac_adapter (every 5 min)"
fi

command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)

echo "==> Launching training in tmux session 'train'…"
tmux new-session -d -s train \
  "python scripts/train_dac_adapter.py $ARGS 2>&1 | tee /root/mst/train.out; echo EXIT=\$?; exec bash"

echo
echo "Training started. Useful commands:"
echo "  tmux attach -t train       # watch live (Ctrl-b then d to detach)"
echo "  tail -f /root/mst/train.out"
echo "  nvidia-smi                 # GPU utilisation"
echo "When done, fetch checkpoints to your Mac with:  cloud/fetch.sh"