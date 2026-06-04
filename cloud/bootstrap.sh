#!/usr/bin/env bash
# bootstrap.sh  — runs ON the instance (invoked by deploy.sh).
# Installs rclone + Python deps, pulls code + latents from B2, persists B2 creds
# so train.sh can push checkpoints back.
set -euo pipefail

: "${B2_KEY_ID:?B2_KEY_ID not passed}"
: "${B2_APP_KEY:?B2_APP_KEY not passed}"
: "${B2_BUCKET:?B2_BUCKET not passed}"

export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="$B2_KEY_ID"
export RCLONE_CONFIG_B2_KEY="$B2_APP_KEY"

echo "==> Installing rclone…"
if ! command -v rclone >/dev/null; then
  curl -fsSL https://rclone.org/install.sh | bash || (apt-get update -qq && apt-get install -y -qq rclone)
fi

echo "==> Pulling code…"
mkdir -p /root/mst
rclone copyto "b2:$B2_BUCKET/code.tar.gz" /root/code.tar.gz --progress
tar -xzf /root/code.tar.gz -C /root/mst

echo "==> Pulling latents -> /root/mst/data/vst_latents …"
mkdir -p /root/mst/data/vst_latents
rclone copy "b2:$B2_BUCKET/vst_latents" /root/mst/data/vst_latents --progress --transfers=16 --fast-list

# Persist creds (0600) so train.sh can sync checkpoints to B2.
cat > /root/.b2.env <<EOF
B2_KEY_ID=$B2_KEY_ID
B2_APP_KEY=$B2_APP_KEY
B2_BUCKET=$B2_BUCKET
EOF
chmod 600 /root/.b2.env

echo "==> Installing Python deps (torch/torchaudio come from the image)…"
cd /root/mst
pip install -q -r cloud/requirements-cloud.txt

echo "==> Sanity check…"
python -c "import torch, dac; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

TRAIN=$(ls data/vst_latents/train/sample_*.pt 2>/dev/null | wc -l)
VAL=$(ls data/vst_latents/val/sample_*.pt 2>/dev/null | wc -l)
echo "==> Latents ready: $TRAIN train / $VAL val samples"
echo "Bootstrap complete. Start training:  bash /root/mst/cloud/train.sh"