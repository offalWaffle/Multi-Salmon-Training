#!/usr/bin/env bash
# upload.sh  — run on your Mac.
# Stage code + DAC latents to Backblaze B2 so a vast.ai instance can pull them.
# Re-run any time to refresh; rclone skips unchanged files.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
[ -f "$HERE/.env" ] && source "$HERE/.env"

: "${B2_KEY_ID:?set B2_KEY_ID in cloud/.env (see cloud/.env.example)}"
: "${B2_APP_KEY:?set B2_APP_KEY in cloud/.env}"
: "${B2_BUCKET:?set B2_BUCKET in cloud/.env}"

command -v rclone >/dev/null || { echo "rclone not found — install with: brew install rclone"; exit 1; }

# Configure an ephemeral 'b2' remote purely from env vars (no rclone config file needed).
export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="$B2_KEY_ID"
export RCLONE_CONFIG_B2_KEY="$B2_APP_KEY"

LATENTS="$ROOT/data/vst_latents"
[ -d "$LATENTS" ] || { echo "Missing $LATENTS — nothing to upload."; exit 1; }

echo "==> Packaging code…"
TARBALL="$(mktemp -t mst-code-XXXX).tar.gz"
# Tar an explicit list of code paths. Do NOT switch to whole-dir + name excludes:
# macOS bsdtar matches --exclude names unanchored, so --exclude=data also drops src/data.
tar --exclude-vcs --exclude='*.pyc' --exclude='__pycache__' --exclude='.DS_Store' \
    -czf "$TARBALL" -C "$ROOT" \
    src scripts config cloud tests setup.py requirements.txt README.md
echo "    code.tar.gz = $(du -h "$TARBALL" | cut -f1)"

echo "==> Uploading code.tar.gz -> b2:$B2_BUCKET/code.tar.gz"
# --no-check-dest: always overwrite, and avoids a 403 from copyto's HEAD-on-existing check.
rclone copyto "$TARBALL" "b2:$B2_BUCKET/code.tar.gz" --no-check-dest --progress
rm -f "$TARBALL"

echo "==> Uploading latents -> b2:$B2_BUCKET/vst_latents  (~$(du -sh "$LATENTS" | cut -f1))"
rclone copy "$LATENTS" "b2:$B2_BUCKET/vst_latents" --progress --transfers=16 --fast-list

echo
echo "Done. Code + latents staged in b2://$B2_BUCKET"
echo "Next:  cloud/provision.sh"