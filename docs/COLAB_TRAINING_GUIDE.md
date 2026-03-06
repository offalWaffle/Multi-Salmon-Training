# Google Colab Training Guide

Train the VQ-VAE on a cloud GPU to overcome Mac MPS memory limits.

**What you get on Colab (free T4):**
- 15 GB VRAM → `hidden_dim=128`, `num_embeddings=1024`, `batch_size=16`
- ~5–10× faster training than Mac MPS
- 4-second audio clips instead of 2-second

---

## Before You Start (on your Mac)

### 1. Push code to GitHub

Create a private GitHub repo and push the project. Keep audio data out of git.

```bash
cd /Users/laurentclerc/PycharmProjects/Multi-Salmon-Training

# If not already a git repo:
git init
echo "data/raw/" >> .gitignore
echo "data/vqvae_piano/" >> .gitignore
echo "checkpoints/" >> .gitignore
echo "outputs/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/Multi-Salmon-Training.git
git push -u origin main
```

### 2. Upload audio data to Google Drive

Install the [Google Drive desktop app](https://www.google.com/drive/download/) if you haven't already — it's much faster than uploading via browser.

Copy your raw audio into Drive. Keep the folder structure intact:

```
My Drive/
└── mst-data/
    └── raw/
        ├── equator/
        └── equator2/
```

Also upload the catalog:
```
My Drive/
└── mst-data/
    └── processed/
        └── sample_catalog.csv
```

> **How long does upload take?** Depends on library size. The equator/equator2 libraries are
> typically 5–30 GB. On a typical home connection this can take 1–8 hours. Start it before bed.

---

## Setting Up the Colab Notebook

### 3. Create a new notebook and set runtime to GPU

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → T4 GPU** (free) or A100 (Colab Pro)
4. Click **Connect**

Verify GPU is available:
```python
import torch
print(torch.cuda.is_available())        # should print True
print(torch.cuda.get_device_name(0))    # should print something like "Tesla T4"
```

---

### 4. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 5. Clone your code

```python
# Replace with your actual repo URL
!git clone https://github.com/YOUR_USERNAME/Multi-Salmon-Training.git /content/mst
%cd /content/mst
```

---

### 6. Install dependencies

```python
!pip install -q \
    torch torchaudio \
    numpy scipy \
    librosa soundfile \
    pyyaml tqdm \
    matplotlib einops \
    omegaconf pandas
```

> `wandb`, `sf2utils` are optional — skip them unless you need experiment tracking or SF2 export.

---

### 7. Copy audio data from Drive to local storage

Colab's `/content/` is fast local SSD. Training directly from Drive is significantly slower.
**Always copy to `/content/` first.**

```python
import shutil
from pathlib import Path

drive_raw   = Path('/content/drive/MyDrive/mst-data/raw')
drive_csv   = Path('/content/drive/MyDrive/mst-data/processed/sample_catalog.csv')
local_raw   = Path('/content/mst/data/raw')
local_proc  = Path('/content/mst/data/processed')

local_raw.mkdir(parents=True, exist_ok=True)
local_proc.mkdir(parents=True, exist_ok=True)

print("Copying audio data... (this may take a few minutes)")
shutil.copytree(str(drive_raw), str(local_raw), dirs_exist_ok=True)
shutil.copy(str(drive_csv), str(local_proc / 'sample_catalog.csv'))
print("Done.")
```

---

### 8. Fix catalog paths

The `sample_catalog.csv` has absolute Mac paths like
`/Users/laurentclerc/PycharmProjects/Multi-Salmon-Training/data/raw/...`.
These need to point to `/content/mst/data/raw/...`.

```python
import pandas as pd

catalog_path = Path('/content/mst/data/processed/sample_catalog.csv')
df = pd.read_csv(catalog_path)

# Replace Mac prefix with Colab prefix
old_prefix = '/Users/laurentclerc/PycharmProjects/Multi-Salmon-Training/data/raw'
new_prefix = '/content/mst/data/raw'

df['path'] = df['path'].str.replace(old_prefix, new_prefix, regex=False)

df.to_csv(catalog_path, index=False)
print(f"Updated {len(df)} paths in catalog.")
print("Sample path:", df['path'].iloc[0])
```

---

### 9. Regenerate train/val/test splits

This creates new JSON split files with the correct Colab paths.

```python
!python scripts/prepare_vqvae_data.py --instrument "Baby Grand Piano"
```

Verify the splits were created:
```python
import json
for split in ['train', 'val', 'test']:
    with open(f'data/vqvae_piano/{split}.json') as f:
        data = json.load(f)
    print(f"{split}: {len(data)} samples — first path: {data[0]['path']}")
```

---

### 10. Update the training config for cloud GPU

```python
import yaml

config_path = '/content/mst/config/vqvae_config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

# Model — bigger capacity
config['model']['hidden_dim']      = 128    # was 64
config['model']['num_embeddings']  = 1024   # was 512

# Training — larger batches
config['training']['batch_size']   = 16     # was 4
config['training']['num_epochs']   = 100    # was 50

# Audio — longer clips
config['audio']['duration']        = 4.0    # was 2.0

# Data — enable parallel workers now that we're on CUDA
config['data']['num_workers']      = 4      # was 0
config['data']['pin_memory']       = True   # was false

# Save checkpoints to Drive so they survive session disconnects
config['checkpoint_dir'] = '/content/drive/MyDrive/mst-checkpoints/vqvae'
config['log_dir']        = '/content/drive/MyDrive/mst-checkpoints/vqvae/logs'

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Config updated.")
```

---

### 11. Train

```python
!python scripts/train_vqvae.py --config config/vqvae_config.yaml
```

You should see CUDA picked up automatically:
```
Device: cuda
Starting VQ-VAE Training
...
```

---

## Handling Disconnections

Colab free sessions disconnect after ~90 minutes of inactivity and have a ~12-hour max runtime.

**Checkpoints are saved to Drive** (step 10 sets `checkpoint_dir` to Drive), so your progress
is safe. When you reconnect, repeat steps 3–7 (clone, install, copy data, fix paths) and then
resume from the last checkpoint:

```python
!python scripts/train_vqvae.py \
    --config config/vqvae_config.yaml \
    --resume /content/drive/MyDrive/mst-checkpoints/vqvae/best_model.pt
```

> **Tip:** Keep the browser tab active or use a Colab Pro subscription to avoid the inactivity
> timeout. You can also use [this keep-alive trick](https://stackoverflow.com/a/57721546):
> open the browser console (F12) and run:
> ```javascript
> function KeepAlive() { document.querySelector("colab-connect-button").click(); }
> setInterval(KeepAlive, 60000);
> ```

---

## After Training: Copy Results Back to Mac

Once training is done, download the best checkpoint from Drive normally via the Drive web UI,
or `rsync` it if you have a static IP on the cloud instance.

Then update your local config back to MPS settings before running locally:

```yaml
# config/vqvae_config.yaml — restore for local use
model:
  hidden_dim: 128        # keep the improved model size
  num_embeddings: 1024

data:
  num_workers: 0         # back to 0 for MPS
  pin_memory: false
```

---

## Quick Reference: Cell Execution Order

For a fresh session after a disconnect:

| Step | Cell |
|------|------|
| 1 | Mount Drive |
| 2 | Clone code |
| 3 | Install deps |
| 4 | Copy data from Drive |
| 5 | Fix catalog paths |
| 6 | Regenerate splits |
| 7 | Resume training with `--resume` |

Steps 1–6 take about 5–10 minutes depending on data size.

---

## Troubleshooting

**`CUDA out of memory`**
Reduce `batch_size` to 8, or `hidden_dim` back to 64. Check VRAM usage with:
```python
!nvidia-smi
```

**`No such file or directory: data/vqvae_piano/train.json`**
You're running from the wrong directory. Run `%cd /content/mst` first.

**`path not found` errors during training**
The catalog path fix (step 8) didn't apply correctly. Re-run step 8 and check the sample path
printed at the end matches `/content/mst/data/raw/...`.

**Session disconnected mid-training**
Normal on free Colab. Repeat steps 1–7 and resume with `--resume`. Your best checkpoint is
safe on Drive.
