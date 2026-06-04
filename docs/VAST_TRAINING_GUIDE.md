# vast.ai Training Guide — DAC Pitch Adapter

Train `scripts/train_dac_adapter.py` on a rented CUDA GPU. The adapter trains purely
on the precomputed latents in `data/vst_latents/` (1896 train / 216 val), so there's
**no raw audio, catalog, or split prep** — just stage the latents, rent a GPU, train,
pull checkpoints back.

Data flows through a **Backblaze B2** bucket so you can relaunch instances cheaply
without re-uploading from your Mac each time.

```
  Mac ──upload.sh──▶  B2 bucket  ◀──bootstrap.sh── vast.ai GPU
   ▲                  (code +                          │ train.sh
   └──fetch.sh────────  latents +  ◀──checkpoints──────┘
                        checkpoints)
```

The model is tiny (latent-space adapter, no DAC decoder in the gradient path), so any
single mid-range GPU (RTX 3090/4090) is plenty — no need for an A100.

---

## One-time setup (on your Mac)

### 1. Install the CLIs

```bash
pip install vastai          # vast.ai control
brew install rclone         # Backblaze B2 transfers
```

### 2. Create a vast.ai account + API key

1. Sign up at [cloud.vast.ai](https://cloud.vast.ai).
2. **Billing → Add Credit** (a few dollars is plenty; a 4090 is ~$0.30–0.50/hr and a
   full run here is well under an hour of GPU time).
3. **Account → copy your API key**, then register it with the CLI:
   ```bash
   
   vastai set api-key <YOUR_API_KEY>
   ```
   This stores the key in vast's own config (`~/.config/vastai/vast_api_key`); every
   `vastai` command — and all the scripts here — read it from there automatically.
   **Do not put the vast API key in `cloud/.env`** — that file is only for Backblaze B2
   credentials and provisioning overrides.
4. Add an SSH key so you can connect. If you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "vast"          # press enter through the prompts
   pbcopy < ~/.ssh/id_ed25519.pub           # copies the PUBLIC key
   ```
   Paste it into **vast.ai → Account → SSH Keys**.

### 3. Create a Backblaze B2 bucket + key

1. Sign up at [backblaze.com/b2](https://www.backblaze.com/cloud-storage) (10 GB free).
2. **Buckets → Create Bucket** (private). Note the bucket name, e.g. `mst-dac-adapter`.
3. **Application Keys → Add a New Application Key**, scoped to that one bucket.
   Copy the **keyID** and **applicationKey** (the secret is shown only once).

### 4. Fill in your secrets

```bash
cp cloud/.env.example cloud/.env
# edit cloud/.env: B2_KEY_ID, B2_APP_KEY, B2_BUCKET
```

`cloud/.env` is gitignored — it never gets committed.

---

## Each run

### 5. Stage code + latents to B2

```bash
cloud/upload.sh
```

Packages the code (data/checkpoints excluded) and uploads it plus `data/vst_latents`
(~2.8 GB) to your bucket. Re-running skips unchanged files. Re-run after any code change.

### 6. Rent a GPU

```bash
cloud/provision.sh
```

Prints the cheapest matching offers and launches an instance from the cheapest one.
Adjust the GPU/price filter via `VAST_QUERY` in `cloud/.env`, or pass an explicit
offer id: `cloud/provision.sh 1234567`.

Wait until it's running:

```bash
vastai show instances
```

### 7. Deploy + train

Grab the SSH endpoint and bootstrap it (installs deps, pulls code + latents from B2):

```bash
cloud/deploy.sh $(vastai ssh-url <INSTANCE_ID>)
```

Then start training (detached tmux session; checkpoints auto-sync to B2 every 5 min):

```bash
ssh -p <PORT> root@<HOST> 'bash /root/mst/cloud/train.sh'
```

Watch progress:

```bash
ssh -p <PORT> root@<HOST> 'tail -f /root/mst/train.out'
```

You should see `Device: cuda` and the per-epoch line:
`Epoch N | train=… (dir=… mag=… probe=…) | val=… (… tgt_acc=…% src_acc=…%)`.
With 100 epochs and this dataset it finishes in well under an hour on a 4090.

### 8. Fetch checkpoints + shut down

```bash
cloud/fetch.sh                              # pulls checkpoints/dac_adapter back to the repo
vastai destroy instance <INSTANCE_ID>       # STOP BILLING — do this when done
```

`best_model.pt` is what `scripts/generate_instrument_dac.py` consumes.

---

## Resuming

Checkpoints live in B2 during the run, so a dead/destroyed instance loses nothing.
On a fresh instance, repeat steps 6–7 but resume:

```bash
# pull existing checkpoints onto the new instance first
ssh -p <PORT> root@<HOST> \
  'cd /root/mst && rclone copy "b2:$B2_BUCKET/checkpoints/dac_adapter" checkpoints/dac_adapter'
ssh -p <PORT> root@<HOST> \
  'bash /root/mst/cloud/train.sh checkpoints/dac_adapter/best_model.pt'
```

---

## Tuning the run

Edit `config/dac_adapter_config.yaml` before `upload.sh`, or override on the instance.
A CUDA GPU has far more headroom than your Mac's MPS:

```yaml
training:
  batch_size: 64        # up from 16 — the adapter is small; the GPU can take it
data:
  num_workers: 0        # latents are cached in RAM at init; workers would duplicate that, leave 0
```

`num_epochs`, `learning_rate`, and the loss weights (`pitch_probe_weight`,
`magnitude_loss_weight`) carry over unchanged from local training.

---

## Troubleshooting

**`torch.cuda.is_available()` is False after bootstrap**
The image's CUDA torch got replaced. `cloud/requirements-cloud.txt` deliberately omits
torch/torchaudio — confirm nothing reinstalled them, or pick a different `VAST_IMAGE`.

**`vastai create` rejects a flag (`--ssh`/`--direct`)**
CLI flags vary by version. Run `vastai create instance --help` and adjust `provision.sh`.

**`ImportError: dac` / descript-audiotools dependency conflict**
Run `pip install descript-audio-codec==1.0.0` manually on the instance and read the
resolver output; pin the conflicting package if needed.

**SSH connection refused**
Instance isn't fully up yet, or the SSH key wasn't added before launch. Check
`vastai show instances`; re-add the key under Account → SSH Keys and relaunch.

**`No offers matched`**
Loosen `VAST_QUERY` in `cloud/.env` (drop the `gpu_name` filter, or lower `reliability`).