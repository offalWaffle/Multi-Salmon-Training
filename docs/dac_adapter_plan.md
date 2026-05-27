# DAC Pitch Adapter — Current State (as of 2026-03-09)

## Context

Goal: one-shot piano instrument generation. DAC encoder/decoder are fully frozen. Two small
adapter networks learn to strip pitch from a DAC latent and inject a new target pitch. The
resulting pipeline can take a single source note and generate all 88 MIDI pitches from it.

### What was tried and abandoned

- **Adversarial disentanglement (GRL + PitchClassifier)**: attempted in many configurations.
  The pitch classifier consistently dominated the stripper regardless of lambda/alpha/LR tuning.
  Root cause: adversarial disentanglement in a 1024-dim continuous latent space is too hard for
  a small residual network. Abandoned after ~40 epochs of failed training runs.
- **Same-instrument cross-recording pairs**: pairs from different recordings of the same
  instrument differ in more than pitch (room acoustics, mic variation, string coupling).
  The irreducible noise floor prevented val loss from decreasing below ~2.37.
- **Audio reconstruction loss (DAC decoder in training loop)**: removed — L1 on latents is
  sufficient and much cheaper.

### Current approach

Cross-pitch pair training, purely supervised, no adversarial loss:

```
src_latent (inst A, vel V, pitch P1) → PitchStripper [no_grad] → z_timbre
z_timbre + src_midi + tgt_midi → PitchInjector → z_modified
loss = L1(z_modified, tgt_latent)   # tgt_latent is inst A, vel V, pitch P2
```

PitchStripper runs under `torch.no_grad()` — it has no gradient path. The injector learns
the full mapping. This sidesteps the adversarial failure mode entirely.

Pairs are grouped by **(instrument_id, velocity)** so within each group only pitch differs —
zero irreducible noise between source and target.

PitchInjector is conditioned on **both src_midi and tgt_midi** separately (not just the
interval) because the pitch transformation in DAC latent space is register-dependent:
C2→C3 looks nothing like C6→C7 due to inharmonicity and spectral content differences.

---

## Data Pipeline (VST Synthetic)

Physical modelling VST generates clean piano samples. Only pitch varies within each
(piano_type, velocity) group — zero irreducible noise between pairs. Dry signal, no reverb.

**Filename convention**: `{PianoType}_{NoteName}_{MIDINote:03d}_v{Velocity:03d}.wav`
Example: `American_Small_Studio_A0_021_v040.wav`
Sharp notation: uses `s` suffix (e.g. `As0` not `A#0`)

**Pipeline**:
```
data/raw/synthetic/piano/{PianoType}/*.wav
  → scripts/prepare_vst_piano.py
  → data/vst_piano/{train,val}/*.pt        # audio .pt files
  → scripts/precompute_dac_latents.py
  → data/vst_latents/{train,val}/*.pt      # latent .pt files
  → scripts/train_dac_adapter.py
```

**Current dataset** (8 piano types, ~2,112 WAVs):

| Piano Type | Notes |
|---|---|
| American_Small_Studio | |
| American_Softened_for_salmon | |
| Classical_Jazz_Player_for_salmon | |
| Classical_U3_Studio__for_salmon | |
| German_Concert_for_salmon | |
| German_Jazz_Concert_for_salmon | |
| Japanese_Studio_for_salmon | |
| Jazz_Player_for_salmon | |

- 3 velocities: 40 / 80 / 120
- 88 MIDI notes (21–108)
- Split: 90% train / 10% val, stratified per (piano_type, velocity) group

---

## Architecture

### PitchStripper
Residual network, near-identity. No adversarial pressure. Runs under `torch.no_grad()`.
Output: `(z_timbre, None)` — z_timbre ≈ z_dac at init.

### PitchInjector
Conditioned on both src_midi AND tgt_midi:

```
z_timbre [B, 1024, T]
  → proj_in: Conv1d(1024, 512, 1)
  → 6 × FiLMResBlock(512, cond_dim=512)   # cond = concat(src_emb[256], tgt_emb[256])
  → proj_out: Conv1d(512, 1024, 1)         # unbounded (no tanh)
z_modified = z_timbre + pitch_delta
```

Zero-init on `proj_out` → identity at epoch 0.
`SinusoidalPitchEmbedding(embed_dim=256)` used for both src and tgt embeddings.

---

## Key Files

| File | Status | Role |
|---|---|---|
| `src/models/dac_pitch_adapter.py` | Done | PitchStripper + PitchInjector + DACPitchAdapter |
| `src/data/dac_latent_dataset.py` | Done | DACPitchPairDataset — pairs by (inst, vel) |
| `src/training/dac_adapter_trainer.py` | Done | Clean trainer, no adversarial code |
| `config/dac_adapter_config.yaml` | Done | inner_dim=512, blocks=6, embed_dim=256 |
| `scripts/train_dac_adapter.py` | Done | Entry point, uses create_pair_dataloader |
| `scripts/generate_instrument_dac.py` | Done | One-shot sweep MIDI 21–108, uses soundfile |
| `scripts/prepare_vst_piano.py` | Done | WAV → .pt, parses VST filename convention |
| `scripts/precompute_dac_latents.py` | Existing | audio .pt → latent .pt |

---

## Config (current)

```yaml
model:
  inner_dim: 512
  num_residual_blocks: 6
  pitch_embed_dim: 256
  dac_latent_dim: 1024
  dac_model_type: "44khz"
audio:
  sample_rate: 44100
  duration: 2.0
training:
  batch_size: 16
  learning_rate: 3.0e-4
  min_lr: 1.0e-6
  num_epochs: 100
  reconstruction_weight: 1.0
  gradient_clip: 1.0
adversarial:
  enabled: false
data:
  latent_dir: data/vst_latents   # update after running precompute_dac_latents.py
  num_workers: 0
  pin_memory: false
checkpoint:
  save_every_n_epochs: 10
```

---

## Dataset Details (DACPitchPairDataset)

- Groups latent files by (instrument_id, velocity)
- Builds all ordered pairs within each group (different pitches only)
- Caches all latents in RAM at init
- Crop: always `latent[:, :crop_frames]` — frame 0 preserves attack transient (not centre crop)
- 2 s window → 172 latent frames at DAC hop=512, 44100 Hz
- `collate_fn_pairs` packs: `{src_latent, src_midi, tgt_latent, tgt_midi}`

---

## Trainer

```python
with torch.no_grad():
    z_timbre, _ = self.adapter.stripper(src_latent)
z_modified = self.adapter.injector(z_timbre, src_midi, tgt_midi)
loss = F.l1_loss(z_modified, tgt_latent.detach())
```

No adversarial components. Clean single-optimizer loop.

---

## Commands

```bash
# 1. Prepare audio .pt files from VST WAVs
python scripts/prepare_vst_piano.py \
    --source-dir data/raw/synthetic/piano \
    --output-dir data/vst_piano

# 2. Encode to DAC latents
python scripts/precompute_dac_latents.py \
    --data-dir data/vst_piano/train --output-dir data/vst_latents/train
python scripts/precompute_dac_latents.py \
    --data-dir data/vst_piano/val   --output-dir data/vst_latents/val

# 3. Train
python scripts/train_dac_adapter.py

# 4. Resume
python scripts/train_dac_adapter.py --resume checkpoints/dac_adapter/best_model.pt

# 5. Generate
python scripts/generate_instrument_dac.py \
    --source note.wav \
    --source-midi 60 \
    --checkpoint checkpoints/dac_adapter/best_model.pt
```

Note: audio loading in generate script uses `soundfile.read` (not torchaudio — TorchCodec
ImportError on this system).

---

## Next Steps

1. Run `precompute_dac_latents.py` on `data/vst_piano/{train,val}` → `data/vst_latents/{train,val}`
2. Update config: `latent_dir: data/vst_latents`
3. Restart training on clean VST synthetic data
4. Monitor val loss — expect faster convergence vs noisy cross-recording pairs (baseline ~2.37)
5. Generate and listen after ~20 epochs to assess pitch transfer quality

### Future architecture improvement
Add dilated convolutions to `ResBlock1d` for wider temporal receptive field. Pitch is a global
temporal property; the current `kernel_size=3` window sees only ~3 frames (~35 ms) of a 172-frame
(2 s) sequence. Dilated convolutions would let the network see longer context without increasing
parameter count.
