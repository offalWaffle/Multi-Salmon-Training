# Multi-Salmon-Training — One-Shot Instrument Generation via DAC Pitch Translation

Generate a full chromatic piano instrument (all 88 keys) from a **single input note** by
translating pitch inside the latent space of a frozen neural audio codec.

This README is the single source of truth for the project's goal, architecture, data
pipeline, and workflows. It is written to give both a human and an LLM enough context to
plan and implement any task after one read. Where it goes deep on a single subsystem, it
points to the design doc that owns the detail.

---

## Status & results

**Working end-to-end.** A trained adapter takes one piano note and renders a playable 88-key
instrument with the source timbre preserved across the range.

| | |
|---|---|
| Best validation loss | **0.323** (cosine-direction + log-magnitude, see §3) at epoch 97 |
| Trainable params | **~27.3M** (stripper + injector); the DAC codec stays frozen |
| Training data | 1,896 train / 216 val DAC latents — 8 piano types × 3 velocities × 88 notes |
| Train cost | a single mid-range GPU (RTX 3090/4090), well under an hour |
| Hardware | develops on Apple Silicon (MPS); trains on CUDA via the `cloud/` pipeline |

**Hear it for yourself** (artifacts are gitignored, so regenerate locally):

```bash
python scripts/generate_instrument_dac.py \
    --source <one_note.wav> --source-midi 60 \
    --checkpoint checkpoints/dac_adapter/best_model.pt \
    --midi-low 21 --midi-high 108 --out-dir output/demo --sfz --diag
```

This writes one WAV per key (`output/demo/note_021_A0.wav` … `note_108_C8.wav`), an SFZ you can
load in any sampler, plus `--diag` reference files that isolate codec roundtrip vs. adapter
artefacts. See §5 for the meaning of every flag.

> This project deliberately keeps its **full research history** in the repo — including the
> approaches that *failed* and why (§8, [`legacy/`](legacy/)). The path to the working model
> (abandoning adversarial disentanglement, killing a hidden noise floor, fixing a decay "buzz")
> is the most informative part; it isn't hidden.

---

## 1. What this project does

**Goal:** take one recorded/synthesised note (e.g. a piano C4) and produce a playable,
multi-sample instrument spanning MIDI 21–108 (A0–C8), preserving the timbre of the source
while changing only the pitch.

**Core idea (current approach):** a pretrained **DAC** (Descript Audio Codec) encoder/decoder
is **frozen**. We train a small adapter network that operates entirely in DAC's continuous
1024-dim latent space. The adapter takes a source latent + the source MIDI note + a target
MIDI note, and outputs a new latent that decodes to the same timbre at the target pitch.

```
                          ┌─────────── trained (small) ───────────┐
 source.wav ──▶ DAC.encode ──▶ z_dac ──▶ PitchStripper ──▶ PitchInjector ──▶ z_modified ──▶ DAC.decode ──▶ note@target_pitch.wav
   (1 note)      [frozen]                 [near-identity]   (src_midi, tgt_midi)               [frozen]
```

Run inference once per target note (21..108) from the same source latent → a full instrument.

**Why a codec latent and not raw audio or a VAE we train ourselves?** DAC already gives a
high-fidelity, invertible audio representation for free. We never put the DAC decoder in the
training gradient path — the adapter is trained purely on a **latent-to-latent** objective,
which is cheap (no decoder backprop, no audio loss) and fast enough to train on a single
mid-range GPU in well under an hour.

---

## 2. Architecture (current: DAC Pitch Adapter)

All adapter code lives in `src/models/dac_pitch_adapter.py`. Latent shape throughout is
`[B, 1024, T]` where `T` = latent frames (172 for a 2 s window — see §4).

### `DACPitchAdapter` (top-level wrapper)
Holds the frozen `dac_model` plus three trainable submodules: `stripper`, `injector`, `probe`.
Only adapter parameters are optimised; every DAC parameter has `requires_grad_(False)`.

- `forward(z_dac, src_midi, tgt_midi) -> (z_modified, z_timbre)` — training path.
- `transfer_pitch(audio, src_midi, tgt_midi) -> audio_out` — full no-grad inference path
  (encode → strip → inject → decode).
- `adapter_parameters()` — the only params handed to the optimizer.

### `PitchStripper` — residual net, stays near-identity
`proj_in (1024→512) → 6× ResBlock1d(512) → proj_out (512→1024)`, with `proj_out` **zero-initialised**
so it outputs `z_dac` unchanged at init (`z_timbre = z_dac − tanh(delta)`, delta≈0). It carries
**no gradient** during training — it runs under `torch.no_grad()`. It exists for architectural
symmetry with the abandoned adversarial design; effectively it passes the latent through. **Do
not** spend effort "fixing" it unless you are reintroducing disentanglement (see §8).

### `PitchInjector` — the model that actually learns
FiLM-conditioned residual stack that predicts a pitch-translation **delta**:

```
z_timbre [B,1024,T]
  → proj_in: Conv1d(1024→512)
  → 6× FiLMResBlock(512, cond_dim=512, dilation=[1,2,4,8,16,32])
  → proj_out: Conv1d(512→1024)            # zero-init → identity at epoch 0
z_modified = z_timbre + pitch_delta       # delta is unbounded; loss bounds its magnitude
```

Two design decisions that matter:

1. **Conditioned on BOTH `src_midi` and `tgt_midi`** (not the interval). Each is mapped through
   a `SinusoidalPitchEmbedding(256)`; the two are concatenated into a 512-dim conditioning
   vector fed to every FiLM block. The latent-space pitch transform is **not interval-invariant**:
   C2→C3 looks nothing like C6→C7 because of inharmonicity and register-dependent spectra, so the
   network needs the absolute source and target, not just the shift.
2. **Exponential dilations `[1,2,4,8,16,32]`** across the 6 blocks give a receptive field of
   ~132 of the 172 frames. Pitch is a global property of the 2 s window; a plain `kernel_size=3`
   stack would only see ~3 frames (~35 ms).

### `PitchProbe` — auxiliary pitch classifier (currently OFF)
Global-average-pools `z_modified` → `Linear(1024→88)` MIDI-class logits. Provides a direct
gradient that pushes the injector to encode the *target* pitch and suppress the *source* pitch,
with no decoder needed. **Currently disabled**: `training.pitch_probe_weight: 0.0` in the config
(see commit `fc22637`). The probe and its loss terms remain wired in the trainer so it can be
re-enabled by raising that weight.

### Supporting building blocks
- `src/models/film_layers.py` — `FiLM`, `FiLMResBlock` (the conditioned residual block used above).
- `src/models/pitch_conditioning.py` — `SinusoidalPitchEmbedding` (MIDI normalised by /127, then
  standard sin/cos frequency bands), plus learned/hybrid/delta variants and `midi_to_frequency`.
- `src/models/conditioned_dac.py` — `load_pretrained_dac(model_type="44khz", device)` is the DAC
  loader used everywhere. The `ConditionedDAC`/`FiLMConditionedDAC` wrappers in this file are an
  earlier idea (conditioning at decode time) and are **not** used by the current adapter path.

---

## 3. Training objective

Implemented in `src/training/dac_adapter_trainer.py` (`DACAdapterTrainer`, `_recon_loss`).
The loss is computed **latent-to-latent** — `z_modified` vs the real target latent `tgt_latent`.

```
direction_loss = mean over (B,T) of (1 − cosine_sim(z_modified, z_target) along channels)
magnitude_loss = L1( log|z_modified|, log|z_target| )            # per-frame L2 norm, in log space
recon_loss     = direction_loss + magnitude_loss_weight · magnitude_loss      # weight = 0.1
loss           = recon_loss + pitch_probe_weight · probe_loss                 # probe_weight = 0.0
```

**Why cosine + log-magnitude instead of flat L1 on the latent?** Flat L1 under-weights
low-energy tail/decay frames (their absolute error is tiny even when their *direction* is
wrong). That let a learned bias-delta leak into the decay and decode as a fixed tonal "buzz".
Per-frame cosine gives every frame equal directional pressure regardless of its magnitude;
the separate log-magnitude term restores the energy envelope.

Optimiser: Adam, `lr 3e-4` → cosine-annealed to `min_lr 1e-6` over `num_epochs`. Gradient
clip 1.0. NaN/Inf guards skip bad batches. Best checkpoint is by **val loss**.

Checkpoints save **stripper + injector** state dicts (not the frozen DAC) plus optimizer,
scheduler, epoch, and `best_val_loss`. Files: `best_model.pt`, `final_model.pt`,
`checkpoint_epoch_N.pt` under `checkpoints/dac_adapter/`.

---

## 4. Data pipeline

The adapter trains on **precomputed DAC latents** of clean synthetic piano notes. Using a
physical-modelling VST means that within a `(piano_type, velocity)` group **only pitch varies**
— there is *zero irreducible noise* between a source and target latent, which is what makes the
supervised latent objective work. (Earlier attempts paired different real recordings of the
"same" instrument; room/mic/string variation put a noise floor under val loss — see §8.)

### Filename convention (VST output)
`{PianoType}_{NoteName}_{MIDINote:03d}_v{Velocity:03d}.wav`
e.g. `American_Small_Studio_A0_021_v040.wav`. Sharps use `s` **or** `#` (`As0` == `A#0`).
Parsed by the regex in `scripts/prepare_vst_piano.py`.

### Flow
```
data/raw/synthetic/piano/{PianoType}/*.wav
  └─ scripts/prepare_vst_piano.py     # WAV → mono 44.1k, 4 s, .pt; stratified 90/10 train/val split
        → data/vst_piano/{train,val}/sample_*.pt          (audio tensors + metadata)
  └─ scripts/precompute_dac_latents.py # DAC.encode each clip once (huge training speedup)
        → data/vst_latents/{train,val}/sample_*.pt        (latent [1024,T] + midi/vel/inst id)
  └─ scripts/train_dac_adapter.py      # trains the adapter on the latents
```

### Latent `.pt` record (what the dataset consumes)
```python
{ 'latent': Tensor[1024, T], 'midi_note': int, 'velocity': int,
  'instrument_id': int, 'original_audio_path': str }
```

### Dataset: `DACPitchPairDataset` (`src/data/dac_latent_dataset.py`)
- Caches all latents in RAM at init.
- Groups files by `(instrument_id, velocity)` and emits **all ordered (src, tgt) pairs with
  different pitch** within each group.
- Adds **identity pairs** (`src == tgt`) at `identity_ratio` (~10%) so the injector also learns
  "change nothing" — without these, a learned bias delta corrupts the tail.
- Crop: always `latent[:, :crop_frames]` from frame 0 (preserves the attack transient — **not** a
  centre crop). 2 s × 44100 / hop 512 = **172 frames**.
- `collate_fn_pairs` packs `{src_latent, src_midi, tgt_latent, tgt_midi}`.
- Use `create_pair_dataloader(...)` to build a loader.

Current dataset: 8 piano types × 3 velocities (40/80/120) × 88 notes (MIDI 21–108) ≈ 2,112 WAVs
→ **1,896 train / 216 val** latents.

> Note: `DACLatentDataset` (non-pair) and the `data/processed*`, `data/vqvae_piano` directories
> belong to earlier approaches. The active path is `vst_piano → vst_latents → pair dataset`.

---

## 5. Quickstart

### Install
```bash
pip install -r requirements.txt     # includes descript-audio-codec; PyTorch w/ MPS on Apple Silicon
```

### Local end-to-end (Apple Silicon / MPS, CUDA, or CPU — auto-detected)
```bash
# 1. WAV → audio .pt  (+ stratified train/val split)
python scripts/prepare_vst_piano.py \
    --source-dir data/raw/synthetic/piano --output-dir data/vst_piano

# 2. Encode audio .pt → DAC latents (.pt)
python scripts/precompute_dac_latents.py --data-dir data/vst_piano/train --output-dir data/vst_latents/train
python scripts/precompute_dac_latents.py --data-dir data/vst_piano/val   --output-dir data/vst_latents/val

# 3. Train the adapter  (reads config/dac_adapter_config.yaml)
python scripts/train_dac_adapter.py
#    resume:
python scripts/train_dac_adapter.py --resume checkpoints/dac_adapter/best_model.pt
#    force device:
python scripts/train_dac_adapter.py --device cuda      # or mps / cpu

# 4. Generate a full instrument from ONE source note
python scripts/generate_instrument_dac.py \
    --source path/to/note.wav --source-midi 60 \
    --checkpoint checkpoints/dac_adapter/best_model.pt \
    --midi-low 21 --midi-high 108 \
    --out-dir output/my_instrument --sfz
```

### Generation flags worth knowing (`generate_instrument_dac.py`)
- `--source-midi` **(required)** — the MIDI note of the source clip; the injector needs it.
- `--sfz` — also write an SFZ manifest mapping each WAV to its key.
- `--requantize` — snap injector output back onto DAC's RVQ codebook manifold before decoding.
- `--env-from-source` — replace per-frame magnitude with the source's (inherit amplitude envelope,
  keep injector's direction).
- `--diag` — also write `diag_roundtrip.wav` (decode source latents, no adapter) and
  `diag_identity.wav` (injector with tgt=src) to isolate where artefacts come from.
- Audio I/O uses `soundfile` (not torchaudio) because TorchCodec raises ImportError on this system.

---

## 6. Cloud training (vast.ai + Backblaze B2)

For a GPU run, everything is scripted under `cloud/`. Data flows through a Backblaze B2 bucket so
you can relaunch cheap instances without re-uploading. **Full walkthrough:
[`docs/VAST_TRAINING_GUIDE.md`](docs/VAST_TRAINING_GUIDE.md).**

```
  Mac ──upload.sh──▶  B2 bucket  ◀──bootstrap.sh── vast.ai GPU
   ▲                  (code +                          │ train.sh (tmux + checkpoint sync)
   └──fetch.sh────────  latents + ◀──checkpoints───────┘
                        checkpoints)
```

| Script | Runs on | Does |
|---|---|---|
| `cloud/upload.sh`    | Mac      | Tar code + push `data/vst_latents` to B2 |
| `cloud/provision.sh` | Mac      | Find a cheap reliable single-GPU offer, launch a vast.ai instance |
| `cloud/deploy.sh`    | Mac      | Copy + run `bootstrap.sh` on the instance |
| `cloud/bootstrap.sh` | instance | Install deps, pull code + latents from B2 |
| `cloud/train.sh`     | instance | Train in tmux; sync checkpoints to B2 every 5 min |
| `cloud/fetch.sh`     | Mac      | Pull trained checkpoints back into the repo |

Config: copy `cloud/.env.example` → `cloud/.env` (gitignored) and fill in B2 credentials. The
vast.ai API key is **not** stored there — register it once with `vastai set api-key <KEY>`.
`cloud/requirements-cloud.txt` intentionally omits torch/torchaudio (the vast PyTorch base image
ships CUDA builds; reinstalling from PyPI can silently swap in a CPU-only build).

---

## 7. Repository map

The active system lives entirely in `src/` + `scripts/` + `cloud/`. Everything from earlier
abandoned approaches has been moved under `legacy/` so the active path is unambiguous.

```
config/
  dac_adapter_config.yaml      the ACTIVE config (model dims, training, data, checkpoint paths)
  dac_latent_stats.yaml        latent normalisation stats
src/                           ← ACTIVE: the DAC pitch-adapter system, and only that
  models/
    dac_pitch_adapter.py         PitchStripper, PitchInjector, PitchProbe, DACPitchAdapter
    conditioned_dac.py           load_pretrained_dac(); ConditionedDAC decode-time wrappers (unused by adapter)
    film_layers.py               FiLM / FiLMResBlock building blocks
    pitch_conditioning.py        SinusoidalPitchEmbedding etc.
  data/
    dac_latent_dataset.py        DACPitchPairDataset + create_pair_dataloader
    audio_utils.py, sfz_parser.py   generic audio / SFZ helpers
  training/
    dac_adapter_trainer.py       DACAdapterTrainer + cosine/magnitude loss
scripts/                       ← ACTIVE entry points
  prepare_vst_piano.py           data prep (WAV → .pt + stratified split)
  precompute_dac_latents.py      audio .pt → latent .pt
  train_dac_adapter.py           training entry point
  generate_instrument_dac.py     inference / instrument export
  analyze_latent_pitch_correlation.py   latent↔pitch diagnostic
cloud/                         vast.ai + B2 GPU training pipeline (see §6)
docs/                          design docs (see §9)
legacy/                        ARCHIVED prior approaches (VQ-VAE+transformer, diffusion-era)
                               — models/data/training/losses/config/scripts/tests + its own README
data/                          datasets (gitignored); active: vst_piano/, vst_latents/
checkpoints/, logs/, output/   training + generation artefacts (gitignored)
```

**Legacy vs active:** the repo's earlier directions (a VQ-VAE + latent-transformer pipeline, an
adversarial-disentanglement variant, and a still-earlier diffusion era) are archived under
[`legacy/`](legacy/) with [`legacy/README.md`](legacy/README.md) explaining each. They are kept
to document the project's evolution (§8) but are **not** on the active path — everything outside
`legacy/` is the current system.

---

## 8. Key design decisions & rationale (read before changing the model)

These are the hard-won decisions encoded in the current design. Each exists because the obvious
alternative was tried and failed.

1. **DAC is frozen; we only train a latent-space adapter.** No audio reconstruction loss, no
   decoder in the gradient path. L1/cosine on latents is sufficient and far cheaper.
2. **Stripper has no gradient; the injector does all the work.** This sidesteps the adversarial
   disentanglement failure mode (below) entirely — it is a purely supervised mapping.
3. **Inject on (src_midi, tgt_midi), not the interval.** The latent pitch transform is
   register-dependent; interval-only conditioning cannot represent it.
4. **Dilated FiLM blocks.** Needed to see the global pitch structure of a 2 s window.
5. **Cosine-direction + log-magnitude loss, not flat L1.** Prevents a tonal "buzz" in decay
   regions caused by under-weighted low-energy frames.
6. **Pairs grouped by (instrument, velocity); identity pairs included.** Guarantees only pitch
   differs between src/tgt (zero noise floor) and teaches a clean identity transform.

### Abandoned approaches (do not re-attempt without new evidence)
- **Adversarial disentanglement (GRL + PitchClassifier):** the classifier consistently dominated
  the stripper across all λ/α/LR settings — adversarial disentanglement of a 1024-dim continuous
  latent with a small residual net is too hard. (`docs/ADVERSARIAL_DISENTANGLEMENT_APPROACH.md`,
  `legacy/models/pitch_adversary.py`.)
- **Same-instrument cross-recording pairs:** acoustic variation between recordings put a hard
  floor (~2.37) under val loss. Fixed by switching to single-VST synthetic data.
- **DAC decoder in the training loop (audio loss):** removed; latent loss is enough and much
  cheaper.
- **VQ-VAE + latent transformer:** the project's original direction; superseded by the DAC adapter.

### Known next steps / open ideas
- Re-enable the pitch probe (`pitch_probe_weight > 0`) if pitch identity needs reinforcing.
- Listen-test after ~20 epochs; assess pitch-transfer quality across registers.
- Velocity is currently a *grouping key only* — the adapter does not yet translate velocity. A
  natural extension is to condition the injector on src/tgt velocity the same way it does pitch.

---

## 9. Design docs

- [`docs/dac_adapter_plan.md`](docs/dac_adapter_plan.md) — the adapter's design log and current
  state (closest companion to this README; some loss details predate the cosine+magnitude switch).
- [`docs/VAST_TRAINING_GUIDE.md`](docs/VAST_TRAINING_GUIDE.md) — full cloud-training walkthrough.
- [`docs/ADVERSARIAL_DISENTANGLEMENT_APPROACH.md`](docs/ADVERSARIAL_DISENTANGLEMENT_APPROACH.md),
  [`docs/PITCH_NORMALIZATION_APPROACH.md`](docs/PITCH_NORMALIZATION_APPROACH.md),
  [`docs/COLAB_TRAINING_GUIDE.md`](docs/COLAB_TRAINING_GUIDE.md),
  [`docs/SINGLE_SAMPLE_INSTRUMENT_FIX.md`](docs/SINGLE_SAMPLE_INSTRUMENT_FIX.md) — earlier
  approaches; historical context.

---

## 10. Glossary

- **DAC** — Descript Audio Codec; the frozen pretrained neural audio codec (44.1 kHz model, hop
  512, 1024-dim continuous latent). Loaded via `load_pretrained_dac`.
- **Latent / `z_dac`** — DAC's continuous encoder output, shape `[B, 1024, T]`. The adapter
  operates here, never on raw audio.
- **Latent frame** — one time step of the latent; 172 frames ≈ 2 s at hop 512 / 44.1 kHz.
- **RVQ** — DAC's residual vector quantiser; `--requantize` snaps a latent back onto its codebook.
- **FiLM** — Feature-wise Linear Modulation; per-channel scale+shift predicted from conditioning.
- **Identity pair** — a training pair where `src_midi == tgt_midi` ("change nothing").
- **MPS** — Apple Metal Performance Shaders backend for PyTorch (the local dev device).

---

## References
- Descript Audio Codec (DAC): https://github.com/descriptinc/descript-audio-codec
- FiLM: Perez et al., 2018 — *FiLM: Visual Reasoning with a General Conditioning Layer*
- SFZ format: https://sfzformat.com/