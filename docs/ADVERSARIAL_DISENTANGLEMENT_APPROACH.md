# Adversarial Disentanglement Approach — Design Document

## Problem

The VQ-VAE encoder conflates timbre and pitch in `z_q`. The quantized codes carry
both "what instrument is this" and "what pitch is this", making pitch transformation
in latent space intractable (evidenced by the LatentTransformer plateauing at latent
MSE ~3.2 after 63 epochs).

The decoder already has FiLM pitch conditioning — architecturally it was designed to
render pitch from a pitch-invariant code. The encoder just never learned to honour
that contract because nothing in the training forced it to.

---

## Goal

Force `z_q` to contain **only timbre information** by making it impossible for a pitch
classifier to read the pitch from `z_q`. The decoder's FiLM layers then become the
sole carrier of pitch information, which is exactly what they were designed for.

---

## Mechanism: Gradient Reversal

A **gradient reversal layer (GRL)** sits between `z_q` and a small pitch classifier.

```
                        ┌─────────────────────────────────────────┐
                        │  Normal VQ-VAE forward pass             │
  audio ──→ Encoder ──→ z ──→ Quantizer ──→ z_q ──→ Decoder(FiLM=pitch) ──→ recon
                                              │                              loss
                        └──────────────────── │ ────────────────────────────┘
                                              │
                              ┌───────────────┘
                              │
                        [Gradient Reversal Layer]
                              │
                              ↓
                        Pitch Classifier  ──→  predicted MIDI  ──→  CE loss
                        (3-layer MLP)
```

### Forward pass (normal)
The GRL is an identity function — it passes `z_q` through unchanged to the classifier.
The classifier predicts the MIDI note. Cross-entropy loss measures how well it succeeds.

### Backward pass (the trick)
The GRL **flips the sign** of the gradient before passing it back to the encoder.

- The **classifier** receives normal gradients → it gets better at reading pitch from `z_q`
- The **encoder** receives sign-flipped gradients → it is penalised whenever `z_q` leaks pitch

This is a minimax game:
- Classifier: *"I will learn to extract pitch from z_q"*
- Encoder: *"I will hide pitch from z_q"*

Equilibrium: `z_q` contains no pitch-discriminable information. The classifier
performs at chance (~1/88 accuracy for MIDI 21–108). The encoder is forced to route
pitch through the only remaining pathway: the FiLM conditioning in the decoder.

### GRL implementation (5 lines)

```python
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x  # identity in forward pass

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None  # flip gradient sign
```

`alpha` is a scaling factor for the reversal strength. Typically ramped from 0 → 1
over the first ~10 epochs to let the reconstruction loss stabilise before the
adversarial pressure kicks in.

---

## Pitch Classifier

A small MLP that takes a **pooled summary** of `z_q` and predicts the MIDI note.

```
z_q  [B, 64, 5512]
  │
  └── global average pool ──→ [B, 64]
                                 │
                            Linear(64, 256) + ReLU
                                 │
                            Linear(256, 128) + ReLU
                                 │
                            Linear(128, 88)   ← 88 piano MIDI notes (21–108)
                                 │
                            CrossEntropyLoss(predicted, true_midi_note)
```

Parameter count: ~50k. Negligible overhead.

The classifier uses global average pooling over the time dimension because pitch is
a global property of the audio clip — it doesn't vary frame by frame.

---

## Full Training Loss

```
total_loss = recon_loss + vq_loss − λ * pitch_classifier_loss
```

Where:
- `recon_loss` — existing AudioReconstructionLoss (STFT + Mel + time-domain L1)
- `vq_loss` — existing commitment + codebook loss
- `pitch_classifier_loss` — CrossEntropy(predicted_midi, true_midi)
- `λ` — adversarial weight, typically 0.1–1.0

Note the **minus sign**: we minimise the total loss, so subtracting the classifier
loss means the encoder is rewarded for making the classifier *worse*.

The classifier itself is updated normally (gradient ascent on its own parameters),
separate from the main optimiser. In practice this is implemented by:
1. Detaching `z_q` for the classifier's own parameter update
2. Using the GRL for the encoder's update

---

## Why This Works Better Than Alternatives for This Dataset

| Approach | Requires | Problem with 4 instruments |
|---|---|---|
| Consistency loss | Same-instrument pairs | Works, but no force to keep instruments apart → possible collapse |
| Triplet / contrastive | Many negatives | Very few inter-instrument negatives |
| **Gradient reversal** | Per-sample pitch labels | Works on every sample independently — number of instruments irrelevant |

The adversarial signal comes from **pitch prediction accuracy**, not from
comparing instruments. With 88 possible MIDI notes and clear pitch labels already
in the dataset metadata, the classifier has a rich training signal regardless of
how many instruments are present.

---

## Dataset Requirements for Excellent Quality

### What each component needs

**The pitch classifier** needs enough examples per MIDI class to learn pitch discrimination.
With 88 classes, target at minimum ~50 samples per pitch. Five full-coverage sets gives
~100 per class — already adequate. This is the easiest requirement to satisfy.

**The encoder** needs timbres that are *uncorrelated with pitch*. This is the critical
requirement. Piano naturally has a brightness-vs-pitch correlation (high notes are brighter,
low notes darker) — a real acoustic property. With only a few similar piano types, the
encoder can exploit "brightness predicts pitch" as a shortcut and the adversarial loss has
to fight an acoustically real correlation. More diverse timbres break this shortcut.

**The FiLM decoder** needs enough (timbre × pitch) combinations to learn the full rendering
function — how to produce timbre X at pitch Y for all X and Y in the training distribution.
The more it sees "bright timbre at C2", "dark timbre at C2", "bright timbre at C7"..., the
better it generalises to unseen timbres at any pitch.

### Target numbers

| Goal | Real sets | Synthesized sets | Total samples |
|---|---|---|---|
| Minimum (decent) | 4–6 | 5–10 | ~2,000 |
| Good | 8–12 | 10–15 | ~5,000–8,000 |
| **Excellent** | **10–15** | **20–30** | **~15,000–25,000** |

Aim for **~40 distinct timbral variants total** with full pitch coverage. The split between
real and synthesized matters less than timbral diversity.

### Why synthesized data is especially valuable

Real recordings always have the natural brightness-vs-pitch correlation baked in. A synthesized
set lets you decouple them completely:

- Generate a "very bright" timbre at *every* pitch, including the low register
- Generate a "very dark" timbre at *every* pitch, including the high register

This is acoustically unusual (a real bright-sounding A0 piano doesn't exist in nature), but
that is exactly what the encoder needs to see. It learns: *brightness is a timbre property,
not a pitch property.*

### What makes good timbral diversity

**For real recordings**, prioritise sets that differ on multiple axes:
- Grand vs upright vs toy vs prepared piano
- Dry close-mic vs roomy/reverberant recording
- Bright/hard-hammered vs mellow/felt-damped
- Different piano models (Steinway, Yamaha, Bösendorfer, Bechstein have meaningfully
  different harmonic characters)
- Honky-tonk / detuned / unusual treatment

**For synthesized sets**, vary these parameters independently across the full pitch range:
- **Brightness** — harmonic rolloff from very dull to very bright
- **Attack sharpness** — from percussive/clicky to soft/slow
- **Sustain character** — fast decay vs long sustain
- **Inharmonicity** — tight vs stretched harmonics (affects perceived "warmth")
- **Body resonance** — different room/body transfer functions

The goal is that no two sets occupy the same region of timbre space. Thirty sets that all
sound like "a slightly different grand piano" give diminishing returns compared to fifteen
sets that clearly span different regions.

### Full pitch coverage requirement

Every set should cover all or most of MIDI 21–108. Sparse coverage per instrument weakens
the pitch classifier's signal and creates gaps where the decoder has never seen a particular
(timbre, pitch) combination. If a real set has sparse coverage in the extreme registers,
synthesized variants can fill those gaps.

---

## Changes Required to Existing Codebase

### New: `src/models/pitch_adversary.py`
- `GradientReversalLayer` — autograd function wrapping the sign flip
- `PitchClassifier` — pooling + MLP + CrossEntropyLoss

### Modified: `src/training/vqvae_trainer.py`
- Instantiate `PitchClassifier` with a separate optimiser
- Add adversarial step to `train_epoch`:
  1. Forward: encode → z_q → GRL → classifier → CE loss
  2. Classifier update: normal gradient step on classifier params
  3. Encoder update: adversarial gradient (via GRL) included in main loss
- Add `alpha` schedule: ramp from 0 → 1 over first 10 epochs

### Modified: `config/vqvae_config.yaml`
```yaml
adversarial:
  enabled: true
  lambda: 0.1          # adversarial loss weight — tune between 0.05–1.0
  alpha_warmup_epochs: 10  # ramp GRL strength over first N epochs
  classifier_lr: 1.0e-3    # classifier can use higher LR than main model
```

### No changes needed
- `src/models/vqvae.py` — architecture unchanged
- `src/models/quantizers.py` — unchanged
- `src/losses/spectral_loss.py` — unchanged
- `src/data/vqvae_dataset.py` — MIDI labels already in metadata

---

## Expected Training Dynamics

| Phase | Epochs | What to watch |
|---|---|---|
| Warmup | 0–10 | Reconstruction loss stabilises, alpha ramps, classifier loss drops (it learns to read pitch) |
| Adversarial kicks in | 10–30 | Classifier loss starts rising (encoder hiding pitch), reconstruction loss may temporarily increase |
| Convergence | 30–80 | Reconstruction stable, classifier at/near chance, z_q perplexity should remain >100 |

**Key diagnostic**: after training, run the classifier on the frozen encoder's outputs.
If accuracy is near `1/88 ≈ 1.1%`, disentanglement succeeded.
If accuracy stays high (>50%), increase `lambda` or `alpha`.

---

## Inference — No Transformer Required

```python
# Encode any sample at its original pitch
z_q = vqvae.encode(source_audio)          # timbre code only

# Decode at any target pitch — FiLM handles the pitch rendering
audio_out = vqvae.decode(z_q, target_midi)
```

One forward pass. No paired data. No pitch normalization. No transformer.

---

## Risks & Open Questions

- **Reconstruction quality**: adversarial pressure may hurt reconstruction if lambda is
  too high. The encoder may struggle to encode enough timbre detail while also hiding pitch.
  Mitigation: start with small lambda (0.05) and increase only if disentanglement is poor.

- **Codebook collapse**: if the adversarial loss dominates, the encoder may map all inputs
  to similar z vectors (easy way to hide pitch). Monitor perplexity — it should stay >100.

- **FiLM capacity**: the decoder's FiLM layers must now do all the pitch rendering.
  With the current architecture (SinusoidalPitchEmbedding + FiLMConv1d/FiLMResBlock),
  this may be insufficient for a 7-octave pitch range. May need deeper FiLM conditioning.

- **Retraining**: the existing VQ-VAE checkpoint can be used as a warm start
  (load weights, then fine-tune with adversarial loss), which should be faster than
  training from scratch.
