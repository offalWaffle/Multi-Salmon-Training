# Pitch Normalization Approach — Design Document

## Problem Being Solved

The VQ-VAE encoder was trained without a disentanglement loss (`perceptual_weight: 0.0`),
so the quantized codes `z_q` encode **both timbre and pitch**. This makes the LatentTransformer
approach unworkable: the transformer must learn a pitch-remapping in a space that conflates
pitch and timbre, and has plateaued at latent MSE ~3.2 after 63 epochs.

The root issue: the VQ-VAE was designed with a pitch-conditioned decoder (FiLM) that assumes
the encoder produces pitch-invariant codes, but nothing in the training forced that invariance.

---

## The Idea

Force pitch-invariance at the **data level** rather than through a loss term. Before the encoder
ever sees audio, resample every sample to a fixed reference pitch (tape-style: change playback
speed). The encoder then always receives audio at one pitch and must learn to encode only timbre.
The decoder's FiLM conditioning then has a real job: render that timbre code at the original pitch.

```
Training forward pass:

  audio_original (G4)
       │
       ├─── resample to C4 ──→ audio_normalized ──→ Encoder ──→ z_q
       │                                                          │
       │                                              Decoder(z_q, FiLM=G4)
       │                                                          │
       └────────────────────────────── reconstruction loss ───────┘

Inference:

  source_audio (any pitch P_src)
       │
       └─── resample to C4 ──→ Encoder ──→ z_q ──→ Decoder(z_q, FiLM=P_target) ──→ audio_out
```

No transformer is needed. Phase 2 is eliminated.

---

## Signal Processing Constraint

Tape-style pitch normalization = resample = speed change. This is asymmetric in its effect
on frequency content:

### Pitching UP a low note (e.g. A0 → C4, ×9.5 speed-up)
- Frequency content shifts UP by the same factor
- Anything in the original above `22050 / pitch_ratio` Hz is pushed past Nyquist
- The anti-aliasing filter discards it — information is **permanently lost**
- A0 has piano harmonics well above 2.3 kHz (the cutoff at ×9.5)
- When the decoder tries to reconstruct A0, z_q has no record those harmonics existed
- Result: reconstructed low notes sound muffled / lacking brightness

### Pitching DOWN a high note (e.g. C7 → C4, ×8 slow-down)
- Content shifts DOWN in frequency — stays within Nyquist
- No information is lost (piano has no energy below its fundamental)
- Clean round-trip

The constraint is **asymmetric**: pitch-up loses highs, pitch-down loses nothing (spectrally).

---

## Reference Pitch & Frequency Loss by Range

| Reference | Max pitch-up from A0 | Nyquist cutoff on original | Harmonic loss |
|---|---|---|---|
| C6 (MIDI 72) | +51 semitones, ×45 | 490 Hz | catastrophic |
| C4 (MIDI 60) | +39 semitones, ×9.5 | 2.3 kHz | severe |
| C3 (MIDI 48) | +27 semitones, ×4 | 5.5 kHz | moderate |
| C2 (MIDI 36) | +15 semitones, ×2 | 11 kHz | acceptable |

But a low reference compounds the **duration** problem in the other direction:
pitching a C8 sample down to C2 (6 octaves) slows it by 64× — a 2s sample becomes 128s.

---

## Duration Problem

Tape-style repitch changes playback speed, so duration scales inversely with pitch ratio.

| Note → C4 | Pitch ratio | Original 2s sample becomes |
|---|---|---|
| C6 pitched down | ÷4 | 8s (trim to 2s → lose most of decay) |
| C5 pitched down | ÷2 | 4s (trim to 2s → lose tail) |
| C4 (reference) | ×1 | 2s |
| C3 pitched up | ×2 | 1s (pad with 1s silence) |
| C2 pitched up | ×4 | 0.5s (pad with 1.5s silence) |

Very low notes become mostly silence after normalization. Very high notes lose their decay.

---

## Recommended Constraints

To keep both frequency loss and duration expansion within acceptable bounds:

- **Training pitch range: MIDI 48–84 (C3–C6, 3 octaves)**
- **Reference pitch: C4 (MIDI 60)**
- Max pitch-up: +12 semitones (×2) → lose above **11 kHz** (acceptable for piano)
- Max pitch-down: −24 semitones (×4) → 2s sample expands to 8s, trim to first 2s
  (the trimmed portion is mostly sustain/release — the attack and body are preserved)

This excludes MIDI 21–47 (A0–B2) and MIDI 85–108 (C#6–C8) from training.
For a one-shot instrument generator focused on the playable range this is acceptable.

---

## Changes Required to Existing Codebase

### `src/data/vqvae_dataset.py`
- Add `pitch_normalize(audio, src_midi, ref_midi=60)` using `scipy.signal.resample`
- Filter samples to MIDI 48–84 at load time
- Return `audio_input` (normalized) and `audio_target` (original)

### `src/models/vqvae.py` — no changes
The architecture already supports this: encoder takes audio, decoder takes `z_q + pitch`.

### `config/vqvae_config.yaml`
- Add `reference_midi: 60`
- Add `pitch_range: [48, 84]`

### Training loop (`vqvae_trainer.py`)
- `encoder(audio_input)` — normalized audio
- `decoder(z_q, midi_note)` — original pitch via FiLM
- `loss vs audio_target` — original audio

---

## Advantages
- Forces pitch-invariance without any architectural change or disentanglement loss
- Eliminates Phase 2 (transformer) entirely — inference is just encode + decode
- Conceptually clean: encoder = timbre codec, decoder = pitch renderer
- Tested approach in sample library design (analogous to "root key" normalization in samplers)

## Disadvantages / Open Questions
- Loses ~3 octaves of training range (bottom and top of piano)
- High-frequency content of low-register notes is degraded during normalization
- Does the VQ-VAE decoder's FiLM actually have sufficient capacity to render
  the same timbre across a 3-octave pitch range? Untested.
- Tape-style pitch shift introduces temporal smearing artifacts at large ratios —
  phase vocoder (PSOLA) would be higher quality but more complex
- Retraining VQ-VAE from scratch required
