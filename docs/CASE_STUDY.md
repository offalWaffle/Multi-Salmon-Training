
# Case Study - One-Shot Instrument Generation by Pitch Translation in a Codec Latent Space

> How a one-sentence goal ("turn one piano note into a whole playable instrument") survived five
> rejected approaches and converged on something small, supervised, and reliable - and how a single
> non-negotiable, *commercial-grade sound quality*, eliminated each dead end along the way.

This is the narrative companion to the [README](../README.md). The README tells you *what the
system is and how to run it*. This document tells you *how it got there and why it looks the way
it does*. If you only read one thing for the engineering reasoning behind the design, read this.

---

## At a glance

| | |
|---|---|
| **Goal** | Generate a full 88-key piano instrument from a single input note, preserving timbre |
| **North-star constraint** | **Commercial-grade sound quality** - output must pass as a real recorded multisample in a mix, not as "AI audio." This is the lens that killed every rejected approach. |
| **Final approach** | Freeze a pretrained neural audio codec (DAC), then train a small adapter that translates pitch in its latent space |
| **Result** | Working end-to-end. **0.323** validation loss, **~27.3M** trainable params, trains in <1 hr on one mid-range GPU |
| **Approaches tried first** | Three established paradigms rejected on fidelity - (1) DDSP additive synthesis, (2) RAVE, (3) mel-spectrogram + neural vocoder - then two of my own codec-lineage builds - (4) VQ-VAE + latent transformer, (5) adversarial pitch disentanglement |
| **Headline lesson** | The biggest wins came from *reframing the problem* and *fixing the data*, not from a bigger model - but sound quality was the filter that decided which problems were even worth solving |

---

## 1. The problem

A multi-sampled software instrument needs a recording of *every* key, often at several velocities -
hundreds of samples. I wanted to collapse that to **one**: give the system a single note, and have
it synthesise the same instrument across the full MIDI 21–108 range (A0–C8), changing pitch while
keeping timbre identical.

The hard part is the second clause. Naïvely pitch-shifting audio (resampling) also shifts the
formants and timbre - a piano shifted up two octaves sounds like a toy, not the same piano higher
up. Real instruments are *inharmonic* and *register-dependent*: a low C and a high C of the same
piano differ in spectral envelope, attack, and decay in ways that simple shifting can't reproduce.
So the task is really: **change pitch as a perceptual attribute while holding everything else
constant.** That is a disentanglement problem, and disentanglement is where this project spent most
of its blood.

**And there was a second, harder constraint underneath it: this has to be a commercial tool.** The
output is meant to ship inside a sample-library product and sit in a real mix next to hand-recorded
multisamples without anyone hearing the seam. That makes *sound quality the number-one target* - not
an after-the-fact polish step but the spec itself. Anything that sounds like "AI audio" - smeared
transients, a metallic or buzzy decay, a low-pass haze, vocoder warble, unstable pitch - is an
automatic fail no matter how elegant the method that produced it. Every approach below was judged
first against that bar, and most of them died there.

---

## 2. Three paradigms I rejected on sound quality

Before the codec-latent lineage that the rest of this document is about, I prototyped the three most
obvious established paradigms for audio generation. Each is a reasonable, well-cited way to attack
this problem. Each was rejected for the same reason: its *best realistic output* didn't clear the
commercial-fidelity bar from §1. They're worth recording because ruling them out is what made the
eventual design inevitable.

### 2a. DDSP - differentiable additive synthesis

**The idea.** Extract a per-frame pitch (F0 via CREPE) and loudness from the note, feed them to a
*differentiable* harmonic-plus-noise synthesizer, and learn the time-varying harmonic amplitudes and
filtered-noise envelopes. Physically grounded, tiny models, and pitch is a native control input - on
paper an ideal fit for "same timbre, any pitch."

**What broke (fidelity).** The harmonic+noise model is a strong prior, and it flattens precisely the
things that make a real piano sound real: inharmonic partials, the noisy hammer/transient attack,
sympathetic string resonance, the slightly non-harmonic structure of a struck string. Output had the
characteristic DDSP "synthy / organ-like" timbre - clean, but unmistakably synthetic. The pipeline
also ran at 16 kHz (the ddsp + CREPE default), which caps brightness well below a 44.1 kHz product.
It stalled in data prep - CREPE F0-confidence and TFRecord failures on real library samples - before
it could be a fair fight, but the timbre ceiling was the real disqualifier. I tried it twice. A
clean-slate second attempt (`DDSPMultiSampler2`) hit the same wall.

### 2b. RAVE - realtime audio VAE

**The idea.** Train RAVE on a large corpus (~5k samples drawn from ~170 SFZ instrument libraries) to
learn a general instrument latent, then generate a full instrument from just 1–5 input notes by
encoding them and interpolating/extrapolating through that latent space.

**What broke (fidelity, and tooling).** RAVE is engineered for *realtime* timbre transfer and
streaming synthesis, and its adversarial decoder explicitly trades fidelity for speed. On sustained
tonal material that shows up as the signature RAVE "fuzz"/granular texture and subtly unstable pitch
- tolerable for an effect, fatal for a clean sampled instrument. On top of the wrong fidelity
profile, it never actually trained here: RAVE has no MPS support, the CPU fallback was hopeless for a
multi-day run on this dataset, and the native training run failed outright. Even with the tooling
fixed, the sound it produces was the wrong target.

### 2c. Mel-spectrogram + neural vocoder (HiFi-GAN / BigVGAN)

**The idea.** Treat it as image-to-image translation: an encoder compresses the reference note's mel
spectrogram into a latent timbre vector, a decoder predicts the mel of the target pitch/velocity, and
a pretrained neural vocoder (HiFi-GAN / BigVGAN / Vocos) inverts mel → audio. This lets you borrow
the whole mature toolbox of image ML (U-Nets, pix2pix, latent diffusion).

**What broke (fidelity).** Two stacked lossy steps in the signal path. The mel transform throws away
phase, and the vocoder's "indistinguishable from the original" reputation comes from *speech/TTS*
listening tests that don't transfer to sustained musical tones - on piano you get metallic artefacts
in the decay tail and a smeared attack. Worse, it only works at all if the training mels match the
vocoder's exact config (n_fft, hop, fmin, fmax, sample rate), and any mismatch adds its own artefacts.
The error budget for commercial-grade audio was spent before modelling even started.

**The throughline.** Every one of these put a *lossy or low-fidelity audio model* somewhere in the
signal path, and that's where the quality leaked out. The lesson that survived into the final design:
keep audio in a single, high-fidelity, *invertible* representation end-to-end, and never retrain the
part responsible for fidelity. That is exactly what a frozen, near-transparent codec (DAC) provides -
but I reached it through two more builds in that direction, which failed for a *different* reason:
not fidelity, but disentanglement.

---

## 3. Attempt - VQ-VAE + latent transformer

**The idea.** Train a VQ-VAE to compress piano audio into a discrete latent code, then train a
transformer over those codes to map *(source latent, target pitch) → target latent*. The decoder
already had FiLM pitch conditioning, so in principle the latent could carry "timbre" and the FiLM
path could carry "pitch."

**What broke.** The encoder had no incentive to *separate* the two. It happily baked pitch into the
discrete code, because that minimised reconstruction loss most directly. The transformer then had to
learn a transformation over a representation that entangled the very thing it was supposed to change.
It plateaued at a latent MSE of **~3.2 after 63 epochs** and never produced clean pitch transfer.

**The lesson.** A latent that wasn't *designed* to be pitch-invariant won't become so just because a
downstream model wishes it were. The entanglement has to be addressed at its source - in the encoder.

→ Archived under [`legacy/`](../legacy/) (`models/vqvae.py`, `models/latent_transformer.py`).

---

## 4. Attempt - adversarial pitch disentanglement

This was the elegant idea, and the one I most wanted to work.

**The mechanism.** Put a small pitch classifier on the VQ-VAE's latent `z_q`, connected through a
**gradient reversal layer (GRL)**. In the forward pass the GRL is the identity, and in the backward pass
it flips the sign of the gradient flowing back to the encoder. The result is a minimax game:

- the **classifier** is rewarded for reading pitch out of `z_q`,
- the **encoder** is rewarded (via the flipped gradient) for *hiding* pitch from `z_q`.

At equilibrium the classifier sits at chance (~1/88 ≈ 1.1%), `z_q` is pitch-free timbre, and the
decoder's FiLM layers become the sole carrier of pitch - exactly what they were architected for. No
paired data, no transformer, just one encode + a FiLM-conditioned decode at any target pitch. On paper
it's beautiful. (Full design: [`docs/ADVERSARIAL_DISENTANGLEMENT_APPROACH.md`](ADVERSARIAL_DISENTANGLEMENT_APPROACH.md).)

**What broke.** The adversary wouldn't reach the equilibrium I needed. Across every setting of the
adversarial weight λ, the GRL strength α, and the classifier learning rate I tried, **the classifier
kept winning** - it could always recover pitch from the 1024-dim continuous latent faster than the
encoder could learn to hide it. Disentangling a high-dimensional continuous representation with a
small residual network turned out to be genuinely hard, not a tuning detail. I burned ~40 epochs of
runs confirming it from multiple angles before calling it.

**The lesson - and the reframe it forced.** I had been treating "remove pitch from the
representation" as the goal. But I didn't actually need a pitch-*invariant* latent. I needed a
pitch-*translated* output. Those are different problems, and the second one is *much* easier: it can
be solved with direct supervision instead of an adversarial game. That realisation is the hinge of
the whole project.

→ Archived under [`legacy/`](../legacy/) (`models/pitch_adversary.py`).

---

## 5. The reframe - stop disentangling, start translating

Two decisions fell out of the reframe, and together they define the current system.

**Decision 1: don't train a codec at all - borrow one.** Instead of fighting to train a VQ-VAE with
a clean latent, I took a *pretrained, frozen* neural audio codec - Descript's **DAC** - off the
shelf. It already provides a high-fidelity, invertible audio representation. I never touch its
weights. This removes an entire failure surface (codec training, codebook collapse, reconstruction
vs. disentanglement tradeoffs) in one stroke.

**Decision 2: supervised latent-to-latent translation, not disentanglement.** Train one small
network to map a *source* DAC latent to a *target* DAC latent at a new pitch, supervised directly by
a real example of the target. Concretely:

```
src_latent (instrument A, velocity V, pitch P1)
    → PitchInjector(·, src_midi=P1, tgt_midi=P2)
    → predicted target latent
loss = distance( predicted, real_latent(instrument A, velocity V, pitch P2) )
```

No adversary, no minimax, no equilibrium to balance. Just a regression with a ground-truth target.
The frozen DAC decoder turns the predicted latent back into audio. Crucially, the decoder is **not**
in the training gradient path - the loss is computed entirely in latent space, which makes training
cheap and fast.

This is the architecture the README documents. But making it *actually* produce clean audio took two
more insights that had nothing to do with the model.

---

## 6. Making it work was a data problem, then a loss problem

### 5a. The hidden noise floor (a data problem)

My first pairs came from different *recordings* of the "same" instrument at different pitches. Val
loss refused to drop below **~2.37**, no matter the model. The reason wasn't the model at all: two
real recordings of the same piano differ in more than pitch - room acoustics, mic placement, string
coupling, take-to-take variation. My "translate only pitch" target secretly contained all that
*other* variation too, an irreducible noise floor the network couldn't (and shouldn't) fit.

**The fix was to change the data, not the model.** I switched to a physical-modelling **VST** and
generated samples where, within each `(piano_type, velocity)` group, *only pitch varies* - same dry
signal path, no reverb, deterministic synthesis. Now the difference between a source and target
latent is *purely* the thing I want to learn. The noise floor vanished.

This is also why the dataset pairs are grouped by `(instrument, velocity)`: it's not a convenience,
it's the guarantee that the supervision signal is clean. (See [README §4](../README.md#4-data-pipeline).)

### 5b. The decay "buzz" (a loss problem)

With clean data the model learned pitch transfer - but translated notes had a faint, fixed tonal
**buzz** in their decay tails. I traced it to the loss. A flat L1 on the latent under-weights
low-energy frames: in a 2-second piano note, the long decay tail has tiny magnitude, so its absolute
error is tiny *even when its direction is completely wrong*. The optimiser had no reason to fix those
frames, and a small learned bias-delta leaked into them and decoded as a steady buzz.

**The fix was to change what the loss measures.** I replaced flat L1 with a **per-frame
cosine-direction + log-magnitude** loss:

- *direction*: `1 − cosine_similarity` per frame, giving every frame equal directional pressure
  regardless of its energy - so quiet tail frames must point the right way too,
- *magnitude*: an L1 on `log‖z‖` per frame, restoring the amplitude envelope separately.

The buzz went away. (See [README §3](../README.md#3-training-objective) for the exact formulation.)

> The pattern across 5a and 5b is the same: when output is wrong, the instinct is to reach for a
> bigger or different model. Both times the real fix was upstream - in the *supervision signal* and
> in *what the objective actually rewards*. Identity training pairs (`src == tgt`, "change nothing")
> were added for the same reason: to explicitly teach the boundary case the model was otherwise
> getting subtly wrong.

---

## 7. The final architecture (briefly)

The full spec is in [README §2](../README.md#2-architecture-current-dac-pitch-adapter). The short
version:

```
source.wav → [DAC encode | frozen] → z_dac
           → PitchStripper  (near-identity, no-grad - a vestige of the disentanglement era)
           → PitchInjector(src_midi, tgt_midi)   ← the network that actually learns
           → [DAC decode | frozen] → note at target pitch
```

A few decisions worth calling out, each a direct descendant of a lesson above:

- **Condition on *both* source and target MIDI, not the interval.** The latent pitch transform is
  *not* interval-invariant: C2→C3 looks nothing like C6→C7 because of inharmonicity and
  register-dependent spectra. The network needs absolute endpoints, not just the shift.
- **Dilated FiLM blocks** (`[1,2,4,8,16,32]`) so the receptive field spans the 2-second window -
  pitch is a global property of the note, not a local one.
- **The PitchStripper survives as a no-grad near-identity.** It's an honest fossil of the
  disentanglement approach. I kept it for architectural symmetry rather than pretend the history
  didn't happen - and because it's the natural place to reintroduce learned timbre/pitch separation
  if a future version wants it.

---

## 8. Results

- **Validation loss 0.323** (cosine-direction + log-magnitude) at epoch 97.
- **~27.3M trainable parameters** in the adapter. The DAC codec (the bulk of the compute) stays
  frozen and untrained.
- **Dataset:** 1,896 train / 216 val DAC latents - 8 piano types × 3 velocities × 88 notes.
- **Training cost:** well under an hour on a single mid-range GPU (RTX 3090/4090). Because the
  decoder is out of the gradient path and audio is pre-encoded to latents once, each step is
  latent-only and cheap. Dev happens on Apple Silicon (MPS), and full runs go to a scripted vast.ai + B2
  GPU pipeline (see [README §6](../README.md#6-cloud-training-vastai--backblaze-b2)).

To hear it, generate a full instrument from one note:

```bash
python scripts/generate_instrument_dac.py \
    --source <one_note.wav> --source-midi 60 \
    --checkpoint checkpoints/dac_adapter/best_model.pt \
    --midi-low 21 --midi-high 108 --out-dir output/demo --sfz --diag
```

---

## 9. What I'd do next

- **High-frequency detail.** The latent loss is blind to fine spectral structure, which can leave
  translated notes slightly low-pass / "regressed to the mean." A multi-resolution **STFT loss** on a
  small decoded sub-batch (putting the decoder back in the gradient path for a handful of samples per
  step) directly targets the HF detail the latent loss can't see. *(In progress.)*
- **Velocity translation.** Velocity is currently only a *grouping key*. The adapter doesn't change
  it. The natural extension is to condition the injector on source/target velocity exactly the way it
  already conditions on pitch.
- **Beyond piano.** Nothing in the method is piano-specific - it needs only a codec and clean
  single-source pitch pairs. Other monophonic, pitched instruments are the obvious next domain.

---

## 10. Takeaways

1. **Sound quality was the selection filter, not a finishing touch.** Treating commercial-grade
   fidelity as the spec - and being willing to reject DDSP, RAVE, and the mel-vocoder paradigm
   outright on it - is what saved time. The cheapest way to ship great-sounding audio was to *not*
   put a lossy or low-fidelity audio model in the signal path in the first place.
2. **Reframing beat optimisation.** The single biggest jump came from realising I needed pitch
   *translation* (supervisable) rather than pitch *disentanglement* (an adversarial game I kept
   losing). No amount of λ/α tuning would have closed that gap.
3. **Borrow the hard part.** Freezing a pretrained codec deleted an entire category of problems I'd
   been fighting. Not every component needs to be yours.
4. **Most "model" bugs were data or loss bugs.** A ~2.37 noise floor (fixed by changing the data
   source) and a decay buzz (fixed by changing what the loss measures) both *looked* like model
   failures and were neither.
5. **Keep the dead ends visible.** The rejected approaches - three whole paradigms plus two of my own
   builds - live in [`legacy/`](../legacy/) and these docs on purpose. They're the most honest
   evidence of how the working design was reached.

---

### Pointers
- System reference & how-to-run: [README](../README.md)
- Adapter design log: [`docs/dac_adapter_plan.md`](dac_adapter_plan.md)
- The adversarial approach in full: [`docs/ADVERSARIAL_DISENTANGLEMENT_APPROACH.md`](ADVERSARIAL_DISENTANGLEMENT_APPROACH.md)
- Archived prior approaches: [`legacy/`](../legacy/) (+ its README)