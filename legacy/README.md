# legacy/ — archived prior approaches

This directory holds code from approaches the project tried and moved on from. It is kept
deliberately, not by neglect: the **iteration is part of the story** (see root
[`README.md` §8](../README.md#8-key-design-decisions--rationale-read-before-changing-the-model)
and the design docs under [`docs/`](../docs)). None of it is on the active path — the live
system is the DAC pitch adapter under `src/` + `scripts/`.

> **Status:** archived, not maintained. The core legacy *models* still import as a package
> (`from legacy.models.vqvae import VQVAE`), but some scripts reference an even earlier
> generation of modules that no longer exist and are retained only as a historical record.

## The three generations of this codebase

1. **Diffusion / Stable-Audio era** *(dead)* — the original direction (latent diffusion, a
   StableAudio-style VAE, a conditional UNet). Those modules were removed long ago; a few
   scripts here still `import` them (`diagnose_audio_quality.py`, `test_dac_reconstruction.py`,
   `prepare_piano_dataset.py`, `verify_training_data.py`) and will not run.
2. **VQ-VAE + latent transformer era** *(legacy, coherent)* — train a VQ-VAE on piano audio,
   then a transformer over its discrete latents, with an adversarial pitch classifier to try to
   disentangle pitch from timbre. This is where the **adversarial-disentanglement failure** was
   discovered (the classifier always dominated the stripper). That finding is *why* the active
   approach avoids adversarial training entirely.
3. **DAC pitch adapter** *(active — not here, see `src/`)* — freeze a pretrained codec, learn a
   small supervised latent-to-latent pitch translator. The current system.

## What's in here

```
legacy/
  models/    vqvae.py, quantizers.py, latent_transformer.py, pitch_adversary.py
  data/      vqvae_dataset.py, transformer_dataset.py
  training/  vqvae_trainer.py, transformer_trainer.py
  losses/    spectral_loss.py, disentanglement_loss.py, perceptual_losses.py
  config/    vqvae_config.yaml, latent_transformer_config.yaml, data_prep_config.json
  scripts/   train_vqvae.py, train_latent_transformer.py, evaluate_*, listen_vqvae.py,
             prepare_* , catalog_builder.py, resplit_dataset.py, … (+ dead diffusion-era scripts)
  tests/     test_transformer_pipeline.py
```

`vqvae.py` imports two shared building blocks (`FiLMConv1d`, `SinusoidalPitchEmbedding`) from
the **active** package (`src.models`), since those are reused by the current system.

## Related design docs
- [`docs/ADVERSARIAL_DISENTANGLEMENT_APPROACH.md`](../docs/ADVERSARIAL_DISENTANGLEMENT_APPROACH.md)
- [`docs/PITCH_NORMALIZATION_APPROACH.md`](../docs/PITCH_NORMALIZATION_APPROACH.md)
- [`docs/COLAB_TRAINING_GUIDE.md`](../docs/COLAB_TRAINING_GUIDE.md) (VQ-VAE Colab training)
- [`docs/SINGLE_SAMPLE_INSTRUMENT_FIX.md`](../docs/SINGLE_SAMPLE_INSTRUMENT_FIX.md)