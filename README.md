# Multi-Sample Instrument Generator

An AI-powered tool that generates complete multi-sample instruments from 1-5 input audio samples using latent diffusion models.

## Overview

This project uses a Variational Autoencoder (VAE) + Latent Diffusion Model approach to generate high-quality, multi-sample instruments with:
- Full chromatic coverage (MIDI notes C0-C8)
- Multiple velocity layers (4-8 layers)
- Timbral consistency with input samples
- Export to SFZ and SF2 formats
- Automatic loop point detection

## Features

- **Input**: 1-5 audio samples
- **Output**: Complete multi-sample instrument with hundreds of samples
- **Quality**: Professional audio fidelity (44.1kHz, 16-bit or higher)
- **Format**: Industry-standard SFZ/SF2 files
- **Platform**: Optimized for Apple Silicon (M1/M2/M3) with MPS backend

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Dataset

```bash
python scripts/preprocess_dataset.py --input data/raw --output data/processed
```

### 2. Train VAE

```bash
python scripts/train_vae.py --config config/model_config.yaml
```

### 3. Train Diffusion Model

```bash
python scripts/train_diffusion.py --config config/model_config.yaml
```

### 4. Generate Instrument

```bash
python scripts/generate.py \
  --input sample1.wav sample2.wav \
  --output my_instrument \
  --checkpoint checkpoints/diffusion_best.pt
```

## Project Structure

```
Multi-Salmon-Training/
├── config/              # Configuration files
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training loops and losses
│   ├── inference/      # Generation pipeline
│   └── export/         # SFZ/SF2 export and loop detection
├── scripts/            # Training and generation scripts
├── tests/              # Unit and integration tests
└── data/               # Dataset storage
```

## Technical Details

- **Framework**: PyTorch with MPS (Metal Performance Shaders)
- **Model Size**: ~260M parameters
- **Training**: ~100 epochs on 4000+ instrument dataset
- **Generation**: ~5-10 minutes per full instrument

## References

- Latent Diffusion Models (Rombach et al., 2022)
- AudioLDM, Stable Audio
- SFZ Format: https://sfzformat.com/
