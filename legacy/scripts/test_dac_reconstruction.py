#!/usr/bin/env python3
"""Test DAC encode/decode reconstruction quality."""

import torch
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import InstrumentDataset
from src.data.audio_utils import save_audio

print("="*80)
print("Testing DAC Reconstruction Quality")
print("="*80)

# Load DAC
import dac
device = 'mps'
model_path = dac.utils.download(model_type="44khz")
dac_model = dac.DAC.load(model_path)
dac_model = dac_model.to(device)
dac_model.eval()

# Load a sample from training data
train_dataset = InstrumentDataset(
    data_dir='data/processed/train',
    num_input_samples=5,
    normalize_midi=True,
    normalize_velocity=True,
)

print(f"\nLoaded {len(train_dataset)} training samples")

# Test reconstruction on a few samples
print("\nTesting DAC encode/decode on training samples:")
print("-" * 80)

for idx in [0, 10, 50]:
    sample = train_dataset[idx]
    audio = sample['audio'].unsqueeze(0).to(device)  # [1, 1, 176400]

    print(f"\nSample {idx}:")
    print(f"  Original audio shape: {audio.shape}")
    print(f"  Original range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

    with torch.no_grad():
        # Encode
        z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(audio)
        print(f"  Latent z shape: {z.shape}")
        print(f"  Latent z range: [{z.min().item():.4f}, {z.max().item():.4f}]")

        # Decode
        reconstructed = dac_model.decode(z)
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Reconstructed range: [{reconstructed.min().item():.4f}, {reconstructed.max().item():.4f}]")

        # Calculate reconstruction error (handle size mismatch)
        min_len = min(audio.shape[2], reconstructed.shape[2])
        mse = torch.mean((audio[:, :, :min_len] - reconstructed[:, :, :min_len])**2).item()
        print(f"  MSE: {mse:.6f}")
        if audio.shape[2] != reconstructed.shape[2]:
            print(f"  ⚠️  Size mismatch! Input: {audio.shape[2]}, Output: {reconstructed.shape[2]}")

        # Save for listening
        save_audio(audio[0], f'/tmp/original_{idx}.wav', sample_rate=44100)
        save_audio(reconstructed[0], f'/tmp/reconstructed_{idx}.wav', sample_rate=44100)

print(f"\n✓ Saved audio files to /tmp/original_*.wav and /tmp/reconstructed_*.wav")
print("  Listen to these to verify DAC reconstruction quality")

# Now test what the model generates vs random noise
print("\n" + "="*80)
print("Testing generated latents from diffusion model")
print("="*80)

from src.models.diffusion import LatentDiffusion
from src.models.unet import ConditionalUNet
from src.models.input_encoder import InputSampleEncoder
import yaml

# Load config
with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

# Load checkpoint
checkpoint_path = 'checkpoints/diffusion/checkpoint_epoch010.pt'
print(f"\nLoading checkpoint: {checkpoint_path}")

# Create models
unet = ConditionalUNet(
    in_channels=1024,
    model_channels=256,
    out_channels=1024,
    num_res_blocks=2,
    channel_mult=(1, 2, 3, 4),
    attention_resolutions=(2, 4),
    embed_dim=model_config['diffusion']['time_embed_dim'],
    input_embedding_dim=128,
    dropout=0.1,
).to(device)

input_encoder = InputSampleEncoder(
    embed_dim=128,
    sample_rate=44100,
    duration=4.0,
    aggregation='attention',
    num_heads=4,
).to(device)

diffusion = LatentDiffusion(
    unet=unet,
    dac_model=dac_model,
    num_timesteps=model_config['diffusion']['num_timesteps'],
    beta_schedule=model_config['diffusion']['beta_schedule'],
    prediction_type='epsilon',
).to(device)

# Load weights
checkpoint = torch.load(checkpoint_path, map_location=device)
unet.load_state_dict(checkpoint['unet_state_dict'])
input_encoder.load_state_dict(checkpoint['input_encoder_state_dict'])

diffusion.eval()
input_encoder.eval()

print("✓ Models loaded")

# Get input embedding
batch = train_dataset[0]
input_samples = batch['input_samples'].unsqueeze(0).to(device)
num_input_samples = batch['num_input_samples'].unsqueeze(0).to(device)

with torch.no_grad():
    input_embedding = input_encoder(input_samples, num_input_samples)

    # Generate with model
    print("\nGenerating with diffusion model...")
    generated_latent = diffusion.ddim_sample(
        batch_size=1,
        midi_notes=torch.tensor([0.5]).to(device),
        velocities=torch.tensor([0.7]).to(device),
        input_embeddings=input_embedding,
        latent_shape=(1024, 344),
        num_inference_steps=50,
        eta=0.0,
        progress=False,
    )

    print(f"  Generated latent shape: {generated_latent.shape}")
    print(f"  Generated latent range: [{generated_latent.min().item():.4f}, {generated_latent.max().item():.4f}]")
    print(f"  Generated latent mean: {generated_latent.mean().item():.4f}")
    print(f"  Generated latent std: {generated_latent.std().item():.4f}")

    # Decode to audio
    generated_audio = dac_model.decode(generated_latent)
    print(f"  Generated audio shape: {generated_audio.shape}")
    print(f"  Generated audio range: [{generated_audio.min().item():.4f}, {generated_audio.max().item():.4f}]")

    save_audio(generated_audio[0], '/tmp/model_generated.wav', sample_rate=44100)

# Compare with random latent
print("\nComparing with random noise latent...")
with torch.no_grad():
    random_latent = torch.randn(1, 1024, 344).to(device)
    print(f"  Random latent range: [{random_latent.min().item():.4f}, {random_latent.max().item():.4f}]")

    random_audio = dac_model.decode(random_latent)
    print(f"  Random audio range: [{random_audio.min().item():.4f}, {random_audio.max().item():.4f}]")

    save_audio(random_audio[0], '/tmp/random_noise.wav', sample_rate=44100)

print("\n" + "="*80)
print("Test files saved:")
print("  /tmp/original_*.wav - Real training samples")
print("  /tmp/reconstructed_*.wav - DAC reconstructed samples")
print("  /tmp/model_generated.wav - Model generated (epoch 10)")
print("  /tmp/random_noise.wav - Random noise through DAC")
print("\nListen to these files to diagnose the issue!")
print("="*80)
