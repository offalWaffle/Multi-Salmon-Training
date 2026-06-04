#!/usr/bin/env python3
"""
Diagnostic script to investigate why generated audio sounds garbled.

Based on plan.md suggestions:
1. Check DAC reconstruction quality
2. Analyze timestep-specific losses
3. Test unconditional vs conditional generation
4. Verify latent normalization
"""

import warnings
warnings.filterwarnings('ignore', message='.*distutils.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import soundfile as sf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.diffusion import LatentDiffusion
from src.models.unet import ConditionalUNet
from src.models.input_encoder import InputSampleEncoder
from src.data.dataset import LatentDataset, collate_fn_latents
from torch.utils.data import DataLoader


def load_models(device='mps'):
    """Load trained models and DAC."""
    print("Loading configurations...")
    with open('config/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)

    with open('config/training_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)

    with open('config/dac_latent_stats.yaml', 'r') as f:
        latent_stats = yaml.safe_load(f)

    config = {**model_config, **training_config, **latent_stats}

    # Load DAC
    print("Loading DAC model...")
    import dac
    model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(model_path)
    dac_model = dac_model.to(device)
    dac_model.eval()

    # Create models
    print("Creating model architectures...")
    unet = ConditionalUNet(
        in_channels=1024,
        model_channels=256,
        out_channels=1024,
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),
        attention_resolutions=(2, 4),
        embed_dim=config['diffusion']['time_embed_dim'],
        input_embedding_dim=128,
        dropout=0.1,
    )

    input_encoder = InputSampleEncoder(
        embed_dim=128,
        sample_rate=config['audio']['sample_rate'],
        duration=config['audio']['duration'],
        aggregation='attention',
        num_heads=4,
    )

    diffusion = LatentDiffusion(
        unet=unet,
        dac_model=dac_model,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
        prediction_type='epsilon',
        latent_mean=config['latent_mean'],
        latent_std=config['latent_std'],
    )

    # Load checkpoint
    print("Loading best model checkpoint...")
    checkpoint = torch.load('checkpoints/diffusion/best_model.pt', map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    input_encoder.load_state_dict(checkpoint['input_encoder_state_dict'])

    diffusion = diffusion.to(device)
    input_encoder = input_encoder.to(device)

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation loss: {checkpoint['best_val_loss']:.6f}")

    return diffusion, input_encoder, dac_model, config, checkpoint


def test_dac_reconstruction(dac_model, latent_dataset, device='mps', output_dir='outputs/diagnostics'):
    """Test 1: Verify DAC reconstruction quality."""
    print("\n" + "="*80)
    print("TEST 1: DAC Reconstruction Quality")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a sample
    sample = latent_dataset[0]
    latent = sample['latent'].unsqueeze(0).to(device)  # [1, 1024, 345]

    print(f"Latent shape: {latent.shape}")
    print(f"Latent range: [{latent.min():.2f}, {latent.max():.2f}]")
    print(f"Latent mean: {latent.mean():.4f}, std: {latent.std():.4f}")

    # Decode
    with torch.no_grad():
        audio = dac_model.decode(latent)

    audio_np = audio.squeeze().cpu().numpy()
    print(f"Decoded audio shape: {audio.shape}")
    print(f"Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")

    # Save
    output_path = output_dir / "dac_reconstruction.wav"
    sf.write(output_path, audio_np, 44100)
    print(f"✓ Saved DAC reconstruction to {output_path}")
    print("   → Listen to this file. If it sounds good, DAC is working correctly.")
    print("   → If it sounds bad, there may be an issue with the latent data itself.")

    return audio_np


def test_latent_statistics(diffusion, latent_dataset, device='mps'):
    """Test 2: Verify latent normalization is working correctly."""
    print("\n" + "="*80)
    print("TEST 2: Latent Normalization")
    print("="*80)

    # Get multiple samples
    latents = []
    for i in range(min(100, len(latent_dataset))):
        sample = latent_dataset[i]
        latents.append(sample['latent'])

    latents = torch.stack(latents)  # [N, 1024, 345]

    print(f"Computed from {len(latents)} samples:")
    print(f"  Raw latent mean: {latents.mean():.6f} (expected: {diffusion.latent_mean.item():.6f})")
    print(f"  Raw latent std:  {latents.std():.6f} (expected: {diffusion.latent_std.item():.6f})")

    # Normalize
    normalized = diffusion.normalize_latent(latents)
    print(f"\n  Normalized mean: {normalized.mean():.6f} (target: 0.0)")
    print(f"  Normalized std:  {normalized.std():.6f} (target: 1.0)")

    if abs(normalized.mean()) > 0.1 or abs(normalized.std() - 1.0) > 0.1:
        print("  ⚠️  WARNING: Normalization seems incorrect!")
        print("     Run scripts/compute_latent_stats.py to recompute statistics")
    else:
        print("  ✓ Normalization looks correct")


def test_unconditional_generation(diffusion, input_encoder, latent_dataset, device='mps', output_dir='outputs/diagnostics'):
    """Test 3: Generate samples with and without conditioning."""
    print("\n" + "="*80)
    print("TEST 3: Conditional vs Unconditional Generation")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diffusion.eval()
    input_encoder.eval()

    # Get input sample
    sample = latent_dataset[0]
    input_latents = sample['input_latents'].unsqueeze(0).to(device)
    num_input_samples = sample['num_input_samples'].unsqueeze(0).to(device)

    with torch.no_grad():
        input_embedding = input_encoder.forward_latents(input_latents, num_input_samples)

    # Test MIDI note
    midi_note = torch.tensor([0.5]).to(device)  # Middle C

    print("\nGenerating with conditioning...")
    print("  MIDI note: 0.5 (normalized)")
    print("  CFG guidance scale: 5.0")

    with torch.no_grad():
        # Conditional generation (with guidance)
        latent_cond = diffusion.ddim_sample(
            batch_size=1,
            midi_notes=midi_note,
            input_embeddings=input_embedding,
            latent_shape=(1024, 345),
            num_inference_steps=200,
            eta=0.0,
            guidance_scale=5.0,
            progress=True,
        )

        # Denormalize and decode
        latent_cond = diffusion.denormalize_latent(latent_cond)
        audio_cond = diffusion.dac_model.decode(latent_cond)

        # Save
        audio_cond_np = audio_cond.squeeze().cpu().numpy()
        sf.write(output_dir / "conditional_generation.wav", audio_cond_np, 44100)
        print(f"✓ Saved to {output_dir / 'conditional_generation.wav'}")

    print("\nGenerating WITHOUT conditioning (unconditional)...")
    with torch.no_grad():
        # Unconditional generation (no guidance)
        midi_uncond = torch.zeros_like(midi_note)
        emb_uncond = torch.zeros_like(input_embedding)

        latent_uncond = diffusion.ddim_sample(
            batch_size=1,
            midi_notes=midi_uncond,
            input_embeddings=emb_uncond,
            latent_shape=(1024, 345),
            num_inference_steps=200,
            eta=0.0,
            guidance_scale=1.0,  # No CFG
            progress=True,
        )

        # Denormalize and decode
        latent_uncond = diffusion.denormalize_latent(latent_uncond)
        audio_uncond = diffusion.dac_model.decode(latent_uncond)

        # Save
        audio_uncond_np = audio_uncond.squeeze().cpu().numpy()
        sf.write(output_dir / "unconditional_generation.wav", audio_uncond_np, 44100)
        print(f"✓ Saved to {output_dir / 'unconditional_generation.wav'}")

    print("\n  → Compare the two files:")
    print("     - If unconditional sounds better, conditioning may be causing issues")
    print("     - If both sound garbled, the core diffusion model isn't learning properly")
    print("     - If both sound similar, CFG might not be working")


def test_timestep_losses(diffusion, input_encoder, val_loader, device='mps'):
    """Test 4: Analyze losses across different timestep ranges."""
    print("\n" + "="*80)
    print("TEST 4: Timestep-Specific Loss Analysis")
    print("="*80)

    diffusion.eval()
    input_encoder.eval()

    timestep_ranges = [(0, 250), (250, 500), (500, 750), (750, 1000)]
    timestep_losses = {f"t{start}-{end}": [] for start, end in timestep_ranges}

    print("Analyzing validation set losses by timestep range...")

    num_batches = 0
    max_batches = 50  # Analyze subset for speed

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_batches:
                break

            latents = batch['latent'].to(device)
            input_latents = batch['input_latents'].to(device)
            midi_notes = batch['midi_note'].to(device)
            num_input_samples = batch['num_input_samples'].to(device)

            # Encode input
            input_embeddings = input_encoder.forward_latents(input_latents, num_input_samples)

            # Forward pass
            loss, timesteps, loss_per_sample = diffusion.forward_latents(
                latents, midi_notes, input_embeddings, return_timesteps=True
            )

            # Track by timestep
            timesteps_np = timesteps.cpu().numpy()
            loss_per_sample_np = loss_per_sample.cpu().numpy()

            for t_val, loss_val in zip(timesteps_np, loss_per_sample_np):
                for start, end in timestep_ranges:
                    if start <= t_val < end:
                        timestep_losses[f"t{start}-{end}"].append(float(loss_val))

            num_batches += 1

    print(f"\nAnalyzed {num_batches} batches:")
    for range_name, losses in timestep_losses.items():
        if losses:
            avg = np.mean(losses)
            std = np.std(losses)
            print(f"  {range_name}: {avg:.4f} ± {std:.4f}")

    print("\n  → Analysis:")
    losses_list = [np.mean(losses) for losses in timestep_losses.values() if losses]
    if max(losses_list) - min(losses_list) > 0.2:
        print("     ⚠️  Large variation across timesteps detected!")
        print("     Consider implementing timestep weighting (see plan.md)")
    else:
        print("     ✓ Losses are relatively balanced across timesteps")


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load models
    diffusion, input_encoder, dac_model, config, checkpoint = load_models(device)

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = LatentDataset(
        data_dir='data/processed_latents/val',
        num_input_samples=5,
        normalize_midi=True,
        normalize_velocity=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_latents,
    )
    print(f"✓ Loaded {len(val_dataset)} validation samples")

    # Run tests
    test_dac_reconstruction(dac_model, val_dataset, device)
    test_latent_statistics(diffusion, val_dataset, device)
    test_unconditional_generation(diffusion, input_encoder, val_dataset, device)
    test_timestep_losses(diffusion, input_encoder, val_loader, device)

    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("\nNext steps based on plan.md:")
    print("1. Listen to outputs/diagnostics/dac_reconstruction.wav")
    print("   - If garbled: Issue with latent data or DAC")
    print("   - If clear: DAC is fine, issue is with diffusion training")
    print()
    print("2. Listen to conditional vs unconditional generations")
    print("   - If unconditional better: Conditioning is hurting performance")
    print("   - If both garbled: Core diffusion model needs more training or architecture changes")
    print()
    print("3. Check timestep loss analysis")
    print("   - If unbalanced: Implement timestep weighting")
    print("   - If balanced but high: Model needs more capacity or different architecture")
    print()
    print("4. Current validation loss: {:.6f} (target: < 0.5 for good quality)".format(
        checkpoint['best_val_loss']
    ))
    print("   - Loss is still quite high, suggesting model hasn't converged to good quality")
    print()
    print("Possible solutions from plan.md:")
    print("- Try unconditional training mode (set unconditional: true in config)")
    print("- Implement timestep weighting (focus on harder timesteps)")
    print("- Increase CFG dropout from 0.1 to 0.2-0.3")
    print("- Try v-prediction instead of epsilon prediction")
    print("- Increase model capacity")
    print("- Train for more epochs with reduced learning rate")


if __name__ == "__main__":
    main()
