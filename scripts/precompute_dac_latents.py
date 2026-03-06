#!/usr/bin/env python3
"""
Pre-compute DAC latents for all audio samples in the dataset.
This dramatically speeds up training by avoiding re-encoding the same audio every batch.
"""

import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import dac

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def precompute_latents(data_dir, output_dir, device='mps'):
    """
    Pre-compute and save DAC latents for all samples in the dataset.

    Args:
        data_dir: Directory containing processed audio samples
        output_dir: Directory to save latent files
        device: Device to use for encoding ('mps', 'cuda', or 'cpu')
    """
    # Load DAC model
    print("Loading DAC model...")
    model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(model_path)
    dac_model = dac_model.to(device)
    dac_model.eval()
    print(f"✓ DAC model loaded on {device}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (to get list of all files)
    print(f"\nScanning dataset: {data_dir}")
    data_dir = Path(data_dir)

    # Find all .pt files (processed audio samples)
    audio_files = sorted(data_dir.glob("*.pt"))
    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print(f"ERROR: No .pt files found in {data_dir}")
        return

    # Process each file
    print(f"\nPre-computing DAC latents...")
    stats = {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'std': 0}

    with torch.no_grad():
        for audio_file in tqdm(audio_files, desc="Encoding samples"):
            # Load audio
            data = torch.load(audio_file, map_location='cpu')
            audio = data['audio']  # [channels, samples]

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Add batch dimension: [channels, samples] -> [1, channels, samples]
            audio = audio.unsqueeze(0).to(device)

            # Encode to DAC latent
            z, _, _, _, _ = dac_model.encode(audio)
            # z is [1, 1024, time]

            # Move to CPU for saving
            z = z.cpu()

            # Update statistics
            stats['min'] = min(stats['min'], z.min().item())
            stats['max'] = max(stats['max'], z.max().item())
            stats['mean'] += z.mean().item()
            stats['std'] += z.std().item()

            # Save latent with same metadata
            output_file = output_dir / audio_file.name
            torch.save({
                'latent': z.squeeze(0),  # Save as [1024, time]
                'midi_note': data['midi_note'],
                'velocity': data['velocity'],
                'instrument_id': data['instrument_id'],
                'file_path': data.get('file_path', 'unknown'),
                'original_audio_path': str(audio_file)
            }, output_file)

    # Compute final statistics
    n = len(audio_files)
    stats['mean'] /= n
    stats['std'] /= n

    print(f"\n✓ Pre-computed {n} latent files")
    print(f"  Output directory: {output_dir}")
    print(f"\nLatent statistics:")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")

    # Save statistics for normalization
    stats_file = output_dir.parent / f"{output_dir.name}_stats.pt"
    torch.save(stats, stats_file)
    print(f"\n✓ Saved statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute DAC latents for training dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing processed audio samples')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save latent files')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'cuda', 'cpu'],
                       help='Device to use for encoding')

    args = parser.parse_args()

    precompute_latents(args.data_dir, args.output_dir, args.device)
    print("\n✓ Pre-computation complete!")
    print(f"\nNext steps:")
    print(f"1. Update training config to use latent dataset")
    print(f"2. Resume training (will be 10-20x faster!)")


if __name__ == "__main__":
    main()
