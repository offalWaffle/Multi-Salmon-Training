#!/usr/bin/env python3
"""
Prepare Baby Grand Piano - Standard dataset for training.

This script:
1. Finds all Baby Grand Piano - Standard samples
2. Processes them into the standard format
3. Encodes them with Stable Audio VAE
4. Computes latent statistics
5. Splits into train/val sets
"""

import sys
from pathlib import Path
import shutil
import torch
import soundfile as sf
from tqdm import tqdm
import yaml
import numpy as np
from scipy import signal
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.stable_audio_vae import StableAudioVAE


def note_name_to_midi(note_name):
    """
    Convert note name to MIDI number.

    Examples:
        C4 -> 60 (middle C)
        C#4 -> 61
        Db4 -> 61
        A0 -> 21
        C8 -> 108
        B-1 -> 11

    Args:
        note_name: Note name like "C4", "C#4", "Db4", "B-1"

    Returns:
        MIDI note number (0-127)
    """
    # Note to semitone mapping (C=0)
    note_map = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11
    }

    note_name = note_name.strip().upper()

    # Parse note name
    note = note_name[0]
    rest = note_name[1:]

    # Handle accidentals
    accidental = 0
    if rest.startswith('#'):
        accidental = 1
        rest = rest[1:]
    elif rest.startswith('B'):  # Flat
        accidental = -1
        rest = rest[1:]

    # Parse octave (can be negative like B-1)
    octave = int(rest)

    # Calculate MIDI number: (octave + 1) * 12 + note + accidental
    # C-1 = 0, C0 = 12, C1 = 24, ..., C4 = 60
    midi = (octave + 1) * 12 + note_map[note] + accidental

    return midi


def find_piano_samples(source_dir):
    """Find all Baby Grand Piano - Standard WAV files."""
    piano_dir = Path(source_dir) / "Baby Grand Piano" / "Baby Grand Piano - Standard"

    if not piano_dir.exists():
        raise FileNotFoundError(f"Piano directory not found: {piano_dir}")

    wav_files = list(piano_dir.glob("*.wav"))
    wav_files = [f for f in wav_files if not f.name.startswith('._')]

    print(f"Found {len(wav_files)} Baby Grand Piano - Standard samples")
    return sorted(wav_files)


def load_midi_catalog(catalog_path='data/processed/sample_catalog.csv', velocity_filter=None, instrument_filter=None):
    """
    Load sample catalog with MIDI note and velocity information.

    Args:
        catalog_path: Path to sample catalog CSV
        velocity_filter: Tuple of (lovel, hivel) to filter samples, or None for all
        instrument_filter: Instrument name substring to filter (e.g., "Baby Grand"), or None for all

    Returns:
        Dictionary mapping file path -> {'midi_note': int, 'velocity': int}
    """
    try:
        df = pd.read_csv(catalog_path)
        total_count = len(df)

        # Filter by instrument if specified
        if instrument_filter is not None:
            df = df[df['instrument'].str.contains(instrument_filter, case=False, na=False)]
            print(f"✓ Instrument filter '{instrument_filter}': {len(df)}/{total_count} samples")

        # Filter by velocity range if specified (using velocity midpoint)
        if velocity_filter is not None:
            vel_min, vel_max = velocity_filter
            before_count = len(df)
            df = df[(df['velocity_mid'] >= vel_min) & (df['velocity_mid'] <= vel_max)]
            print(f"✓ Velocity filter [mid: {vel_min}-{vel_max}]: {len(df)}/{before_count} samples")

        # Create lookup dict: file path -> {midi_note, velocity}
        catalog_lookup = {}
        for _, row in df.iterrows():
            path = str(row['path'])
            catalog_lookup[path] = {
                'midi_note': int(row['midi_note']),
                'velocity': int(row['velocity_mid'])  # Use velocity midpoint
            }
        print(f"✓ Loaded catalog with {len(catalog_lookup)} samples")
        return catalog_lookup
    except Exception as e:
        print(f"⚠ Could not load catalog: {e}")
        return None


def process_and_encode_samples(wav_files, output_dir, device='mps', midi_catalog=None):
    """Process samples and encode with Stable Audio VAE."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only files in catalog (if catalog provided)
    if midi_catalog is not None:
        print(f"\nFound {len(wav_files)} WAV files on disk")
        filtered_files = []
        for wav_path in wav_files:
            resolved_path = str(wav_path.resolve())
            absolute_path = str(wav_path.absolute())
            if resolved_path in midi_catalog or absolute_path in midi_catalog:
                filtered_files.append(wav_path)

        skipped = len(wav_files) - len(filtered_files)
        if skipped > 0:
            print(f"  → {len(filtered_files)} files in catalog (using verified pitch/velocity values)")
            print(f"  → {skipped} files not in catalog or filtered by velocity (skipping)")
        wav_files = filtered_files

    # Load VAE
    print("\nLoading Stable Audio VAE...")
    vae = StableAudioVAE(device=device)

    # Target audio parameters
    sample_rate = 44100
    duration = 4.0
    target_length = int(sample_rate * duration)

    print(f"\nProcessing and encoding {len(wav_files)} samples...")
    print(f"Target: {duration}s at {sample_rate}Hz = {target_length} samples\n")

    processed = []

    for i, wav_path in enumerate(tqdm(wav_files, desc="Processing")):
        try:
            # Load audio with soundfile
            audio_np, sr = sf.read(str(wav_path), always_2d=False)

            # Resample if needed
            if sr != sample_rate:
                num_samples = int(len(audio_np) * sample_rate / sr)
                audio_np = signal.resample(audio_np, num_samples)

            # Convert to mono if stereo
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            # Pad or trim to target length
            if len(audio_np) < target_length:
                # Pad with zeros
                padding = target_length - len(audio_np)
                audio_np = np.pad(audio_np, (0, padding), mode='constant')
            else:
                # Trim
                audio_np = audio_np[:target_length]

            # Normalize audio
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.95

            # Convert to torch tensor [1, samples]
            audio = torch.from_numpy(audio_np).float().unsqueeze(0)

            # Add batch dimension and move to device
            audio = audio.unsqueeze(0).to(device)

            # Encode with VAE
            with torch.no_grad():
                latent = vae.encode(audio)

            # Move back to CPU for saving
            latent = latent.squeeze(0).cpu()

            # Extract MIDI note and velocity - try catalog first, then filename parsing
            midi_note = None
            velocity = 95  # Default to mf velocity

            # Try catalog lookup (try both absolute and resolved paths due to symlinks)
            if midi_catalog is not None:
                # Try resolved path first (follows symlinks)
                resolved_path = str(wav_path.resolve())
                absolute_path = str(wav_path.absolute())

                if resolved_path in midi_catalog:
                    midi_note = midi_catalog[resolved_path]['midi_note']
                    velocity = midi_catalog[resolved_path]['velocity']
                elif absolute_path in midi_catalog:
                    midi_note = midi_catalog[absolute_path]['midi_note']
                    velocity = midi_catalog[absolute_path]['velocity']

            # All files should be in catalog now (filtered upfront)
            # If somehow not found, use filename parsing as fallback
            if midi_note is None:
                try:
                    # Format: "Baby Grand - Standard-{NOTE}-{VELOCITY}.wav"
                    # E.g., "Baby Grand - Standard-C#4-113.wav"
                    parts = wav_path.stem.split('-')
                    if len(parts) >= 2:
                        note_name = parts[-2]  # Second from end is note name
                        midi_note = note_name_to_midi(note_name)
                except Exception as e:
                    print(f"\n⚠ Could not parse MIDI note from {wav_path.name}: {e}")
                    # Last resort: use sample index
                    midi_note = 21 + (i % 88)

            # Save processed data
            data = {
                'latent': latent,
                'midi_note': midi_note,
                'velocity': velocity,  # Use actual velocity from catalog
                'instrument_id': 0,  # Single instrument
                'file_path': str(wav_path),
            }

            output_path = output_dir / f"sample_{i:06d}.pt"
            torch.save(data, output_path)

            processed.append({
                'path': output_path,
                'midi_note': midi_note,
                'latent_shape': latent.shape,
            })

        except Exception as e:
            print(f"\nError processing {wav_path.name}: {e}")
            continue

    print(f"\n✓ Successfully processed {len(processed)} samples")
    return processed


def split_train_val(processed, output_base, train_ratio=0.9):
    """Split into train/val sets."""

    np.random.seed(42)
    indices = np.random.permutation(len(processed))
    split_idx = int(len(processed) * train_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dir = Path(output_base) / 'train'
    val_dir = Path(output_base) / 'val'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSplitting dataset:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")

    # Copy to train/val directories
    for i, idx in enumerate(tqdm(train_indices, desc="Copying to train")):
        src = processed[idx]['path']
        dst = train_dir / f"sample_{i:06d}.pt"
        shutil.copy2(src, dst)

    for i, idx in enumerate(tqdm(val_indices, desc="Copying to val")):
        src = processed[idx]['path']
        dst = val_dir / f"sample_{i:06d}.pt"
        shutil.copy2(src, dst)

    return train_dir, val_dir


def compute_latent_statistics(latent_dir):
    """Compute mean and std of latents."""

    latent_files = list(Path(latent_dir).glob('*.pt'))

    print(f"\nComputing latent statistics from {len(latent_files)} samples...")

    all_means = []
    all_stds = []

    # Sample 500 files or all if less
    sample_files = np.random.choice(latent_files, min(500, len(latent_files)), replace=False)

    for f in tqdm(sample_files, desc="Computing stats"):
        data = torch.load(f)
        latent = data['latent']

        all_means.append(latent.mean().item())
        all_stds.append(latent.std().item())

    mean = np.mean(all_means)
    std = np.mean(all_stds)

    print(f"\n{'='*70}")
    print("BABY GRAND PIANO LATENT STATISTICS")
    print(f"{'='*70}")
    print(f"Mean: {mean:.6f}")
    print(f"Std:  {std:.6f}")
    print(f"{'='*70}")

    return mean, std


def main():
    print("\n" + "="*70)
    print("BABY GRAND PIANO - STANDARD DATASET PREPARATION")
    print("="*70)

    # Configuration
    source_dir = Path('data/raw/equator2/sampler/wav/Keys')
    temp_dir = Path('data/piano_temp')
    output_dir = Path('data/piano_latents')

    # No velocity filter - use ALL velocities
    velocity_filter = None

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Using ALL velocity layers")

    # Step 1: Load MIDI catalog
    print("\n" + "-"*70)
    print("STEP 1: Loading MIDI note catalog")
    print("-"*70)
    midi_catalog = load_midi_catalog(
        velocity_filter=velocity_filter,
        instrument_filter="Baby Grand Piano - Standard"
    )

    # Step 2: Find samples
    print("\n" + "-"*70)
    print("STEP 2: Finding Baby Grand Piano samples")
    print("-"*70)
    wav_files = find_piano_samples(source_dir)

    # Step 3: Process and encode
    print("\n" + "-"*70)
    print("STEP 3: Processing and encoding samples")
    print("-"*70)
    processed = process_and_encode_samples(wav_files, temp_dir, device, midi_catalog)

    # Step 4: Split train/val
    print("\n" + "-"*70)
    print("STEP 4: Splitting train/validation sets")
    print("-"*70)
    train_dir, val_dir = split_train_val(processed, output_dir, train_ratio=0.9)

    # Step 5: Compute statistics
    print("\n" + "-"*70)
    print("STEP 5: Computing latent statistics")
    print("-"*70)
    mean, std = compute_latent_statistics(train_dir)

    # Step 6: Save statistics
    stats_file = Path('config/piano_latent_stats.yaml')
    with open(stats_file, 'w') as f:
        f.write("# Baby Grand Piano - Standard latent statistics\n")
        f.write(f"latent_mean: {mean:.6f}\n")
        f.write(f"latent_std: {std:.6f}\n")

    print(f"\n✓ Statistics saved to: {stats_file}")

    # Clean up temp directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nTrain samples: {train_dir}")
    print(f"Val samples:   {val_dir}")
    print(f"Statistics:    {stats_file}")
    print(f"\nReady to train with:")
    print(f"  python scripts/train_diffusion.py \\")
    print(f"    --use-latents \\")
    print(f"    --latent-train-dir {train_dir} \\")
    print(f"    --latent-val-dir {val_dir} \\")
    print(f"    --checkpoint-dir checkpoints/piano \\")
    print(f"    --model-config config/model_config_piano.yaml \\")
    print(f"    --training-config config/training_config.yaml")


if __name__ == "__main__":
    main()
