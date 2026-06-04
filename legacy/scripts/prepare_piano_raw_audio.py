#!/usr/bin/env python3
"""
Prepare Baby Grand Piano dataset for training with RAW AUDIO.
This allows pitch augmentation during training.

Unlike prepare_piano_dataset.py which pre-encodes to latents,
this script prepares data in a format that allows on-the-fly augmentation.
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


def load_midi_catalog(catalog_path='data/processed/sample_catalog.csv'):
    """Load sample catalog with MIDI note information."""
    try:
        df = pd.read_csv(catalog_path, low_memory=False)
        # Create lookup dict: file path -> MIDI note
        midi_lookup = {}
        for _, row in df.iterrows():
            path = str(row['path'])
            midi_lookup[path] = int(row['midi_note'])
        print(f"✓ Loaded catalog with {len(midi_lookup)} samples")
        return midi_lookup
    except Exception as e:
        print(f"⚠ Could not load catalog: {e}")
        return None


def find_piano_samples(source_dir):
    """Find all Baby Grand Piano - Standard WAV files."""
    piano_dir = Path(source_dir) / "Baby Grand Piano" / "Baby Grand Piano - Standard"

    if not piano_dir.exists():
        raise FileNotFoundError(f"Piano directory not found: {piano_dir}")

    wav_files = list(piano_dir.glob("*.wav"))
    wav_files = [f for f in wav_files if not f.name.startswith('._')]

    print(f"Found {len(wav_files)} Baby Grand Piano - Standard samples")
    return sorted(wav_files)


def process_samples(wav_files, output_dir, midi_catalog=None):
    """Process samples and save as raw audio .pt files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only files in catalog
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
            print(f"  → {len(filtered_files)} files in catalog (using verified pitch values)")
            print(f"  → {skipped} files not in SFZ/catalog (skipping)")
        wav_files = filtered_files

    # Target audio parameters
    sample_rate = 44100
    duration = 4.0
    target_length = int(sample_rate * duration)

    print(f"\nProcessing {len(wav_files)} samples...")
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
                padding = target_length - len(audio_np)
                audio_np = np.pad(audio_np, (0, padding), mode='constant')
            else:
                audio_np = audio_np[:target_length]

            # Normalize audio
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.95

            # Convert to torch tensor [1, samples] (mono)
            audio = torch.from_numpy(audio_np).float().unsqueeze(0)

            # Extract MIDI note from catalog
            midi_note = None

            if midi_catalog is not None:
                resolved_path = str(wav_path.resolve())
                absolute_path = str(wav_path.absolute())

                if resolved_path in midi_catalog:
                    midi_note = midi_catalog[resolved_path]
                elif absolute_path in midi_catalog:
                    midi_note = midi_catalog[absolute_path]

            if midi_note is None:
                print(f"\n⚠ Could not find MIDI note for {wav_path.name}")
                continue

            # Save processed data
            data = {
                'audio': audio,
                'midi_note': midi_note,
                'velocity': 100,  # Default velocity
                'instrument_id': 0,  # Single instrument
                'file_path': str(wav_path),
            }

            output_path = output_dir / f"sample_{i:06d}.pt"
            torch.save(data, output_path)

            processed.append({
                'path': output_path,
                'midi_note': midi_note,
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


def main():
    print("\n" + "="*70)
    print("BABY GRAND PIANO - RAW AUDIO DATASET PREPARATION")
    print("="*70)
    print("\nThis prepares data for training with pitch augmentation")

    # Configuration
    source_dir = Path('data/raw/equator2/sampler/wav/Keys')
    temp_dir = Path('data/piano_raw_temp')
    output_dir = Path('data/piano_raw')

    # Step 1: Load MIDI catalog
    print("\n" + "-"*70)
    print("STEP 1: Loading MIDI note catalog")
    print("-"*70)
    midi_catalog = load_midi_catalog()

    # Step 2: Find samples
    print("\n" + "-"*70)
    print("STEP 2: Finding Baby Grand Piano samples")
    print("-"*70)
    wav_files = find_piano_samples(source_dir)

    # Step 3: Process samples
    print("\n" + "-"*70)
    print("STEP 3: Processing samples (raw audio)")
    print("-"*70)
    processed = process_samples(wav_files, temp_dir, midi_catalog)

    # Step 4: Split train/val
    print("\n" + "-"*70)
    print("STEP 4: Splitting train/validation sets")
    print("-"*70)
    train_dir, val_dir = split_train_val(processed, output_dir, train_ratio=0.9)

    # Clean up temp directory
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nTrain samples: {train_dir}")
    print(f"Val samples:   {val_dir}")
    print(f"\nReady to train with pitch augmentation!")
    print(f"\nUse: bash scripts/train_piano_augmented.sh")


if __name__ == "__main__":
    main()
