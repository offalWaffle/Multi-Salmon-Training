#!/usr/bin/env python3
"""
Re-split existing processed dataset with instrument-aware splitting.
This is faster than reprocessing from scratch.
"""

import os
import sys
import json
import random
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    processed_dir = Path('data/processed')
    train_dir = processed_dir / 'train'
    val_dir = processed_dir / 'val'

    print("=" * 80)
    print("RE-SPLIT DATASET WITH INSTRUMENT-AWARE SPLITTING")
    print("=" * 80)
    print("\nThis will:")
    print("1. Load all existing processed samples")
    print("2. Re-split with instrument awareness (≥2 samples per instrument per split)")
    print("3. Exclude instruments with <4 total samples")
    print("4. Backup original split to data/processed.backup/")
    print()

    # Check if processed data exists
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Error: Processed data not found")
        print(f"   Looking for: {train_dir} and {val_dir}")
        print("\nPlease run preprocessing first:")
        print("   python scripts/preprocess_dataset.py")
        return

    # Backup existing split
    backup_dir = Path('data/processed.backup')
    if backup_dir.exists():
        print(f"⚠️  Backup directory already exists: {backup_dir}")
        response = input("Delete existing backup and continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(backup_dir)

    print(f"\n1. Creating backup...")
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(train_dir, backup_dir / 'train')
    shutil.copytree(val_dir, backup_dir / 'val')
    if (processed_dir / 'sample_catalog.csv').exists():
        shutil.copy(processed_dir / 'sample_catalog.csv', backup_dir / 'sample_catalog.csv')
    print(f"   ✓ Backed up to {backup_dir}")

    # Load all samples
    print(f"\n2. Loading existing samples...")
    all_samples = []

    # Load from both train and val
    for split_dir in [train_dir, val_dir]:
        sample_files = sorted(split_dir.glob('sample_*.pt'))
        for sample_file in tqdm(sample_files, desc=f"   Loading {split_dir.name}"):
            sample_data = torch.load(sample_file)
            all_samples.append(sample_data)

    print(f"   ✓ Loaded {len(all_samples)} total samples")

    # Group by instrument
    print(f"\n3. Grouping by instrument...")
    instrument_samples = defaultdict(list)
    for sample in all_samples:
        instrument_samples[sample['instrument_id']].append(sample)

    print(f"   ✓ Found {len(instrument_samples)} unique instruments")

    # Instrument-aware split
    print(f"\n4. Performing instrument-aware split...")
    print(f"   Minimum samples per instrument per split: 2")
    print(f"   Train split ratio: 0.9")

    random.seed(42)
    train_samples = []
    val_samples = []
    excluded_instruments = []

    for instrument_id, samples in instrument_samples.items():
        num_samples = len(samples)

        # Skip instruments with fewer than 4 samples
        if num_samples < 4:
            excluded_instruments.append((instrument_id, num_samples))
            print(f"   ⏭️  Excluding instrument {instrument_id}: only {num_samples} samples (need ≥4)")
            continue

        # Shuffle this instrument's samples
        random.shuffle(samples)

        # Calculate split ensuring at least 2 in each set
        split_idx = max(2, int(num_samples * 0.9))
        # Make sure val also has at least 2
        if num_samples - split_idx < 2:
            split_idx = num_samples - 2

        train_samples.extend(samples[:split_idx])
        val_samples.extend(samples[split_idx:])

    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"\n   Split results:")
    print(f"     Instruments used: {len(instrument_samples) - len(excluded_instruments)}")
    print(f"     Instruments excluded: {len(excluded_instruments)}")
    print(f"     Train samples: {len(train_samples)}")
    print(f"     Val samples: {len(val_samples)}")

    if excluded_instruments:
        print(f"\n   Excluded instrument details:")
        for inst_id, num in sorted(excluded_instruments):
            print(f"     - Instrument {inst_id}: {num} samples")

    # Clear existing train/val directories
    print(f"\n5. Saving new split...")
    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Save train samples
    for i, sample in enumerate(tqdm(train_samples, desc="   Saving train")):
        sample_path = train_dir / f'sample_{i:06d}.pt'
        torch.save(sample, sample_path)

    # Save val samples
    for i, sample in enumerate(tqdm(val_samples, desc="   Saving val")):
        sample_path = val_dir / f'sample_{i:06d}.pt'
        torch.save(sample, sample_path)

    # Update metadata
    train_metadata = {
        'num_samples': len(train_samples),
        'sample_rate': 44100,
        'duration': 4.0,
        'num_instruments': len(set(s['instrument_id'] for s in train_samples))
    }
    val_metadata = {
        'num_samples': len(val_samples),
        'sample_rate': 44100,
        'duration': 4.0,
        'num_instruments': len(set(s['instrument_id'] for s in val_samples))
    }

    with open(train_dir / 'metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2)

    with open(val_dir / 'metadata.json', 'w') as f:
        json.dump(val_metadata, f, indent=2)

    print(f"\n   ✓ Saved {len(train_samples)} train samples")
    print(f"   ✓ Saved {len(val_samples)} val samples")

    print("\n" + "=" * 80)
    print("✓ RE-SPLIT COMPLETE!")
    print("=" * 80)
    print(f"Original data backed up to: {backup_dir}")
    print(f"New split:")
    print(f"  Train: {len(train_samples)} samples ({train_metadata['num_instruments']} instruments)")
    print(f"  Val:   {len(val_samples)} samples ({val_metadata['num_instruments']} instruments)")
    print()
    print("Next steps:")
    print("1. Verify no validation NaN:")
    print("   python scripts/debug_validation_all_batches.py")
    print()
    print("2. Resume training:")
    print("   python scripts/train_diffusion.py --resume checkpoints/diffusion/checkpoint_epoch006.pt")
    print()
    print("Or start fresh:")
    print("   python scripts/train_diffusion.py")


if __name__ == '__main__':
    main()
