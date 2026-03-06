#!/usr/bin/env python3
"""
Prepare dataset for VQ-VAE training from sample_catalog.csv.

Reads data/processed/sample_catalog.csv (produced by catalog_builder.py),
filters to a chosen instrument, applies octave-offset corrections, creates
stratified train/val/test splits, and writes JSON metadata files for
VQVAEDataset.

Usage:
    # List available instruments in catalog
    python scripts/prepare_vqvae_data.py --list-instruments

    # Prepare a specific instrument (default: Baby Grand Piano)
    python scripts/prepare_vqvae_data.py --instrument "Baby Grand Piano"

    # Override output directory
    python scripts/prepare_vqvae_data.py --instrument "Strings" --output data/vqvae_strings
"""

import argparse
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_CATALOG = project_root / 'data' / 'processed' / 'sample_catalog.csv'


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def load_catalog(catalog_path):
    """Load the full sample catalog, exiting with a helpful message if absent."""
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found at {catalog_path}")
        print("Run scripts/catalog_builder.py first to generate the catalog.")
        sys.exit(1)
    df = pd.read_csv(catalog_path, low_memory=False)
    print(f"Loaded catalog: {len(df)} samples across {df['instrument'].nunique()} instruments "
          f"from {catalog_path}")
    return df


def list_instruments(df):
    """Print all instruments available in the catalog, sorted by sample count."""
    print("\nAvailable instruments in catalog:")
    counts = (
        df.groupby(['library', 'instrument'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    for _, row in counts.iterrows():
        print(f"  [{row['library']}] {row['instrument']}: {row['count']} samples")


def filter_instrument(df, instrument_filter):
    """Return rows whose instrument name contains instrument_filter (case-insensitive)."""
    filtered = df[df['instrument'].str.contains(instrument_filter, case=False, na=False)]
    if filtered.empty:
        print(f"ERROR: No instruments matching '{instrument_filter}' found in catalog.")
        print("Use --list-instruments to see available instruments.")
        sys.exit(1)
    print(f"\nFiltered to {len(filtered)} samples matching '{instrument_filter}':")
    for instrument in sorted(filtered['instrument'].unique()):
        count = len(filtered[filtered['instrument'] == instrument])
        print(f"  - {instrument}: {count} samples")
    return filtered


# ---------------------------------------------------------------------------
# Sample building
# ---------------------------------------------------------------------------

def _corrected_midi_note(midi_note, octave_offset):
    """
    Apply octave offset detected by catalog_builder's pyin verification.

    The catalog stores the SFZ-declared midi_note and the offset between
    what pyin detected and what the SFZ declared:
        octave_offset = round((detected_midi - sfz_midi) / 12)

    The true sounding pitch is therefore:
        corrected = sfz_midi + octave_offset * 12
    """
    corrected = int(midi_note) + int(octave_offset) * 12
    return max(0, min(127, corrected))


def build_samples(df):
    """
    Convert filtered catalog rows to sample dicts compatible with VQVAEDataset.

    Applies octave-offset correction to midi_note and skips rows whose audio
    file no longer exists on disk.
    """
    samples = []
    missing = 0

    for _, row in df.iterrows():
        path = Path(str(row['path']))
        if not path.exists():
            missing += 1
            continue

        midi_note = _corrected_midi_note(row['midi_note'], row.get('octave_offset', 0))

        samples.append({
            'path': str(path),
            'filename': row['filename'],
            'instrument': row['instrument'],
            'midi_note': midi_note,
            'velocity': int(row['velocity_mid']),
            'lovel': int(row['lovel']) if pd.notna(row['lovel']) else 0,
            'hivel': int(row['hivel']) if pd.notna(row['hivel']) else 127,
            'has_loop': bool(row['has_loop']),
            'duration': float(row['duration']) if pd.notna(row['duration']) else None,
            'sample_rate': int(row['sample_rate']) if pd.notna(row['sample_rate']) else None,
        })

    if missing:
        print(f"WARNING: {missing} samples skipped — audio files not found on disk")

    return samples


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _midi_to_note(midi):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{notes[midi % 12]}{(midi // 12) - 1}"


def analyze_dataset(samples):
    """Print dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    by_instrument = defaultdict(list)
    for s in samples:
        by_instrument[s['instrument']].append(s)

    print(f"\nTotal samples: {len(samples)}")
    print("\nBy instrument variant:")
    for instrument, samps in sorted(by_instrument.items()):
        print(f"  {instrument}: {len(samps)}")

    midi_notes = [s['midi_note'] for s in samples]
    print(f"\nMIDI note range: {min(midi_notes)} ({_midi_to_note(min(midi_notes))}) "
          f"— {max(midi_notes)} ({_midi_to_note(max(midi_notes))})")

    note_counts = defaultdict(int)
    for s in samples:
        note_counts[s['midi_note']] += 1
    print(f"Unique notes: {len(note_counts)}  "
          f"samples/note: min={min(note_counts.values())} "
          f"max={max(note_counts.values())} "
          f"mean={np.mean(list(note_counts.values())):.1f}")

    velocities = [s['velocity'] for s in samples]
    distinct_vel = len(set(velocities))
    print(f"\nVelocity range: {min(velocities)} — {max(velocities)} ({distinct_vel} distinct values)")

    durations = [s['duration'] for s in samples if s['duration'] is not None]
    if durations:
        print(f"\nDuration:  min={min(durations):.2f}s  max={max(durations):.2f}s  "
              f"mean={np.mean(durations):.2f}s  median={np.median(durations):.2f}s")

    loops = sum(1 for s in samples if s['has_loop'])
    print(f"\nLoop points: {loops}/{len(samples)} samples ({loops / len(samples) * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def create_splits(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Stratified train/val/test split by MIDI note.

    Each note's samples are split proportionally so every pitch is represented
    in all three splits.
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    np.random.seed(seed)

    by_note = defaultdict(list)
    for s in samples:
        by_note[s['midi_note']].append(s)

    train, val, test = [], [], []
    for note_samples in by_note.values():
        idx = np.random.permutation(len(note_samples))
        n_train = max(1, int(len(note_samples) * train_ratio))
        n_val = max(1, int(len(note_samples) * val_ratio))
        train.extend(note_samples[i] for i in idx[:n_train])
        val.extend(note_samples[i] for i in idx[n_train:n_train + n_val])
        test.extend(note_samples[i] for i in idx[n_train + n_val:])

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    total = len(samples)
    print("\n" + "=" * 70)
    print("SPLITS")
    print("=" * 70)
    print(f"  Train: {len(train):>5}  ({len(train) / total * 100:.1f}%)")
    print(f"  Val:   {len(val):>5}  ({len(val) / total * 100:.1f}%)")
    print(f"  Test:  {len(test):>5}  ({len(test) / total * 100:.1f}%)")

    train_notes = {s['midi_note'] for s in train}
    all_notes = {s['midi_note'] for s in samples}
    missing_notes = all_notes - train_notes
    if missing_notes:
        print(f"  WARNING: {len(missing_notes)} notes absent from train split: {sorted(missing_notes)}")
    else:
        print(f"  All {len(all_notes)} notes present in train split ✓")

    return {'train': train, 'val': val, 'test': test}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_splits(splits, output_dir):
    """Write train/val/test JSON files and a dataset_summary.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)

    for split_name, split_samples in splits.items():
        out = output_dir / f'{split_name}.json'
        with open(out, 'w') as f:
            json.dump(split_samples, f, indent=2)
        print(f"  ✓ {split_name}.json  ({len(split_samples)} samples)  →  {out}")

    all_samples = [s for split in splits.values() for s in split]
    summary = {
        'total_samples': len(all_samples),
        'splits': {name: len(samps) for name, samps in splits.items()},
        'midi_note_range': [
            min(s['midi_note'] for s in all_samples),
            max(s['midi_note'] for s in all_samples),
        ],
        'instruments': sorted({s['instrument'] for s in all_samples}),
    }
    summary_path = output_dir / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ dataset_summary.json  →  {summary_path}")

    return output_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare VQ-VAE training data from sample_catalog.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--catalog',
        default=str(DEFAULT_CATALOG),
        help=f'Path to sample_catalog.csv (default: {DEFAULT_CATALOG})',
    )
    parser.add_argument(
        '--instrument',
        default='Baby Grand Piano',
        help='Instrument name filter — substring match, case-insensitive '
             '(default: "Baby Grand Piano")',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output directory for JSON splits. '
             'Default: data/vqvae_<slugified instrument name>',
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='Training fraction (default: 0.8)',
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.1,
        help='Validation fraction (default: 0.1). Remainder goes to test.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    parser.add_argument(
        '--list-instruments', action='store_true',
        help='Print all instruments in the catalog and exit',
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("VQ-VAE DATASET PREPARATION")
    print("=" * 70)

    df = load_catalog(args.catalog)

    if args.list_instruments:
        list_instruments(df)
        return

    filtered_df = filter_instrument(df, args.instrument)
    samples = build_samples(filtered_df)

    if not samples:
        print("ERROR: No valid samples after filtering. Aborting.")
        sys.exit(1)

    analyze_dataset(samples)
    splits = create_splits(samples, train_ratio=args.train_ratio,
                           val_ratio=args.val_ratio, seed=args.seed)

    if args.output is None:
        slug = args.instrument.lower().replace(' ', '_').replace('-', '_')
        output_dir = project_root / 'data' / f'vqvae_{slug}'
    else:
        output_dir = Path(args.output)

    output_dir = save_splits(splits, output_dir)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nDataset ready at: {output_dir}")
    print(f"\nUpdate config/vqvae_config.yaml data paths to:")
    print(f"  train_path: \"{output_dir.relative_to(project_root)}/train.json\"")
    print(f"  val_path:   \"{output_dir.relative_to(project_root)}/val.json\"")
    print(f"  test_path:  \"{output_dir.relative_to(project_root)}/test.json\"")
    print(f"\nThen run: python scripts/train_vqvae.py")
    print()


if __name__ == "__main__":
    main()
