#!/usr/bin/env python3
"""
Prepare VST piano samples for DAC adapter training.

Expects WAV files named: {PianoType}_{NoteName}_{MIDINote}_v{Velocity}.wav
Example: American_Small_Studio_A0_021_v040.wav

Pipeline:
    1. python scripts/prepare_vst_piano.py --source-dir /path/to/wavs --output-dir data/vst_piano
    2. python scripts/precompute_dac_latents.py --data-dir data/vst_piano/train --output-dir data/vst_latents/train
       python scripts/precompute_dac_latents.py --data-dir data/vst_piano/val   --output-dir data/vst_latents/val
    3. Update config/dac_adapter_config.yaml  latent_dir: data/vst_latents
    4. python scripts/train_dac_adapter.py
"""

import sys
import re
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import torch
import soundfile as sf
import numpy as np
from scipy import signal
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Filename pattern: {PianoType}_{NoteName}_{MIDINote:03d}_v{Velocity:03d}.wav
# NoteName uses either '#' or 's' for sharps, e.g. A#0 or As0
_PATTERN = re.compile(r'^(.+)_([A-G][#s]?[0-9])_(\d{3})_v(\d{3})\.wav$', re.IGNORECASE)

SAMPLE_RATE = 44100
DURATION    = 4.0          # seconds — latent dataset crops to 2 s during training
TARGET_LEN  = int(SAMPLE_RATE * DURATION)


def parse_filename(name):
    """Return (piano_type, note_name, midi_note, velocity) or None."""
    m = _PATTERN.match(name)
    if not m:
        return None
    piano_type = m.group(1)
    note_name  = m.group(2)
    midi_note  = int(m.group(3))
    velocity   = int(m.group(4))
    return piano_type, note_name, midi_note, velocity


def load_and_process(wav_path):
    """Load WAV → mono float32 tensor [1, TARGET_LEN]."""
    audio_np, sr = sf.read(str(wav_path), always_2d=False)

    # Resample if needed
    if sr != SAMPLE_RATE:
        n = int(len(audio_np) * SAMPLE_RATE / sr)
        audio_np = signal.resample(audio_np, n)

    # Mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Pad / trim
    if len(audio_np) < TARGET_LEN:
        audio_np = np.pad(audio_np, (0, TARGET_LEN - len(audio_np)))
    else:
        audio_np = audio_np[:TARGET_LEN]

    # Normalise
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.95

    return torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, T]


def scan_wavs(source_dir):
    """Return list of (wav_path, piano_type, note_name, midi_note, velocity)."""
    source_dir = Path(source_dir)
    wav_files  = sorted(source_dir.rglob('*.wav'))
    wav_files  = [f for f in wav_files if not f.name.startswith('._')]

    found, skipped = [], []
    for f in wav_files:
        parsed = parse_filename(f.name)
        if parsed is None:
            skipped.append(f.name)
        else:
            found.append((f, *parsed))

    if skipped:
        print(f"  ⚠  {len(skipped)} files skipped (name doesn't match pattern):")
        for name in skipped[:10]:
            print(f"      {name}")
        if len(skipped) > 10:
            print(f"      … and {len(skipped)-10} more")

    return found


def build_instrument_map(entries):
    """Assign a stable integer instrument_id to each unique piano type."""
    piano_types = sorted({e[1] for e in entries})
    return {pt: i for i, pt in enumerate(piano_types)}


def process_and_save(entries, instrument_map, out_dir):
    """Process all WAVs and save .pt files to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, (wav_path, piano_type, note_name, midi_note, velocity) in enumerate(
        tqdm(entries, desc=f'Processing → {out_dir.name}')
    ):
        try:
            audio = load_and_process(wav_path)
        except Exception as e:
            print(f'\n  Error loading {wav_path.name}: {e}')
            continue

        instrument_id = instrument_map[piano_type]
        data = {
            'audio':         audio,
            'midi_note':     midi_note,
            'velocity':      velocity,
            'instrument_id': instrument_id,
            'piano_type':    piano_type,
            'note_name':     note_name,
            'file_path':     str(wav_path),
        }
        out_path = out_dir / f'sample_{i:06d}.pt'
        torch.save(data, out_path)
        saved.append(out_path)

    return saved


def split_train_val(entries, train_ratio=0.9, seed=42):
    """
    Split entries into train / val while keeping all pitches of each
    (piano_type, velocity) group together in the same split where possible.
    Falls back to random sample-level split if groups are too small.
    """
    rng = np.random.default_rng(seed)

    # Group by (piano_type, velocity)
    groups = defaultdict(list)
    for entry in entries:
        _, piano_type, _, _, velocity = entry
        groups[(piano_type, velocity)].append(entry)

    train, val = [], []
    for group_entries in groups.values():
        arr    = list(group_entries)
        n_val  = max(1, round(len(arr) * (1 - train_ratio)))
        idx    = rng.permutation(len(arr))
        val   += [arr[i] for i in idx[:n_val]]
        train += [arr[i] for i in idx[n_val:]]

    return train, val


def main():
    parser = argparse.ArgumentParser(description='Prepare VST piano WAVs for DAC adapter training')
    parser.add_argument('--source-dir', required=True,
                        help='Directory containing WAV files')
    parser.add_argument('--output-dir', default='data/vst_piano',
                        help='Output directory (default: data/vst_piano)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Fraction of data for training (default: 0.9)')
    parser.add_argument('--no-split', action='store_true',
                        help='Skip train/val split, save everything to output-dir directly')
    args = parser.parse_args()

    print('\n' + '='*70)
    print('VST PIANO DATASET PREPARATION')
    print('='*70)

    # ── Step 1: Scan ──────────────────────────────────────────────────────────
    print(f'\nScanning: {args.source_dir}')
    entries = scan_wavs(args.source_dir)
    print(f'  Found {len(entries)} valid WAV files')

    if not entries:
        print('ERROR: No matching files found.')
        print('Expected pattern: {PianoType}_{NoteName}_{MIDINote:03d}_v{Velocity:03d}.wav')
        print('Example:          American_Small_Studio_A0_021_v040.wav')
        return

    # ── Step 2: Instrument map ────────────────────────────────────────────────
    instrument_map = build_instrument_map(entries)
    print(f'\nPiano types found ({len(instrument_map)}):')
    for name, idx in sorted(instrument_map.items(), key=lambda x: x[1]):
        count = sum(1 for e in entries if e[1] == name)
        print(f'  [{idx:2d}] {name}  ({count} samples)')

    # Velocity summary
    velocities = sorted({e[4] for e in entries})
    print(f'\nVelocities: {velocities}')
    midi_notes = sorted({e[3] for e in entries})
    print(f'MIDI notes: {min(midi_notes)}–{max(midi_notes)}  ({len(midi_notes)} unique)')

    out_base = Path(args.output_dir)

    if args.no_split:
        process_and_save(entries, instrument_map, out_base)
        print(f'\n✓ Saved {len(entries)} samples to {out_base}')
    else:
        # ── Step 3: Split ─────────────────────────────────────────────────────
        train_entries, val_entries = split_train_val(
            entries, train_ratio=args.train_ratio
        )
        print(f'\nSplit: {len(train_entries)} train / {len(val_entries)} val')

        # ── Step 4: Process ───────────────────────────────────────────────────
        train_dir = out_base / 'train'
        val_dir   = out_base / 'val'

        process_and_save(train_entries, instrument_map, train_dir)
        process_and_save(val_entries,   instrument_map, val_dir)

        print('\n' + '='*70)
        print('PREPARATION COMPLETE')
        print('='*70)
        print(f'  Train: {train_dir}  ({len(train_entries)} samples)')
        print(f'  Val:   {val_dir}  ({len(val_entries)} samples)')
        print(f'\nNext steps:')
        print(f'  python scripts/precompute_dac_latents.py \\')
        print(f'      --data-dir {train_dir} --output-dir data/vst_latents/train')
        print(f'  python scripts/precompute_dac_latents.py \\')
        print(f'      --data-dir {val_dir} --output-dir data/vst_latents/val')
        print(f'\nThen update config/dac_adapter_config.yaml:')
        print(f'  data:')
        print(f'    latent_dir: data/vst_latents')
        print(f'\nThen train:')
        print(f'  python scripts/train_dac_adapter.py')


if __name__ == '__main__':
    import numpy as np
    main()
