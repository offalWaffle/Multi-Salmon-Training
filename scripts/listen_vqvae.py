#!/usr/bin/env python3
"""
Interactive listening test for the VQ-VAE.

Encodes piano sounds and decodes them, saving originals and reconstructions
side-by-side as WAV files so you can evaluate quality by ear.

Two modes
---------
Dataset mode (default): picks N samples from data/vqvae_piano/test.json
  python scripts/listen_vqvae.py
  python scripts/listen_vqvae.py --num-samples 10 --split val

File mode: process a specific audio file
  python scripts/listen_vqvae.py --input /path/to/note.wav --midi 60

Pitch reference (both modes): save resample-based pitch shifts alongside the reconstruction.
NOTE: Phase-1 VQ-VAE cannot truly transpose — the LatentTransformer (Phase 2) is needed for
that. --transpose saves naive resampled versions as a reference for what the target pitch
should sound like (they will be faster/slower than the original).
  python scripts/listen_vqvae.py --transpose 48 60 72
  python scripts/listen_vqvae.py --input note.wav --midi 60 --transpose 48 60 72

Output layout
-------------
  outputs/listen/<timestamp>/
      sample_001_C4_midi60/
          original.wav
          reconstruction.wav
          transposed_C3_midi48.wav
          transposed_C5_midi72.wav
      sample_002_G3_midi55/
          ...
      summary.txt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vqvae import VQVAE
from src.data.vqvae_dataset import VQVAEDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_name(midi):
    """e.g. 60 → 'C4'"""
    octave = (midi // 12) - 1
    name = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def compute_sdr(ref, est):
    noise = ref - est
    ref_power = np.dot(ref, ref)
    noise_power = np.dot(noise, noise)
    if noise_power < 1e-12:
        return float('inf')
    if ref_power < 1e-12:
        return -float('inf')
    return 10.0 * np.log10(ref_power / noise_power)


def load_audio(path, target_sr, target_length):
    """Load an audio file, resample and trim/pad to target_length samples."""
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n = int(len(audio) * target_sr / sr)
        audio = scipy_signal.resample(audio, n)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def save_wav(path, audio, sr):
    sf.write(str(path), audio, sr, subtype='PCM_16')


def try_save_spectrogram(output_path, ref, rec, sr, label_ref, label_rec):
    """Save a side-by-side spectrogram image. Silently skips if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, audio, title in zip(axes, [ref, rec], [label_ref, label_rec]):
            f, t, Sxx = scipy_signal.spectrogram(audio, fs=sr, nperseg=1024, noverlap=768)
            Sxx_db = 10 * np.log10(Sxx + 1e-9)
            ax.pcolormesh(t, f, Sxx_db, norm=Normalize(vmin=-80, vmax=0), cmap='magma')
            ax.set_ylim(0, 8000)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=100)
        plt.close(fig)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core encode/decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct(model, audio_np, midi_note, device, target_length):
    """Encode audio_np then decode at midi_note. Returns (1, T) numpy array."""
    x = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0).to(device)
    pitch = torch.tensor([midi_note], dtype=torch.long, device=device)

    recon, vq_loss, perplexity, encoding_indices, z = model(x, pitch)

    recon_np = recon[0, 0].cpu().numpy()
    # Trim/pad to match input length
    if len(recon_np) < target_length:
        recon_np = np.pad(recon_np, (0, target_length - len(recon_np)))
    else:
        recon_np = recon_np[:target_length]

    n_codes = encoding_indices.shape[-1]
    unique_codes = encoding_indices.unique().numel()
    return recon_np, float(perplexity.item()), n_codes, unique_codes


def pitch_shift_resample(audio_np, source_midi, target_midi, target_length):
    """
    Naive pitch shift by resampling (changes speed too, but useful as a reference).

    NOTE: The VQ-VAE Phase-1 model cannot do true pitch transposition.
    The codebook codes encode timbre AND pitch together because the model was only
    ever trained to reconstruct at the same pitch it encoded from.  Swapping the
    FiLM conditioning vector in the decoder has no effect because the decoder
    learned to read pitch from z_q directly.

    True transposition requires Phase 2 (LatentTransformer), which will learn to
    map z_q from one pitch to another.  This function provides a resample-based
    reference so you can at least hear what the target pitch should sound like.
    """
    semitones = target_midi - source_midi
    ratio = 2.0 ** (semitones / 12.0)          # speed-up/slow-down factor
    n_resampled = max(1, int(len(audio_np) / ratio))
    shifted = scipy_signal.resample(audio_np, n_resampled)

    # Trim or pad back to target_length
    if len(shifted) < target_length:
        shifted = np.pad(shifted, (0, target_length - len(shifted)))
    else:
        shifted = shifted[:target_length]

    peak = np.abs(shifted).max()
    if peak > 0:
        shifted = shifted / peak * 0.95
    return shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# Process one sample
# ---------------------------------------------------------------------------

def process_sample(model, audio_np, midi_note, sr, target_length, output_dir,
                   transpose_notes, save_spectrogram, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{midi_to_name(midi_note)}_midi{midi_note}"

    # Original
    save_wav(output_dir / 'original.wav', audio_np, sr)

    # Reconstruction
    recon_np, perplexity, n_codes, unique_codes = reconstruct(
        model, audio_np, midi_note, device, target_length)
    save_wav(output_dir / 'reconstruction.wav', recon_np, sr)

    sdr = compute_sdr(audio_np.astype(np.float64), recon_np.astype(np.float64))

    if save_spectrogram:
        try_save_spectrogram(
            output_dir / 'spectrogram.png',
            audio_np, recon_np, sr,
            f'Original ({label})', f'Reconstruction (SDR {sdr:.1f} dB)',
        )

    # Transpositions (resample-based reference — see note in pitch_shift_resample)
    for tgt_midi in transpose_notes:
        if tgt_midi == midi_note:
            continue
        tgt_label = f"{midi_to_name(tgt_midi)}_midi{tgt_midi}"
        shifted = pitch_shift_resample(audio_np, midi_note, tgt_midi, target_length)
        save_wav(output_dir / f'ref_pitchshift_{tgt_label}.wav', shifted, sr)

    return {
        'label': label,
        'sdr_db': round(float(sdr), 2) if np.isfinite(sdr) else None,
        'perplexity': round(perplexity, 1),
        'n_codes': n_codes,
        'unique_codes': unique_codes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    root = Path(__file__).parent.parent

    # ---- Device ----
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ---- Checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    epoch = ckpt.get('epoch', '?')
    print(f"  Epoch: {epoch}  |  Best val loss: {ckpt.get('best_val_loss', float('nan')):.4f}")

    audio_cfg = config['audio']
    sr = audio_cfg['sample_rate']
    duration = audio_cfg['duration']
    target_length = int(sr * duration)

    model = VQVAE(config['model']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Output dir ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root = out_root / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}\n")

    transpose_notes = args.transpose or []

    # ---- Collect samples ----
    samples = []  # list of (audio_np, midi_note, sample_label)

    if args.input:
        # File mode: user provides audio file(s)
        for path_str in args.input:
            path = Path(path_str)
            if not path.exists():
                print(f"Warning: file not found, skipping: {path}")
                continue
            midi = args.midi if args.midi else _detect_pitch_midi(path, sr)
            if midi is None:
                print(f"Warning: could not detect pitch for {path.name}. "
                      f"Provide --midi <note> explicitly.")
                continue
            audio = load_audio(path, sr, target_length)
            samples.append((audio, midi, path.stem))
    else:
        # Dataset mode: load from test/val/train split
        split_key = f"{args.split}_path"
        data_path = Path(config['data'].get(split_key, f"data/vqvae_piano/{args.split}.json"))
        if not data_path.is_absolute():
            data_path = root / data_path
        if not data_path.exists():
            print(f"Error: data file not found: {data_path}")
            sys.exit(1)

        with open(data_path) as f:
            metadata = json.load(f)

        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(metadata), min(args.num_samples, len(metadata)),
                            replace=False)
        for idx in chosen:
            entry = metadata[idx]
            audio = load_audio(entry['path'], sr, target_length)
            midi = entry['midi_note']
            samples.append((audio, midi, Path(entry['path']).stem))

    if not samples:
        print("No samples to process.")
        sys.exit(1)

    print(f"Processing {len(samples)} sample(s)...")
    if transpose_notes:
        print(f"Transpose targets: {[midi_to_name(m) for m in transpose_notes]}")
    print()

    # ---- Run ----
    results = []
    for i, (audio_np, midi_note, stem) in enumerate(samples, 1):
        note_label = midi_to_name(midi_note)
        sample_dir_name = f"sample_{i:03d}_{note_label}_midi{midi_note}"
        sample_dir = out_root / sample_dir_name

        print(f"  [{i}/{len(samples)}] {note_label} (MIDI {midi_note}) — {stem}")

        info = process_sample(
            model=model,
            audio_np=audio_np,
            midi_note=midi_note,
            sr=sr,
            target_length=target_length,
            output_dir=sample_dir,
            transpose_notes=transpose_notes,
            save_spectrogram=args.spectrogram,
            device=device,
        )
        results.append({**info, 'dir': sample_dir_name})

        sdr_str = f"{info['sdr_db']:.1f} dB" if info['sdr_db'] is not None else "N/A"
        print(f"    SDR: {sdr_str}  |  Perplexity: {info['perplexity']}  "
              f"|  Codes used: {info['unique_codes']}/{info['n_codes']}")

    # ---- Summary ----
    sdrs = [r['sdr_db'] for r in results if r['sdr_db'] is not None]
    print()
    print(f"{'='*50}")
    print(f"  Summary ({len(samples)} samples)")
    print(f"{'='*50}")
    if sdrs:
        print(f"  SDR  mean:    {np.mean(sdrs):>6.2f} dB")
        print(f"  SDR  median:  {np.median(sdrs):>6.2f} dB")
        print(f"  SDR  min:     {np.min(sdrs):>6.2f} dB")
        print(f"  SDR  max:     {np.max(sdrs):>6.2f} dB")
    print(f"  Output:  {out_root}")
    print(f"{'='*50}")

    # Save summary text
    lines = [
        f"VQ-VAE Listening Test — {timestamp}",
        f"Checkpoint: {ckpt_path}  (epoch {epoch})",
        f"Split: {args.split}  |  Samples: {len(samples)}",
        f"Duration: {duration}s  |  SR: {sr}",
        "",
    ]
    if transpose_notes:
        lines.append(f"Transpositions: {[midi_to_name(m) for m in transpose_notes]}")
        lines.append("")
    lines.append(f"{'Sample':<40} {'SDR (dB)':>9}  {'Perp':>6}  {'Codes':>8}")
    lines.append("-" * 68)
    for r in results:
        sdr_str = f"{r['sdr_db']:.2f}" if r['sdr_db'] is not None else "    N/A"
        lines.append(f"{r['dir']:<40} {sdr_str:>9}  {r['perplexity']:>6}  "
                     f"{r['unique_codes']}/{r['n_codes']:>6}")
    if sdrs:
        lines += [
            "-" * 68,
            f"{'Mean SDR':<40} {np.mean(sdrs):>9.2f}",
            f"{'Median SDR':<40} {np.median(sdrs):>9.2f}",
        ]

    summary_path = out_root / 'summary.txt'
    summary_path.write_text('\n'.join(lines))
    print(f"  Summary: {summary_path}")


def _detect_pitch_midi(path, sr):
    """Try to detect MIDI note from audio using librosa pyin. Returns int or None."""
    try:
        import librosa
        audio, file_sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Use first 3 seconds for detection
        audio = audio[:file_sr * 3].astype(np.float32)
        f0, voiced, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C1'),
            fmax=librosa.note_to_hz('C8'), sr=file_sr)
        voiced_f0 = f0[voiced & np.isfinite(f0)]
        if len(voiced_f0) == 0:
            return None
        median_hz = float(np.median(voiced_f0))
        midi = round(12 * np.log2(median_hz / 440.0) + 69)
        print(f"    Auto-detected pitch: {midi_to_name(midi)} (MIDI {midi}, {median_hz:.1f} Hz)")
        return int(midi)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Encode/decode piano sounds with VQ-VAE for listening tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--checkpoint', '-c', default='checkpoints/vqvae/best_model.pt',
                   help='Path to checkpoint (default: checkpoints/vqvae/best_model.pt)')

    # Dataset mode
    ds = p.add_argument_group('Dataset mode (default)')
    ds.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                    help='Dataset split to sample from (default: test)')
    ds.add_argument('--num-samples', '-n', type=int, default=8,
                    help='Number of random samples (default: 8)')
    ds.add_argument('--seed', type=int, default=42,
                    help='Random seed for sample selection (default: 42)')

    # File mode
    fm = p.add_argument_group('File mode')
    fm.add_argument('--input', nargs='+', metavar='WAV',
                    help='One or more audio files to process')
    fm.add_argument('--midi', type=int, metavar='NOTE',
                    help='MIDI note number for input file(s) (auto-detected if omitted)')

    # Shared
    p.add_argument('--transpose', nargs='+', type=int, metavar='MIDI',
                   help='Also decode at these MIDI notes (tests pitch conditioning). '
                        'Example: --transpose 48 60 72')
    p.add_argument('--spectrogram', action='store_true',
                   help='Save spectrogram comparison image per sample (requires matplotlib)')
    p.add_argument('--output-dir', '-o', default='outputs/listen',
                   help='Base output directory (default: outputs/listen)')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
