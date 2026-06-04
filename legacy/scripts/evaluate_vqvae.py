#!/usr/bin/env python3
"""
Evaluate a trained VQ-VAE model against Phase 1 success criteria.

Reports:
  1. Reconstruction SDR  (target: > 10 dB)
  2. Codebook perplexity (target: > 100)
  3. Pitch accuracy      (target: > 70%)

Usage:
    python scripts/evaluate_vqvae.py
    python scripts/evaluate_vqvae.py --checkpoint checkpoints/vqvae/best_model.pt --split test
    python scripts/evaluate_vqvae.py --split val --max-samples 100
    python scripts/evaluate_vqvae.py --save-audio
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vqvae import VQVAE
from src.data.vqvae_dataset import VQVAEDataset, collate_fn


# ---------------------------------------------------------------------------
# Audio / pitch helpers
# ---------------------------------------------------------------------------

def midi_to_hz(midi_note):
    """MIDI note number → frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def compute_sdr(reference, estimate):
    """
    Signal-to-Distortion Ratio in dB (higher = better).

        SDR = 10 * log10( ||ref||^2 / ||ref - est||^2 )
    """
    noise = reference - estimate
    ref_power = np.dot(reference, reference)
    noise_power = np.dot(noise, noise)
    if noise_power < 1e-12:
        return float('inf')
    if ref_power < 1e-12:
        return -float('inf')
    return 10.0 * np.log10(ref_power / noise_power)


def detect_pitch_hz(audio, sr):
    """
    Estimate fundamental frequency with librosa pyin.

    Returns median voiced f0 (Hz) or None if no voiced frames found.
    """
    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=librosa.note_to_hz('C1'),   # ~32.7 Hz
            fmax=librosa.note_to_hz('C8'),   # ~4186 Hz
            sr=sr,
        )
        voiced = f0[voiced_flag & np.isfinite(f0)]
        if len(voiced) == 0:
            return None
        return float(np.median(voiced))
    except Exception:
        return None


def pitch_cents_error(detected_hz, target_midi):
    """Cents deviation between detected Hz and target MIDI note (None if undetected)."""
    if detected_hz is None or detected_hz <= 0:
        return None
    target_hz = midi_to_hz(target_midi)
    return 1200.0 * np.log2(detected_hz / target_hz)


def is_pitch_correct(detected_hz, target_midi, tolerance_semitones=1.0):
    """True if detected pitch is within tolerance of target MIDI note."""
    cents = pitch_cents_error(detected_hz, target_midi)
    if cents is None:
        return False
    return abs(cents) <= tolerance_semitones * 100.0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path, split, save_audio, output_dir, max_samples, min_velocity=None):

    # ---- Device ----
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    ckpt_epoch = checkpoint.get('epoch', '?')
    best_val_loss = checkpoint.get('best_val_loss', float('nan'))
    print(f"  Trained epoch: {ckpt_epoch}  |  Best val loss: {best_val_loss:.4f}")

    # ---- Build model ----
    model = VQVAE(config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Dataset ----
    audio_cfg = config['audio']
    sr = audio_cfg['sample_rate']
    duration = audio_cfg['duration']

    split_key = f"{split}_path"
    data_path = Path(config['data'].get(split_key, f"data/vqvae_piano/{split}.json"))
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    if not data_path.exists():
        print(f"Error: data file not found: {data_path}")
        sys.exit(1)

    dataset = VQVAEDataset(
        metadata_path=str(data_path),
        sample_rate=sr,
        duration=duration,
        augment=False,
    )

    if min_velocity is not None:
        indices = [i for i, s in enumerate(dataset.samples)
                   if s.get('velocity', 0) >= min_velocity]
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Filtered to {len(dataset)} samples with velocity >= {min_velocity}")

    if max_samples is not None and max_samples < len(dataset):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), max_samples, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Evaluating on {max_samples} randomly chosen samples (seed=42)")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,      # MPS deadlock prevention
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- Output dir ----
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_audio:
        audio_out = output_dir / 'audio_samples'
        audio_out.mkdir(parents=True, exist_ok=True)

    # ---- Accumulate per-sample results ----
    sdrs = []
    perplexities = []
    pitch_hits = []
    per_sample = []
    note_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    print(f"\nEvaluating {len(dataset)} samples from '{split}' split...")
    print("(Pitch detection with pyin is slow — ~0.5 s/sample on CPU)")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            audio = batch['audio'].to(device)       # (1, 1, T)
            midi_note_t = batch['midi_note'].to(device)  # (1,)
            midi_note = int(midi_note_t[0].item())

            # Forward pass
            recon, vq_loss, perplexity, encoding_indices, z, _z_q = \
                model(audio, midi_note_t)

            # SDR
            ref_np = audio[0, 0].cpu().numpy().astype(np.float64)
            rec_np = recon[0, 0].cpu().numpy().astype(np.float64)
            sdr = compute_sdr(ref_np, rec_np)
            if np.isfinite(sdr):
                sdrs.append(sdr)

            # Perplexity
            perp = perplexity.item()
            perplexities.append(perp)

            # Pitch accuracy — run pyin on the *reconstruction*
            detected_hz = detect_pitch_hz(rec_np, sr)
            correct = is_pitch_correct(detected_hz, midi_note)
            pitch_hits.append(int(correct))
            note_stats[midi_note]['total'] += 1
            note_stats[midi_note]['correct'] += int(correct)

            cents = pitch_cents_error(detected_hz, midi_note)

            per_sample.append({
                'sample_idx': i,
                'midi_note': midi_note,
                'target_hz': round(float(midi_to_hz(midi_note)), 2),
                'detected_hz': round(float(detected_hz), 2) if detected_hz else None,
                'cents_error': round(float(cents), 1) if cents is not None else None,
                'pitch_correct': bool(correct),
                'sdr_db': round(float(sdr), 2) if np.isfinite(sdr) else None,
                'perplexity': round(float(perp), 1),
            })

            # Optional audio export (first 20 samples only)
            if save_audio and i < 20:
                sf.write(str(audio_out / f"{i:03d}_ref.wav"),
                         ref_np, sr, subtype='PCM_16')
                sf.write(str(audio_out / f"{i:03d}_rec.wav"),
                         rec_np, sr, subtype='PCM_16')

    # ---- Aggregate metrics ----
    mean_sdr       = float(np.mean(sdrs))     if sdrs         else float('nan')
    median_sdr     = float(np.median(sdrs))   if sdrs         else float('nan')
    pct10_sdr      = float(np.percentile(sdrs, 10)) if sdrs   else float('nan')
    mean_perp      = float(np.mean(perplexities))
    pitch_accuracy = float(np.mean(pitch_hits)) if pitch_hits else float('nan')

    # ---- Success criteria ----
    criteria = config.get('success_criteria', {})
    min_sdr  = criteria.get('min_reconstruction_sdr', 10.0)
    min_pitch = criteria.get('min_pitch_accuracy', 0.70)
    min_perp = criteria.get('min_codebook_perplexity', 100.0)

    sdr_pass   = mean_sdr >= min_sdr
    pitch_pass = pitch_accuracy >= min_pitch
    perp_pass  = mean_perp >= min_perp
    all_pass   = sdr_pass and pitch_pass and perp_pass

    # ---- Print report ----
    SEP = '=' * 62
    print(f"\n{SEP}")
    print("  VQ-VAE Evaluation Report")
    print(SEP)
    print(f"  Checkpoint : {checkpoint_path.name}  (epoch {ckpt_epoch})")
    print(f"  Split      : {split}  ({len(dataset)} samples)")
    print(SEP)
    print(f"  {'Metric':<30} {'Value':>10}  {'Target':>8}  Status")
    print(f"  {'-'*58}")

    def _row(label, val, target, passed, fmt=".2f"):
        status = "PASS" if passed else "FAIL"
        val_s = format(val, fmt)
        tgt_s = format(target, fmt)
        print(f"  {label:<30} {val_s:>10}  {tgt_s:>8}  {status}")

    _row("SDR mean (dB)",            mean_sdr,            min_sdr,       sdr_pass)
    _row("SDR median (dB)",          median_sdr,          min_sdr,       median_sdr >= min_sdr)
    _row("SDR 10th-pct (dB)",        pct10_sdr,           min_sdr,       pct10_sdr >= min_sdr)
    _row("Codebook perplexity",       mean_perp,           min_perp,      perp_pass, fmt=".1f")
    _row("Pitch accuracy (%)",        pitch_accuracy*100,  min_pitch*100, pitch_pass, fmt=".1f")

    print(f"  {'-'*58}")
    if all_pass:
        print("  ALL PHASE 1 SUCCESS CRITERIA MET")
    else:
        failing = []
        if not sdr_pass:
            failing.append(f"SDR {mean_sdr:.1f} dB (need {min_sdr})")
        if not perp_pass:
            failing.append(f"Perplexity {mean_perp:.0f} (need {min_perp:.0f})")
        if not pitch_pass:
            failing.append(f"Pitch {pitch_accuracy*100:.1f}% (need {min_pitch*100:.0f}%)")
        print(f"  FAILING: {' | '.join(failing)}")
    print(SEP)

    # Per-note breakdown (only print notes with >= 2 samples)
    multi_note = {k: v for k, v in note_stats.items() if v['total'] >= 2}
    if multi_note:
        print(f"\n  Pitch accuracy by MIDI note (notes with ≥2 samples):")
        print(f"  {'MIDI':>6}  {'Hz':>8}  {'Correct':>8}  {'Total':>6}  {'Acc':>6}")
        for midi in sorted(multi_note):
            s = multi_note[midi]
            acc = s['correct'] / s['total'] * 100
            print(f"  {midi:>6}  {midi_to_hz(midi):>8.1f}  "
                  f"{s['correct']:>8}  {s['total']:>6}  {acc:>5.0f}%")

    # ---- Save JSON report ----
    report = {
        'checkpoint': str(checkpoint_path),
        'epoch': ckpt_epoch,
        'split': split,
        'num_samples': len(dataset),
        'metrics': {
            'sdr_mean_db':    round(mean_sdr, 3),
            'sdr_median_db':  round(median_sdr, 3),
            'sdr_p10_db':     round(pct10_sdr, 3),
            'perplexity_mean': round(mean_perp, 1),
            'pitch_accuracy': round(pitch_accuracy, 4),
        },
        'success_criteria': {
            'sdr_pass':           sdr_pass,
            'perplexity_pass':    perp_pass,
            'pitch_accuracy_pass': pitch_pass,
            'all_pass':           all_pass,
        },
        'per_sample': per_sample,
    }

    report_path = output_dir / f"eval_{split}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VQ-VAE against Phase 1 success criteria."
    )
    p.add_argument(
        '--checkpoint', '-c',
        default='checkpoints/vqvae/best_model.pt',
        help='Path to .pt checkpoint (default: checkpoints/vqvae/best_model.pt)',
    )
    p.add_argument(
        '--split', '-s',
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate (default: test)',
    )
    p.add_argument(
        '--save-audio', action='store_true',
        help='Save reference and reconstructed .wav files (first 20 samples)',
    )
    p.add_argument(
        '--output-dir', '-o',
        default='outputs/vqvae_eval',
        help='Directory for report + audio (default: outputs/vqvae_eval)',
    )
    p.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='Limit evaluation to N random samples (default: all)',
    )
    p.add_argument(
        '--min-velocity', '-v',
        type=int,
        default=None,
        help='Only evaluate samples at or above this velocity (e.g. 100 for ff)',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    root = Path(__file__).parent.parent

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}")
        sys.exit(1)

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out

    evaluate(
        checkpoint_path=ckpt,
        split=args.split,
        save_audio=args.save_audio,
        output_dir=out,
        max_samples=args.max_samples,
        min_velocity=args.min_velocity,
    )
