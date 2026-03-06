#!/usr/bin/env python3
"""
Evaluate adversarial pitch disentanglement and pitch transfer quality.

Two modes (run both by default):

  1. Classifier accuracy  (--no-transfer to skip mode 2)
     Freeze encoder, run pitch classifier on all val/test samples.
     Target: accuracy near chance (1/88 ≈ 1.1%).
     A high accuracy (>50%) means z_q still leaks pitch.

  2. Pitch transfer  (--no-classifier to skip mode 1)
     Pick one source sample, encode → z_q, decode at a sweep of target
     MIDI notes, save WAVs.  Listen to confirm timbre is preserved while
     pitch changes.

Usage:
    python scripts/evaluate_disentanglement.py
    python scripts/evaluate_disentanglement.py --source-midi 60 --target-range 36 84
    python scripts/evaluate_disentanglement.py --no-classifier
    python scripts/evaluate_disentanglement.py --split val --max-samples 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vqvae import VQVAE
from src.models.pitch_adversary import PitchClassifier
from src.data.vqvae_dataset import VQVAEDataset, collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_checkpoint(ckpt_path, device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    epoch = ckpt.get('epoch', '?')
    print(f"  Epoch: {epoch}  |  Val loss: {ckpt.get('best_val_loss', float('nan')):.4f}")

    model = VQVAE(config['model']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    classifier = None
    if 'pitch_classifier_state_dict' in ckpt:
        latent_dim = config['model']['latent_dim']
        classifier = PitchClassifier(latent_dim=latent_dim, num_midi_classes=88, midi_offset=21)
        classifier.load_state_dict(ckpt['pitch_classifier_state_dict'])
        classifier = classifier.to(device)
        classifier.eval()
        print("  Pitch classifier found in checkpoint.")
    else:
        print("  WARNING: no pitch classifier in checkpoint — was adversarial.enabled=true?")

    return model, classifier, config


def make_dataset(config, split, max_samples):
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

    if max_samples is not None and max_samples < len(dataset):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), max_samples, replace=False).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset, sr


# ---------------------------------------------------------------------------
# Mode 1: classifier accuracy
# ---------------------------------------------------------------------------

def evaluate_classifier(model, classifier, config, split, max_samples, output_dir, device):
    """
    Freeze encoder, run all samples through encoder+quantizer, feed z_q to
    the pitch classifier, report top-1 accuracy.

    Target: ~1.1% (chance for 88 classes) = full disentanglement.
    """
    if classifier is None:
        print("\n[Mode 1] Skipped — no classifier in checkpoint.")
        return None

    dataset, sr = make_dataset(config, split, max_samples)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )

    midi_offset = 21
    num_classes = 88

    correct = 0
    total = 0
    per_class_correct = np.zeros(num_classes, dtype=int)
    per_class_total = np.zeros(num_classes, dtype=int)

    print(f"\n[Mode 1] Classifier accuracy on '{split}' ({len(dataset)} samples)...")
    with torch.no_grad():
        for batch in tqdm(loader):
            audio = batch['audio'].to(device)
            midi_note = batch['midi_note'].to(device)  # raw MIDI

            # Encode → z_q (frozen, no grad needed)
            z = model.encoder(audio)
            z_q, _, _, _ = model.quantizer(z)

            # Classifier prediction
            pooled = z_q.mean(dim=-1)          # [B, latent_dim]
            logits = classifier.net(pooled)    # [B, 88]
            predicted = logits.argmax(dim=-1)  # [B]  (0-indexed)

            class_idx = (midi_note - midi_offset).long().clamp(0, num_classes - 1)
            hits = (predicted == class_idx)

            correct += hits.sum().item()
            total += len(hits)

            for i in range(len(hits)):
                c = class_idx[i].item()
                per_class_total[c] += 1
                per_class_correct[c] += int(hits[i].item())

    accuracy = correct / total if total > 0 else 0.0
    chance = 1.0 / num_classes

    SEP = '=' * 56
    print(f"\n{SEP}")
    print("  Mode 1: Pitch Classifier Accuracy")
    print(SEP)
    print(f"  Samples evaluated  : {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Top-1 accuracy     : {accuracy*100:.2f}%")
    print(f"  Chance level       : {chance*100:.2f}%  (1/88)")
    disentangled = accuracy <= chance * 3   # within 3× chance
    status = "DISENTANGLED" if disentangled else "PITCH STILL LEAKING"
    print(f"  Status             : {status}")
    print(SEP)

    if not disentangled:
        print("  Tip: increase adversarial.lambda in vqvae_config.yaml and retrain.")

    # Top-5 most confused notes
    confused = [(per_class_correct[c] / per_class_total[c] * 100, c)
                for c in range(num_classes) if per_class_total[c] > 0]
    confused.sort(reverse=True)
    if confused:
        print(f"\n  Most predicted MIDI notes (highest per-class accuracy):")
        for acc_pct, c in confused[:5]:
            midi = c + midi_offset
            print(f"    MIDI {midi:3d}  acc={acc_pct:.1f}%  n={per_class_total[c]}")

    result = {
        'accuracy': round(accuracy, 6),
        'chance': round(chance, 6),
        'correct': correct,
        'total': total,
        'disentangled': disentangled,
    }

    out_path = output_dir / 'classifier_accuracy.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Report saved: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Mode 2: pitch transfer sweep
# ---------------------------------------------------------------------------

def evaluate_pitch_transfer(model, config, source_midi, target_range, output_dir, device):
    """
    Find one sample at source_midi in the val set, encode to z_q, then decode
    at each MIDI note in target_range.  Save WAVs and a manifest JSON.
    """
    dataset, sr = make_dataset(config, 'val', max_samples=None)
    audio_cfg = config['audio']

    # Find a sample at the requested source pitch
    source_sample = None
    raw_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset

    for i in range(len(raw_dataset)):
        item = raw_dataset[i]
        if int(item['midi_note']) == source_midi:
            source_sample = item
            break

    if source_sample is None:
        # Fall back to closest available pitch
        available = sorted({int(raw_dataset[i]['midi_note'])
                            for i in range(min(len(raw_dataset), 500))})
        closest = min(available, key=lambda m: abs(m - source_midi))
        print(f"\n[Mode 2] No sample at MIDI {source_midi}, using closest: MIDI {closest}")
        source_midi = closest
        for i in range(len(raw_dataset)):
            item = raw_dataset[i]
            if int(item['midi_note']) == source_midi:
                source_sample = item
                break

    audio_tensor = source_sample['audio'].unsqueeze(0).to(device)   # [1, 1, T]
    source_midi_t = torch.tensor([source_sample['midi_note']], dtype=torch.long, device=device)

    print(f"\n[Mode 2] Pitch transfer from MIDI {source_midi} "
          f"to range {target_range[0]}–{target_range[1]}...")

    transfer_dir = output_dir / 'pitch_transfer'
    transfer_dir.mkdir(parents=True, exist_ok=True)

    # Save source reference
    src_np = audio_tensor[0, 0].cpu().numpy()
    src_path = transfer_dir / f"source_midi{source_midi:03d}.wav"
    sf.write(str(src_path), src_np, sr, subtype='PCM_16')

    # Encode source → z_q once
    with torch.no_grad():
        z = model.encoder(audio_tensor)
        z_q, _, _, _ = model.quantizer(z)

    # Decode at each target pitch
    target_notes = list(range(target_range[0], target_range[1] + 1, 3))
    manifest = []

    with torch.no_grad():
        for target_midi in tqdm(target_notes, desc="Decoding pitches"):
            target_t = torch.tensor([target_midi], dtype=torch.long, device=device)
            recon = model.decoder(z_q, target_t)

            # Trim/pad to source length
            T = audio_tensor.shape[-1]
            if recon.shape[-1] > T:
                recon = recon[..., :T]
            elif recon.shape[-1] < T:
                import torch.nn.functional as F
                recon = F.pad(recon, (0, T - recon.shape[-1]))

            out_np = recon[0, 0].cpu().numpy()
            out_path = transfer_dir / f"transfer_midi{target_midi:03d}.wav"
            sf.write(str(out_path), out_np, sr, subtype='PCM_16')

            manifest.append({
                'target_midi': target_midi,
                'target_hz': round(440.0 * 2 ** ((target_midi - 69) / 12), 2),
                'file': out_path.name,
                'is_source': target_midi == source_midi,
            })

    manifest_path = transfer_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'source_midi': source_midi,
            'source_file': src_path.name,
            'targets': manifest,
        }, f, indent=2)

    print(f"\n  {len(target_notes)} WAVs saved to: {transfer_dir}/")
    print(f"  Manifest: {manifest_path}")
    print("\n  Listen for: same timbre character across all files, pitch changes correctly.")
    print("  Red flags : pitch doesn't shift / timbre changes dramatically at extremes.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pitch disentanglement.")
    p.add_argument('--checkpoint', '-c', default='checkpoints/vqvae/best_model.pt')
    p.add_argument('--split', '-s', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--max-samples', '-n', type=int, default=None,
                   help='Max samples for classifier eval (default: all)')
    p.add_argument('--source-midi', type=int, default=60,
                   help='Source MIDI note for pitch transfer (default: 60 = C4)')
    p.add_argument('--target-range', type=int, nargs=2, default=[36, 84],
                   metavar=('LOW', 'HIGH'),
                   help='MIDI range for transfer sweep, step=3 (default: 36 84)')
    p.add_argument('--output-dir', '-o', default='outputs/disentanglement_eval')
    p.add_argument('--no-classifier', action='store_true',
                   help='Skip Mode 1 (classifier accuracy)')
    p.add_argument('--no-transfer', action='store_true',
                   help='Skip Mode 2 (pitch transfer)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    root = Path(__file__).parent.parent

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    model, classifier, config = load_checkpoint(ckpt_path, device)

    if not args.no_classifier:
        evaluate_classifier(
            model, classifier, config,
            split=args.split,
            max_samples=args.max_samples,
            output_dir=out_dir,
            device=device,
        )

    if not args.no_transfer:
        evaluate_pitch_transfer(
            model, config,
            source_midi=args.source_midi,
            target_range=args.target_range,
            output_dir=out_dir,
            device=device,
        )
