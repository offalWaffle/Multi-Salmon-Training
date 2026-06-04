#!/usr/bin/env python3
"""
Verify that training data MIDI labels match actual audio pitch.
Tests VAE reconstruction to ensure pitch is preserved.
"""

import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.stable_audio_vae import StableAudioVAE
from src.data.audio_utils import save_audio


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print("\n" + "="*70)
    print("VERIFYING TRAINING DATA")
    print("="*70)

    # Load VAE
    print("\nLoading Stable Audio VAE...")
    vae = StableAudioVAE(device=device)
    vae.eval()

    # Find some training samples with known MIDI notes
    train_dir = Path('data/piano_latents/train')
    samples = sorted(train_dir.glob('sample_*.pt'))

    print(f"\nFound {len(samples)} training samples")
    print("\nTesting VAE reconstruction for samples with different MIDI notes:\n")

    output_dir = Path('outputs/verify_training_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test a few samples across the range
    test_samples = []
    midi_targets = [36, 48, 60, 72, 84]  # C2, C3, C4, C5, C6

    # Find samples close to target MIDI notes
    for target_midi in midi_targets:
        best_sample = None
        best_diff = 999

        for sample_path in samples:
            data = torch.load(sample_path, weights_only=False)
            midi = data['midi_note']
            diff = abs(midi - target_midi)

            if diff < best_diff:
                best_diff = diff
                best_sample = (sample_path, midi)

        if best_sample:
            test_samples.append(best_sample)

    for sample_path, actual_midi in test_samples:
        # Load latent
        data = torch.load(sample_path, weights_only=False)
        latent = data['latent'].unsqueeze(0).to(device)  # [1, 64, time]

        print(f"Sample: {sample_path.name}")
        print(f"  MIDI label in training data: {actual_midi}")

        # Decode through VAE
        with torch.no_grad():
            audio = vae.decode(latent)

        # Save
        note_name = {36: "C2", 48: "C3", 60: "C4", 72: "C5", 84: "C6"}.get(actual_midi, f"MIDI{actual_midi}")
        filename = f"reconstructed_{note_name}_labeled_{actual_midi}.wav"
        save_path = output_dir / filename
        save_audio(audio[0], str(save_path), sample_rate=44100)

        print(f"  ✓ Saved: {filename}")
        print()

    print("="*70)
    print("NEXT STEP")
    print("="*70)
    print(f"\nSaved reconstructions to: {output_dir}")
    print("\n1. Use a tuner to check the pitch of each file")
    print("2. Compare the detected pitch to the MIDI label")
    print("\nExpected results:")
    print("  - reconstructed_C4_labeled_60.wav should sound like C4")
    print("  - If it doesn't → Training data MIDI labels are wrong")
    print("  - If it does → Model architecture issue\n")


if __name__ == "__main__":
    main()
