#!/usr/bin/env python3
"""
One-shot instrument generation using the trained DAC Pitch Adapter.

Encodes a source audio file with DAC, strips its pitch, then sweeps a range
of MIDI notes via PitchInjector, producing one WAV per note.  Optionally
exports an SFZ instrument manifest.

Usage:
    python scripts/generate_instrument_dac.py \
        --source path/to/note.wav \
        --checkpoint checkpoints/dac_adapter/best_model.pt \
        --config config/dac_adapter_config.yaml \
        --midi-low 21 --midi-high 108 \
        --out-dir output/my_instrument \
        [--sfz]
"""

import sys
import argparse
from pathlib import Path
import yaml
import torch
import torchaudio
import soundfile as sf
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.conditioned_dac import load_pretrained_dac
from src.models.dac_pitch_adapter import DACPitchAdapter


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_name(midi_note):
    octave = midi_note // 12 - 1
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def parse_args():
    parser = argparse.ArgumentParser(description='Generate instrument with DAC Pitch Adapter')
    parser.add_argument('--source', type=str, required=True,
                        help='Source audio file (WAV/FLAC)')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/dac_adapter/best_model.pt',
                        help='Trained adapter checkpoint')
    parser.add_argument('--config', type=str,
                        default='config/dac_adapter_config.yaml',
                        help='Config file used during training')
    parser.add_argument('--midi-low', type=int, default=21,
                        help='Lowest MIDI note to generate (default: 21 = A0)')
    parser.add_argument('--midi-high', type=int, default=108,
                        help='Highest MIDI note to generate (default: 108 = C8)')
    parser.add_argument('--out-dir', type=str, default='output/instrument_dac',
                        help='Output directory for WAV files')
    parser.add_argument('--sfz', action='store_true',
                        help='Export SFZ instrument manifest')
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_audio(path, target_sr=44100, device='cpu'):
    """Load audio file, resample to target_sr, convert to mono [1, 1, N]."""
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # DAC expects [batch, 1, samples]
    return audio.unsqueeze(0).to(device)


def load_adapter(config, checkpoint_path, dac_model, device):
    model_cfg = config['model']
    adapter = DACPitchAdapter(
        dac_model=dac_model,
        inner_dim=model_cfg['inner_dim'],
        num_residual_blocks=model_cfg['num_residual_blocks'],
        pitch_embed_dim=model_cfg['pitch_embed_dim'],
        dac_latent_dim=model_cfg['dac_latent_dim'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    adapter.stripper.load_state_dict(ckpt['adapter_state_dict'])
    adapter.injector.load_state_dict(ckpt['injector_state_dict'])
    adapter.eval()
    print(f"Loaded adapter from epoch {ckpt.get('epoch', '?')} "
          f"(best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f})")
    return adapter


def export_sfz(out_dir, midi_low, midi_high, instrument_name='instrument'):
    """Write a minimal SFZ file mapping each WAV to its MIDI note range."""
    sfz_path = out_dir / f'{instrument_name}.sfz'
    lines = ['<control>', 'default_path=./\n', '<group>']
    for note in range(midi_low, midi_high + 1):
        name = midi_to_name(note)
        wav_name = f'note_{note:03d}_{name}.wav'
        lines.append(f'<region> sample={wav_name} lokey={note} hikey={note} pitch_keycenter={note}')
    sfz_path.write_text('\n'.join(lines) + '\n')
    print(f"SFZ manifest: {sfz_path}")


def main():
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DAC PITCH ADAPTER — INSTRUMENT GENERATION")
    print("="*70)
    print(f"Source:     {args.source}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MIDI range: {args.midi_low}–{args.midi_high}")
    print(f"Output:     {out_dir}")
    print(f"Device:     {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load frozen DAC
    print("\nLoading DAC...")
    dac_model = load_pretrained_dac(
        model_type=config['model']['dac_model_type'],
        device=device,
    )

    # Load adapter
    print("Loading adapter...")
    adapter = load_adapter(config, args.checkpoint, dac_model, device)

    # Load source audio
    print(f"\nEncoding source: {args.source}")
    audio = load_audio(args.source, target_sr=config['audio']['sample_rate'], device=device)

    # DAC encode once
    with torch.no_grad():
        z_dac, _codes, _latents, _cl, _ql = dac_model.encode(audio)
        z_timbre, _ = adapter.stripper(z_dac)

    print(f"z_dac shape: {z_dac.shape},  z_timbre shape: {z_timbre.shape}")

    # Generate one WAV per MIDI note
    sr = config['audio']['sample_rate']
    print(f"\nGenerating MIDI {args.midi_low}–{args.midi_high}...")

    for midi_note in range(args.midi_low, args.midi_high + 1):
        with torch.no_grad():
            tgt = torch.tensor([midi_note], dtype=torch.long, device=device)
            z_modified = adapter.injector(z_timbre, tgt)
            audio_out = dac_model.decode(z_modified)   # [1, 1, T]

        # Convert to numpy and save
        wav = audio_out[0, 0].cpu().float().numpy()

        # Normalise
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.95

        note_name = midi_to_name(midi_note)
        out_path = out_dir / f'note_{midi_note:03d}_{note_name}.wav'
        sf.write(str(out_path), wav, sr)

        if midi_note % 12 == 0:
            print(f"  MIDI {midi_note:3d} ({note_name:4s}) → {out_path.name}")

    print(f"\n{args.midi_high - args.midi_low + 1} WAV files written to {out_dir}")

    if args.sfz:
        export_sfz(out_dir, args.midi_low, args.midi_high)

    print("\nDone!")


if __name__ == '__main__':
    main()
