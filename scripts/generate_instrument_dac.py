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
    parser.add_argument('--source-midi', type=int, required=True,
                        help='MIDI note number of the source audio (e.g. 60 for C4)')
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
    parser.add_argument('--requantize', action='store_true',
                        help='Snap injector output back onto DAC RVQ codebook manifold before decoding')
    parser.add_argument('--env-from-source', action='store_true', dest='env_from_source',
                        help='Renormalize injector output per-frame so amplitude envelope is inherited from the source')
    parser.add_argument('--diag', action='store_true',
                        help='Also write diagnostic WAVs: roundtrip.wav (decode source latents only) and identity.wav (injector with tgt=src)')
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
    data, sr = sf.read(str(path), always_2d=True)   # [N, channels]
    audio = torch.from_numpy(data.T).float()         # [channels, N]
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)      # mono [1, N]
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
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
    print(f"Source:     {args.source}  (MIDI {args.source_midi})")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"MIDI range: {args.midi_low}–{args.midi_high}")
    print(f"Output:     {out_dir}")
    print(f"Device:     {device}")
    print(f"Requantize: {args.requantize}")
    print(f"EnvFromSrc: {args.env_from_source}")
    print(f"Diag:       {args.diag}")

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
    src_midi_t = torch.tensor([args.source_midi], dtype=torch.long, device=device)

    with torch.no_grad():
        z_dac, _codes, _latents, _cl, _ql = dac_model.encode(audio)
        z_timbre, _ = adapter.stripper(z_dac)

    print(f"z_dac shape: {z_dac.shape},  z_timbre shape: {z_timbre.shape}")

    # Generate one WAV per MIDI note
    sr = config['audio']['sample_rate']

    def save_wav(z_decode_input, path):
        with torch.no_grad():
            audio_out = dac_model.decode(z_decode_input)
        wav = audio_out[0, 0].cpu().float().numpy()
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.95
        sf.write(str(path), wav, sr)

    if args.diag:
        print("\nWriting diagnostic WAVs...")
        # Decode source latents directly — tests DAC encode/decode roundtrip with no adapter
        save_wav(z_dac, out_dir / 'diag_roundtrip.wav')
        print(f"  diag_roundtrip.wav (no adapter)")
        # Injector with tgt = src — tests whether the adapter adds buzz even when not changing pitch
        with torch.no_grad():
            z_identity = adapter.injector(z_timbre, src_midi_t, src_midi_t)
        save_wav(z_identity, out_dir / 'diag_identity.wav')
        print(f"  diag_identity.wav (injector with tgt=src)")

    print(f"\nGenerating MIDI {args.midi_low}–{args.midi_high}...")

    eps = 1e-8
    for midi_note in range(args.midi_low, args.midi_high + 1):
        with torch.no_grad():
            tgt = torch.tensor([midi_note], dtype=torch.long, device=device)
            z_modified = adapter.injector(z_timbre, src_midi_t, tgt)
            if args.env_from_source:
                # Replace per-frame magnitude with source's; keep injector's direction
                n_src = z_timbre.norm(dim=1, keepdim=True)
                n_mod = z_modified.norm(dim=1, keepdim=True)
                z_modified = z_modified * (n_src / (n_mod + eps))
            if args.requantize:
                z_modified, *_ = dac_model.quantizer(z_modified)
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
