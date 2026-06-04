"""
Paired Dataset for LatentTransformer training.

Yields (source, target) pairs from the same instrument at different pitches,
enabling the transformer to learn pitch-invariant timbre transfer.
"""

import json
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy import signal
from torch.utils.data import Dataset
from collections import defaultdict


class TransformerDataset(Dataset):
    """
    Dataset that yields (source, target) audio pairs from the same instrument.

    Source and target share the same timbre (same instrument) but have
    different MIDI pitches, training the transformer to separate timbre
    from pitch in latent space.
    """

    def __init__(self, metadata_path, sample_rate=44100, duration=2.0, augment=False):
        """
        Args:
            metadata_path: Path to JSON file (same format as VQVAEDataset).
            sample_rate: Target sample rate.
            duration: Target duration in seconds.
            augment: Whether to apply data augmentation.
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.augment = augment

        with open(metadata_path) as f:
            all_samples = json.load(f)

        # Group samples by instrument
        by_instrument = defaultdict(list)
        for s in all_samples:
            by_instrument[s['instrument']].append(s)

        # Keep only instruments that have more than one sample (need pairs)
        self.by_instrument = {k: v for k, v in by_instrument.items() if len(v) > 1}

        # Flat list for indexing; build pool lookup per sample
        self.samples = []
        for samples in self.by_instrument.values():
            self.samples.extend(samples)

        print(
            f"TransformerDataset: {len(self.samples)} samples "
            f"from {len(self.by_instrument)} instruments ({metadata_path})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src = self.samples[idx]
        pool = self.by_instrument[src['instrument']]

        # Pick a target with a different MIDI pitch from the same instrument.
        # Try up to 10 times to avoid degenerate same-pitch pairs.
        tgt = src
        for _ in range(10):
            tgt = pool[np.random.randint(len(pool))]
            if tgt['midi_note'] != src['midi_note']:
                break

        src_audio = self._load_audio(src['path'])
        tgt_audio = self._load_audio(tgt['path'])

        if self.augment:
            src_audio = self._augment(src_audio)
            tgt_audio = self._augment(tgt_audio)

        return {
            'source_audio':    torch.from_numpy(src_audio).float().unsqueeze(0),
            'source_midi':     src['midi_note'],
            'source_velocity': src.get('velocity', 80),
            'target_audio':    torch.from_numpy(tgt_audio).float().unsqueeze(0),
            'target_midi':     tgt['midi_note'],
            'target_velocity': tgt.get('velocity', 80),
        }

    # ------------------------------------------------------------------
    # Internal helpers (reuse the same logic as VQVAEDataset)
    # ------------------------------------------------------------------

    def _load_audio(self, path):
        """Load, resample, and normalise a single audio file."""
        try:
            with sf.SoundFile(path) as f:
                file_sr = f.samplerate
                total_frames = f.frames

                frames_needed = int(self.target_length * file_sr / self.sample_rate) + 1

                if total_frames <= frames_needed:
                    f.seek(0)
                    audio = f.read(always_2d=False)
                else:
                    if self.augment:
                        start = np.random.randint(0, total_frames - frames_needed + 1)
                    else:
                        start = (total_frames - frames_needed) // 2
                    f.seek(start)
                    audio = f.read(frames=frames_needed, always_2d=False)

                sr = file_sr

            if sr != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.target_length]

            if not np.isfinite(audio).all():
                return np.zeros(self.target_length, dtype=np.float32)

            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.95

            return audio.astype(np.float32)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(self.target_length, dtype=np.float32)

    def _augment(self, audio):
        if np.random.rand() < 0.5:
            gain_db = np.random.uniform(-3, 3)
            audio = audio * (10 ** (gain_db / 20))
        if np.random.rand() < 0.3:
            audio = -audio
        if np.random.rand() < 0.3:
            audio = audio + np.random.randn(len(audio)) * 0.001
        return np.clip(audio, -1.0, 1.0)


def collate_fn(batch):
    return {
        'source_audio':    torch.stack([b['source_audio'] for b in batch]),
        'source_midi':     torch.tensor([b['source_midi'] for b in batch], dtype=torch.long),
        'source_velocity': torch.tensor([b['source_velocity'] for b in batch], dtype=torch.long),
        'target_audio':    torch.stack([b['target_audio'] for b in batch]),
        'target_midi':     torch.tensor([b['target_midi'] for b in batch], dtype=torch.long),
        'target_velocity': torch.tensor([b['target_velocity'] for b in batch], dtype=torch.long),
    }
