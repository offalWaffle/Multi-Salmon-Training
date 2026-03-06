"""
Dataset for DAC Pitch Adapter training.

Loads matched (latent, audio, midi_note) pairs from precomputed DAC latents.
The latent .pt files already point to their corresponding audio .pt files via
the 'original_audio_path' key, so no separate manifest is needed.

Latent format (data/processed_latents/{split}/sample_XXXXXX.pt):
    {
        'latent':               Tensor [1024, T]
        'midi_note':            int
        'velocity':             int
        'instrument_id':        int
        'original_audio_path':  str  →  data/processed/{split}/sample_XXXXXX.pt
    }

Audio format (data/processed/{split}/sample_XXXXXX.pt):
    {
        'audio': Tensor [2, N]   # stereo, converted to mono [1, N] on load
        ...
    }

Crop alignment:
    crop_frames latent frames ↔ crop_frames * DAC_HOP audio samples
    DAC_HOP = 512  (DAC 44 kHz model)
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

# DAC 44 kHz encoder hop size
DAC_HOP = 512


class DACLatentDataset(Dataset):
    """
    Dataset that yields (latent_crop, audio_crop, midi_note, velocity, instrument_id) tuples.

    All cropping is done on the pre-computed latent tensor; the corresponding audio
    chunk is derived from the same start frame to keep them aligned.
    """

    def __init__(self, latent_dir, duration=2.0, sample_rate=44100, augment=False):
        """
        Args:
            latent_dir:   Directory containing sample_XXXXXX.pt latent files.
            duration:     Crop length in seconds.
            sample_rate:  Audio sample rate (must match DAC model, default 44100).
            augment:      If True, use random crop start; else use the centre.
        """
        self.latent_dir = Path(latent_dir)
        self.augment = augment
        self.sample_rate = sample_rate

        # Number of latent frames to crop
        self.crop_frames = int(duration * sample_rate / DAC_HOP)
        # Corresponding audio samples (what DAC decoder will output for crop_frames)
        self.crop_audio = self.crop_frames * DAC_HOP

        # Collect all .pt files (excluding metadata files)
        self.files = sorted(
            p for p in self.latent_dir.glob('sample_*.pt')
        )

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No sample_*.pt files found in {latent_dir}"
            )

        print(
            f"DACLatentDataset: {len(self.files)} samples from {latent_dir} "
            f"(crop={self.crop_frames} frames / {self.crop_audio} samples)"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=False)

        latent = data['latent']          # [1024, T]
        midi_note = int(data['midi_note'])
        velocity = int(data.get('velocity', 80))
        instrument_id = int(data.get('instrument_id', 0))
        audio_path = data.get('original_audio_path', '')

        # --- Crop latent ---
        T = latent.shape[-1]
        if T < self.crop_frames:
            # Pad with zeros if shorter than crop
            pad = self.crop_frames - T
            latent = torch.nn.functional.pad(latent, (0, pad))
            latent_start = 0
        elif T == self.crop_frames:
            latent_start = 0
        else:
            if self.augment:
                latent_start = np.random.randint(0, T - self.crop_frames + 1)
            else:
                latent_start = (T - self.crop_frames) // 2

        latent_crop = latent[:, latent_start: latent_start + self.crop_frames]  # [1024, crop_frames]

        # --- Load and crop audio ---
        audio_start = latent_start * DAC_HOP
        audio_crop = self._load_audio_crop(audio_path, audio_start)

        return {
            'latent':        latent_crop,       # [1024, crop_frames]
            'audio':         audio_crop,        # [1, crop_audio]
            'midi_note':     midi_note,
            'velocity':      velocity,
            'instrument_id': instrument_id,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_audio_crop(self, audio_path, start_sample):
        """
        Load a .pt audio file and return a mono [1, crop_audio] crop.

        Falls back to silence on any error.
        """
        silence = torch.zeros(1, self.crop_audio)

        if not audio_path:
            return silence

        try:
            adata = torch.load(audio_path, map_location='cpu', weights_only=False)
            if isinstance(adata, dict):
                audio = adata['audio']   # [channels, N]
            else:
                audio = adata            # assume [channels, N] or [N]

            # Convert to mono
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)   # [1, N]
            elif audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)  # [1, N]

            N = audio.shape[-1]
            end = start_sample + self.crop_audio

            if N < self.crop_audio:
                # Pad
                pad = self.crop_audio - N
                audio = torch.nn.functional.pad(audio, (0, pad))
            elif end > N:
                # Start too late after padding latent; clamp
                start_sample = max(0, N - self.crop_audio)
                end = start_sample + self.crop_audio

            crop = audio[:, start_sample: start_sample + self.crop_audio]

            # Guard against NaN/Inf
            if not torch.isfinite(crop).all():
                return silence

            # Peak-normalise to ≤ 0.95.  Without this, near-silent crops make the
            # spectral-convergence denominator collapse to ~1e-8, causing loss values
            # in the millions whose backward() produces Inf gradients.
            peak = crop.abs().max()
            if peak > 1e-4:
                crop = crop / peak * 0.95
            # If peak ≤ 1e-4 the crop is silent; return silence so spectral loss
            # gets a well-conditioned pair (0 vs ~0 prediction).
            else:
                return silence

            return crop.float()

        except Exception as e:
            print(f"Warning: could not load audio {audio_path}: {e}")
            return silence


def collate_fn(batch):
    return {
        'latent':        torch.stack([b['latent'] for b in batch]),
        'audio':         torch.stack([b['audio'] for b in batch]),
        'midi_note':     torch.tensor([b['midi_note'] for b in batch], dtype=torch.long),
        'velocity':      torch.tensor([b['velocity'] for b in batch], dtype=torch.long),
        'instrument_id': torch.tensor([b['instrument_id'] for b in batch], dtype=torch.long),
    }


def create_dataloader(latent_dir, batch_size=8, duration=2.0, sample_rate=44100,
                      augment=False, num_workers=0, shuffle=True, pin_memory=False):
    from torch.utils.data import DataLoader
    dataset = DACLatentDataset(
        latent_dir=latent_dir,
        duration=duration,
        sample_rate=sample_rate,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
