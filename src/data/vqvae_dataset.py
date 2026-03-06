"""
PyTorch Dataset for VQ-VAE training.

Loads audio samples with MIDI pitch information for pitch-conditioned training.
"""

import json
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from scipy import signal
from torch.utils.data import Dataset


class VQVAEDataset(Dataset):
    """
    Dataset for VQ-VAE training.

    Loads audio files and metadata (MIDI note, velocity, instrument).
    """

    def __init__(self, metadata_path, sample_rate=44100, duration=4.0,
                 augment=False):
        """
        Args:
            metadata_path: Path to JSON file with sample metadata
            sample_rate: Target sample rate
            duration: Target duration in seconds
            augment: Whether to apply data augmentation
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.augment = augment

        # Load metadata
        with open(metadata_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from {metadata_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and process a single sample.

        Returns:
            Dictionary with:
                - audio: (channels, length) audio tensor
                - midi_note: MIDI note number (int)
                - velocity: Velocity value (int)
                - instrument_id: Instrument identifier (int)
        """
        sample = self.samples[idx]

        # Load audio
        audio = self._load_audio(sample['path'])

        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment(audio)

        # Convert to tensor
        audio = torch.from_numpy(audio).float()

        # Ensure mono (1, length)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        return {
            'audio': audio,
            'midi_note': sample['midi_note'],
            'velocity': sample['velocity'],
            'instrument_id': self._get_instrument_id(sample['instrument']),
            'path': sample['path'],
        }

    def _load_audio(self, path):
        """Load and preprocess audio file."""
        try:
            # Open file to read metadata before committing to a full read.
            # Many samples are 10-42s long; we only need target_length frames,
            # so seek to the desired start and read only what we need.
            with sf.SoundFile(path) as f:
                file_sr = f.samplerate
                total_frames = f.frames

                # How many source frames do we need at the file's native rate?
                frames_needed = int(self.target_length * file_sr / self.sample_rate) + 1

                if total_frames <= frames_needed:
                    # Short file — read everything
                    f.seek(0)
                    audio = f.read(always_2d=False)
                else:
                    # Pick start before reading to avoid loading unused data
                    if self.augment:
                        start = np.random.randint(0, total_frames - frames_needed + 1)
                    else:
                        start = 0
                    f.seek(start)
                    audio = f.read(frames=frames_needed, always_2d=False)

                sr = file_sr

            # Resample if needed
            if sr != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / sr)
                audio = signal.resample(audio, num_samples)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Pad or trim to target length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.target_length]

            # Reject any NaN/Inf that crept in from file or resampling
            if not np.isfinite(audio).all():
                print(f"Warning: NaN/Inf in audio {path}, replacing with zeros")
                return np.zeros(self.target_length, dtype=np.float32)

            # Normalize to [-1, 1]
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.95

            return audio.astype(np.float32)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(self.target_length, dtype=np.float32)

    def _augment(self, audio):
        """Apply data augmentation."""
        # Random gain adjustment (±3 dB)
        if np.random.rand() < 0.5:
            gain_db = np.random.uniform(-3, 3)
            gain = 10 ** (gain_db / 20)
            audio = audio * gain

        # Random polarity flip
        if np.random.rand() < 0.3:
            audio = -audio

        # Add subtle noise (very low level)
        if np.random.rand() < 0.3:
            noise = np.random.randn(len(audio)) * 0.001
            audio = audio + noise

        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def _get_instrument_id(self, instrument_name):
        """
        Map instrument name to integer ID.

        For single-instrument training, always returns 0.
        For multi-instrument, create a mapping.
        """
        # Simple hash-based mapping
        # In practice, you might want a more stable mapping
        if not hasattr(self, '_instrument_map'):
            self._instrument_map = {}
            self._next_id = 0

        if instrument_name not in self._instrument_map:
            self._instrument_map[instrument_name] = self._next_id
            self._next_id += 1

        return self._instrument_map[instrument_name]


def collate_fn(batch):
    """
    Custom collate function for batching.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with tensors
    """
    # Stack audio
    audio = torch.stack([item['audio'] for item in batch])

    # Stack metadata
    midi_notes = torch.tensor([item['midi_note'] for item in batch], dtype=torch.long)
    velocities = torch.tensor([item['velocity'] for item in batch], dtype=torch.long)
    instrument_ids = torch.tensor([item['instrument_id'] for item in batch], dtype=torch.long)

    return {
        'audio': audio,
        'midi_note': midi_notes,
        'velocity': velocities,
        'instrument_id': instrument_ids,
    }


def create_dataloader(metadata_path, batch_size=16, sample_rate=44100,
                     duration=4.0, augment=False, num_workers=4,
                     shuffle=True, pin_memory=True):
    """
    Factory function to create DataLoader.

    Args:
        metadata_path: Path to JSON metadata
        batch_size: Batch size
        sample_rate: Target sample rate
        duration: Audio duration in seconds
        augment: Enable data augmentation
        num_workers: Number of worker processes
        shuffle: Shuffle data
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    dataset = VQVAEDataset(
        metadata_path=metadata_path,
        sample_rate=sample_rate,
        duration=duration,
        augment=augment
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for stable training
    )

    return dataloader
