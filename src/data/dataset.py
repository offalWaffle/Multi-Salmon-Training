"""PyTorch dataset for multi-sample instruments."""

import os
import json
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path


class InstrumentDataset(Dataset):
    """PyTorch dataset for multi-sample instruments."""

    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        num_input_samples: int = 5,
        normalize_midi: bool = True,
        normalize_velocity: bool = True,
        pitch_augmentation: bool = False,
        pitch_shift_prob: float = 0.5,
        max_pitch_shift: int = 6
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing processed samples
            transform: Optional transform to apply to audio
            num_input_samples: Number of input samples for conditioning (1-5)
            normalize_midi: Whether to normalize MIDI notes to [0, 1]
            normalize_velocity: Whether to normalize velocity to [0, 1]
            pitch_augmentation: Whether to apply random pitch shifting
            pitch_shift_prob: Probability of applying pitch shift (0.0-1.0)
            max_pitch_shift: Maximum pitch shift in semitones (e.g., 6 = ±6 semitones)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.num_input_samples = num_input_samples
        self.normalize_midi = normalize_midi
        self.normalize_velocity = normalize_velocity
        self.pitch_augmentation = pitch_augmentation
        self.pitch_shift_prob = pitch_shift_prob
        self.max_pitch_shift = max_pitch_shift

        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Find all sample files
        self.sample_files = sorted(Path(data_dir).glob('sample_*.pt'))

        if len(self.sample_files) == 0:
            raise ValueError(f"No samples found in {data_dir}")

        # Build index of samples by instrument
        self._build_instrument_index()

        print(f"Loaded {len(self.sample_files)} samples from {data_dir}")
        print(f"Number of instruments: {len(self.instrument_to_samples)}")

    def _build_instrument_index(self):
        """Build index mapping instrument IDs to sample indices."""
        import gc

        self.instrument_to_samples = {}

        for idx, sample_file in enumerate(self.sample_files):
            # Load sample with mmap to avoid loading full tensor into memory
            sample_data = torch.load(sample_file, map_location='cpu', weights_only=False)
            instrument_id = sample_data['instrument_id']

            if instrument_id not in self.instrument_to_samples:
                self.instrument_to_samples[instrument_id] = []

            self.instrument_to_samples[instrument_id].append(idx)

            # Explicitly delete loaded data to free memory
            del sample_data

            # Periodic garbage collection every 100 files
            if (idx + 1) % 100 == 0:
                gc.collect()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Dictionary with:
                - audio: Audio tensor [channels, samples]
                - midi_note: MIDI note (normalized if normalize_midi=True)
                - velocity: Velocity (normalized if normalize_velocity=True)
                - input_samples: Tensor of input samples for conditioning [num_input_samples, channels, samples]
                - num_input_samples: Actual number of input samples (may be less than max)
        """
        # Load target sample
        sample_data = torch.load(self.sample_files[idx])

        # Clone tensors to avoid mmap issues with DataLoader workers
        audio = sample_data['audio'].clone()

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        midi_note = sample_data['midi_note']
        velocity = sample_data['velocity']
        instrument_id = sample_data['instrument_id']

        # Apply pitch augmentation if enabled
        if self.pitch_augmentation and random.random() < self.pitch_shift_prob:
            # Randomly shift pitch by -max_pitch_shift to +max_pitch_shift semitones
            pitch_shift = random.randint(-self.max_pitch_shift, self.max_pitch_shift)

            if pitch_shift != 0:
                # Import pitch shift function
                from src.data.audio_utils import pitch_shift_audio

                # Shift audio
                audio = pitch_shift_audio(audio, n_semitones=pitch_shift, sample_rate=44100)

                # Update MIDI note to match the shifted audio
                midi_note = midi_note + pitch_shift

                # Clamp to valid piano range (21-108)
                midi_note = max(21, min(108, midi_note))

        # Normalize MIDI note (21-108 -> 0-1)
        if self.normalize_midi:
            midi_note_normalized = (midi_note - 21) / (108 - 21)
        else:
            midi_note_normalized = midi_note

        # Normalize velocity (0-127 -> 0-1)
        if self.normalize_velocity:
            velocity_normalized = velocity / 127.0
        else:
            velocity_normalized = velocity

        # Sample input samples from the same instrument
        instrument_samples = self.instrument_to_samples[instrument_id]

        # Determine how many input samples to use (random between 1 and num_input_samples)
        actual_num_input = min(
            random.randint(1, self.num_input_samples),
            len(instrument_samples)
        )

        # Sample random samples from the same instrument (excluding current sample)
        available_samples = [i for i in instrument_samples if i != idx]

        if len(available_samples) >= actual_num_input:
            input_indices = random.sample(available_samples, actual_num_input)
        elif len(available_samples) > 0:
            # If not enough other samples, use what we have
            input_indices = available_samples
            actual_num_input = len(input_indices)
        else:
            # If no other samples (single-sample instrument), use current sample
            input_indices = [idx]
            actual_num_input = 1

        # Load input samples
        input_samples_list = []
        for input_idx in input_indices:
            input_data = torch.load(self.sample_files[input_idx])
            # Clone to avoid mmap issues
            input_audio = input_data['audio'].clone()

            # Convert to mono if stereo
            if input_audio.shape[0] > 1:
                input_audio = torch.mean(input_audio, dim=0, keepdim=True)

            input_samples_list.append(input_audio)

        # Pad with zeros if needed to reach num_input_samples
        while len(input_samples_list) < self.num_input_samples:
            input_samples_list.append(torch.zeros_like(audio))

        # Stack input samples
        input_samples = torch.stack(input_samples_list)

        # Apply transform if provided
        if self.transform is not None:
            audio = self.transform(audio)
            input_samples = torch.stack([self.transform(s) for s in input_samples])

        return {
            'audio': audio,
            'midi_note': torch.tensor(midi_note_normalized, dtype=torch.float32),
            'velocity': torch.tensor(velocity_normalized, dtype=torch.float32),
            'input_samples': input_samples,
            'num_input_samples': torch.tensor(actual_num_input, dtype=torch.long),
            'instrument_id': torch.tensor(instrument_id, dtype=torch.long),
            'original_midi_note': torch.tensor(midi_note, dtype=torch.long),
            'original_velocity': torch.tensor(velocity, dtype=torch.long)
        }

    def get_instrument_ids(self) -> List[int]:
        """Get list of all unique instrument IDs."""
        return list(self.instrument_to_samples.keys())

    def get_samples_for_instrument(self, instrument_id: int) -> List[int]:
        """Get all sample indices for a given instrument."""
        return self.instrument_to_samples.get(instrument_id, [])

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading audio."""
        sample_data = torch.load(self.sample_files[idx])
        return {
            'midi_note': sample_data['midi_note'],
            'velocity': sample_data['velocity'],
            'instrument_id': sample_data['instrument_id'],
            'file_path': sample_data.get('file_path', 'unknown')
        }


class MelSpectrogramTransform:
    """Transform that converts audio to mel-spectrogram."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        """
        Initialize transform.

        Args:
            sample_rate: Sample rate
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length
        """
        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply transform.

        Args:
            audio: Audio tensor [channels, samples]

        Returns:
            Mel-spectrogram [channels, n_mels, time]
        """
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Compute mel-spectrogram
        mel_spec = self.mel_transform(audio)

        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec


class LatentDataset(Dataset):
    """
    PyTorch dataset for pre-encoded DAC latents.
    Loads latents directly instead of raw audio - much faster for training.
    """

    def __init__(
        self,
        data_dir: str,
        num_input_samples: int = 5,
        normalize_midi: bool = True,
        normalize_velocity: bool = True
    ):
        """
        Initialize latent dataset.

        Args:
            data_dir: Directory containing pre-encoded latent files
            num_input_samples: Number of input samples for conditioning (1-5)
            normalize_midi: Whether to normalize MIDI notes to [0, 1]
            normalize_velocity: Whether to normalize velocity to [0, 1]
        """
        self.data_dir = data_dir
        self.num_input_samples = num_input_samples
        self.normalize_midi = normalize_midi
        self.normalize_velocity = normalize_velocity

        # Find all latent files
        self.latent_files = sorted(Path(data_dir).glob('sample_*.pt'))

        if len(self.latent_files) == 0:
            raise ValueError(f"No latent files found in {data_dir}")

        # Build index of samples by instrument
        self._build_instrument_index()

        print(f"Loaded {len(self.latent_files)} latent samples from {data_dir}")
        print(f"Number of instruments: {len(self.instrument_to_samples)}")

    def _build_instrument_index(self):
        """Build index mapping instrument IDs to sample indices."""
        import gc

        self.instrument_to_samples = {}

        for idx, latent_file in enumerate(self.latent_files):
            # Load latent with mmap to avoid loading full tensor into memory
            latent_data = torch.load(latent_file, map_location='cpu', weights_only=False)

            # Handle both old format (instrument_id) and new format (instrument_name)
            if 'instrument_id' in latent_data:
                instrument_id = latent_data['instrument_id']
            else:
                # Create ID from instrument name
                instrument_name = latent_data.get('instrument_name', 'unknown')
                instrument_id = hash(instrument_name) % 10000

            if instrument_id not in self.instrument_to_samples:
                self.instrument_to_samples[instrument_id] = []

            self.instrument_to_samples[instrument_id].append(idx)

            # Explicitly delete loaded data to free memory
            del latent_data

            # Periodic garbage collection every 100 files
            if (idx + 1) % 100 == 0:
                gc.collect()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.latent_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Dictionary with:
                - latent: Pre-encoded DAC latent [1024, time]
                - midi_note: MIDI note (normalized if normalize_midi=True)
                - velocity: Velocity (normalized if normalize_velocity=True)
                - input_samples: Tensor of input latents for conditioning [num_input_samples, 1024, time]
                - num_input_samples: Actual number of input samples
        """
        # Load target latent
        latent_data = torch.load(self.latent_files[idx])

        # Clone tensor to avoid mmap issues
        latent = latent_data['latent'].clone()  # [1024, time]

        midi_note = latent_data['midi_note']
        velocity = latent_data['velocity']

        # Get instrument ID
        if 'instrument_id' in latent_data:
            instrument_id = latent_data['instrument_id']
        else:
            instrument_name = latent_data.get('instrument_name', 'unknown')
            instrument_id = hash(instrument_name) % 10000

        # Normalize MIDI note (21-108 -> 0-1)
        if self.normalize_midi:
            midi_note_normalized = (midi_note - 21) / (108 - 21)
        else:
            midi_note_normalized = midi_note

        # Normalize velocity (0-127 -> 0-1)
        if self.normalize_velocity:
            velocity_normalized = velocity / 127.0
        else:
            velocity_normalized = velocity

        # Sample input latents from the same instrument
        instrument_samples = self.instrument_to_samples[instrument_id]

        # Determine how many input samples to use
        actual_num_input = min(
            random.randint(1, self.num_input_samples),
            len(instrument_samples)
        )

        # Sample random samples from the same instrument (excluding current sample)
        available_samples = [i for i in instrument_samples if i != idx]

        if len(available_samples) >= actual_num_input:
            input_indices = random.sample(available_samples, actual_num_input)
        elif len(available_samples) > 0:
            input_indices = available_samples
            actual_num_input = len(input_indices)
        else:
            # If no other samples, use current sample
            input_indices = [idx]
            actual_num_input = 1

        # Load input latents
        input_latents_list = []
        for input_idx in input_indices:
            input_data = torch.load(self.latent_files[input_idx])
            input_latent = input_data['latent'].clone()
            input_latents_list.append(input_latent)

        # Pad with zeros if needed
        while len(input_latents_list) < self.num_input_samples:
            input_latents_list.append(torch.zeros_like(latent))

        # Stack input latents
        input_latents = torch.stack(input_latents_list)

        return {
            'latent': latent,
            'midi_note': torch.tensor(midi_note_normalized, dtype=torch.float32),
            'velocity': torch.tensor(velocity_normalized, dtype=torch.float32),
            'input_latents': input_latents,
            'num_input_samples': torch.tensor(actual_num_input, dtype=torch.long),
            'instrument_id': torch.tensor(instrument_id, dtype=torch.long),
            'original_midi_note': torch.tensor(midi_note, dtype=torch.long),
            'original_velocity': torch.tensor(velocity, dtype=torch.long)
        }

    def get_instrument_ids(self) -> List[int]:
        """Get list of all unique instrument IDs."""
        return list(self.instrument_to_samples.keys())


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary
    """
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'midi_note': torch.stack([item['midi_note'] for item in batch]),
        'velocity': torch.stack([item['velocity'] for item in batch]),
        'input_samples': torch.stack([item['input_samples'] for item in batch]),
        'num_input_samples': torch.stack([item['num_input_samples'] for item in batch]),
        'instrument_id': torch.stack([item['instrument_id'] for item in batch]),
        'original_midi_note': torch.stack([item['original_midi_note'] for item in batch]),
        'original_velocity': torch.stack([item['original_velocity'] for item in batch])
    }


def collate_fn_latents(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for LatentDataset.

    Args:
        batch: List of samples from LatentDataset.__getitem__

    Returns:
        Batched dictionary
    """
    return {
        'latent': torch.stack([item['latent'] for item in batch]),
        'midi_note': torch.stack([item['midi_note'] for item in batch]),
        'velocity': torch.stack([item['velocity'] for item in batch]),
        'input_latents': torch.stack([item['input_latents'] for item in batch]),
        'num_input_samples': torch.stack([item['num_input_samples'] for item in batch]),
        'instrument_id': torch.stack([item['instrument_id'] for item in batch]),
        'original_midi_note': torch.stack([item['original_midi_note'] for item in batch]),
        'original_velocity': torch.stack([item['original_velocity'] for item in batch])
    }
