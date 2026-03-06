"""Audio utility functions for loading, saving, and processing audio."""

import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional


def load_audio(path: str, sample_rate: int = 44100) -> torch.Tensor:
    """
    Load and resample audio file.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (default: 44100)

    Returns:
        Audio tensor of shape [channels, samples]
    """
    # Load audio using soundfile
    audio_np, sr = sf.read(path, always_2d=True)

    # Convert to torch tensor [channels, samples]
    audio = torch.from_numpy(audio_np.T).float()

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)

    return audio


def save_audio(audio: torch.Tensor, path: str, sample_rate: int = 44100):
    """
    Save audio tensor to file.

    Args:
        audio: Audio tensor of shape [channels, samples] or [samples]
        path: Output file path
        sample_rate: Sample rate (default: 44100)
    """
    # Ensure audio is 2D
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Convert to numpy and save (detach first to handle gradients)
    audio_np = audio.detach().cpu().numpy()
    sf.write(path, audio_np.T, sample_rate)


def pitch_shift_audio(audio: torch.Tensor, n_semitones: float, sample_rate: int = 44100) -> torch.Tensor:
    """
    Pitch shift audio by n semitones using librosa.

    Args:
        audio: Audio tensor of shape [channels, samples]
        n_semitones: Number of semitones to shift (positive = higher, negative = lower)
        sample_rate: Sample rate of audio

    Returns:
        Pitch-shifted audio tensor with same shape
    """
    if n_semitones == 0:
        return audio

    # Convert to numpy
    audio_np = audio.cpu().numpy()

    # Process each channel
    shifted_channels = []
    for channel_audio in audio_np:
        # Pitch shift using librosa
        shifted = librosa.effects.pitch_shift(
            channel_audio,
            sr=sample_rate,
            n_steps=n_semitones
        )
        shifted_channels.append(shifted)

    # Convert back to torch tensor
    shifted_audio = torch.from_numpy(np.stack(shifted_channels)).float()

    return shifted_audio


def normalize_audio(audio: torch.Tensor, method: str = 'peak') -> torch.Tensor:
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio: Audio tensor
        method: Normalization method ('peak' or 'rms')

    Returns:
        Normalized audio tensor
    """
    if method == 'peak':
        # Peak normalization
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    elif method == 'rms':
        # RMS normalization
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            audio = audio / (rms * 10)  # Scale to reasonable level
            audio = torch.clamp(audio, -1.0, 1.0)

    return audio


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 44100,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> torch.Tensor:
    """
    Compute mel-spectrogram representation.

    Args:
        audio: Audio tensor of shape [channels, samples]
        sample_rate: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length for STFT
        f_min: Minimum frequency
        f_max: Maximum frequency (default: sample_rate/2)

    Returns:
        Mel-spectrogram tensor of shape [channels, n_mels, time]
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Compute mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max
    )

    mel_spec = mel_transform(audio)

    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)

    return mel_spec


def pad_or_trim_audio(
    audio: torch.Tensor,
    target_length: int,
    mode: str = 'constant'
) -> torch.Tensor:
    """
    Pad or trim audio to target length.

    Args:
        audio: Audio tensor of shape [channels, samples]
        target_length: Target length in samples
        mode: Padding mode ('constant', 'reflect', 'replicate')

    Returns:
        Audio tensor of shape [channels, target_length]
    """
    current_length = audio.shape[-1]

    if current_length < target_length:
        # Pad
        pad_length = target_length - current_length
        if mode == 'constant':
            audio = torch.nn.functional.pad(audio, (0, pad_length), mode='constant', value=0)
        else:
            audio = torch.nn.functional.pad(audio, (0, pad_length), mode=mode)
    elif current_length > target_length:
        # Trim (center crop)
        start = (current_length - target_length) // 2
        audio = audio[..., start:start + target_length]

    return audio


def extract_envelope(
    audio: torch.Tensor,
    sample_rate: int = 44100,
    hop_length: int = 512
) -> torch.Tensor:
    """
    Extract amplitude envelope from audio.

    Args:
        audio: Audio tensor of shape [channels, samples]
        sample_rate: Sample rate
        hop_length: Hop length for envelope extraction

    Returns:
        Envelope tensor
    """
    # Compute absolute value
    abs_audio = torch.abs(audio)

    # Apply moving average filter
    window_size = hop_length
    kernel = torch.ones(1, 1, window_size) / window_size

    # Pad for same size output
    padding = window_size // 2
    abs_audio_padded = torch.nn.functional.pad(abs_audio.unsqueeze(0), (padding, padding), mode='reflect')

    # Apply convolution
    envelope = torch.nn.functional.conv1d(abs_audio_padded, kernel)

    return envelope.squeeze(0)


def compute_rms_energy(
    audio: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512
) -> torch.Tensor:
    """
    Compute RMS energy over time.

    Args:
        audio: Audio tensor of shape [channels, samples]
        frame_length: Frame length for RMS computation
        hop_length: Hop length between frames

    Returns:
        RMS energy tensor
    """
    # Unfold audio into frames
    frames = audio.unfold(-1, frame_length, hop_length)

    # Compute RMS for each frame
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1))

    return rms


def detect_onset(
    audio: torch.Tensor,
    sample_rate: int = 44100,
    threshold: float = 0.1
) -> int:
    """
    Detect onset (attack) point in audio.

    Args:
        audio: Audio tensor of shape [channels, samples]
        sample_rate: Sample rate
        threshold: Threshold for onset detection (0-1)

    Returns:
        Onset sample index
    """
    # Compute energy
    energy = compute_rms_energy(audio, frame_length=512, hop_length=128)

    # Normalize energy
    max_energy = torch.max(energy)
    if max_energy > 0:
        energy = energy / max_energy

    # Find first point above threshold
    above_threshold = torch.where(energy > threshold)[0]

    if len(above_threshold) > 0:
        onset_frame = above_threshold[0].item()
        onset_sample = onset_frame * 128  # hop_length
        return onset_sample
    else:
        return 0
