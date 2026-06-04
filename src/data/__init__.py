"""Data loading modules (active DAC pitch-adapter path).

VQ-VAE / transformer datasets live in legacy/ — see legacy/README.md.
"""

from .audio_utils import load_audio, save_audio, normalize_audio, compute_mel_spectrogram
from .dac_latent_dataset import (
    DACLatentDataset,
    DACPitchPairDataset,
    create_dataloader,
    create_pair_dataloader,
)

__all__ = [
    'load_audio', 'save_audio', 'normalize_audio', 'compute_mel_spectrogram',
    'DACLatentDataset', 'DACPitchPairDataset', 'create_dataloader', 'create_pair_dataloader',
]