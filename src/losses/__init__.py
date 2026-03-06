"""
Loss functions for VQ-VAE training.
"""

from .spectral_loss import (
    STFTLoss,
    MultiResolutionSTFTLoss,
    MelSpectrogramLoss,
    AudioReconstructionLoss,
    SpectralConvergenceLoss,
)

from .disentanglement_loss import (
    ContrastiveLoss,
    InstrumentConsistencyLoss,
    PitchInvarianceLoss,
    DisentanglementLoss,
    CodebookUsageLoss,
)

__all__ = [
    'STFTLoss',
    'MultiResolutionSTFTLoss',
    'MelSpectrogramLoss',
    'AudioReconstructionLoss',
    'SpectralConvergenceLoss',
    'ContrastiveLoss',
    'InstrumentConsistencyLoss',
    'PitchInvarianceLoss',
    'DisentanglementLoss',
    'CodebookUsageLoss',
]
