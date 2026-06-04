"""Model architectures (active DAC pitch-adapter path).

Earlier VQ-VAE / latent-transformer / adversarial models live in legacy/ — see
legacy/README.md and root README §8.
"""

from .conditioned_dac import ConditionedDAC, FiLMConditionedDAC, create_conditioned_dac, load_pretrained_dac
from .dac_pitch_adapter import DACPitchAdapter, PitchStripper, PitchInjector, PitchProbe
from .film_layers import FiLM, FiLMConv1d, FiLMResBlock
from .pitch_conditioning import SinusoidalPitchEmbedding, PitchEmbedding

__all__ = [
    'ConditionedDAC', 'FiLMConditionedDAC', 'create_conditioned_dac', 'load_pretrained_dac',
    'DACPitchAdapter', 'PitchStripper', 'PitchInjector', 'PitchProbe',
    'FiLM', 'FiLMConv1d', 'FiLMResBlock',
    'SinusoidalPitchEmbedding', 'PitchEmbedding',
]