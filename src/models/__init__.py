"""Model architectures."""

from .conditioned_dac import ConditionedDAC, FiLMConditionedDAC, create_conditioned_dac, load_pretrained_dac
from .dac_pitch_adapter import DACPitchAdapter, PitchStripper, PitchInjector
from .film_layers import FiLM, FiLMConv1d, FiLMResBlock
from .pitch_conditioning import SinusoidalPitchEmbedding, PitchEmbedding
from .pitch_adversary import GradientReversal, PitchClassifier
from .vqvae import VQVAE
from .quantizers import VectorQuantizer
from .latent_transformer import LatentTransformer

__all__ = [
    'ConditionedDAC', 'FiLMConditionedDAC', 'create_conditioned_dac', 'load_pretrained_dac',
    'DACPitchAdapter', 'PitchStripper', 'PitchInjector',
    'FiLM', 'FiLMConv1d', 'FiLMResBlock',
    'SinusoidalPitchEmbedding', 'PitchEmbedding',
    'GradientReversal', 'PitchClassifier',
    'VQVAE', 'VectorQuantizer',
    'LatentTransformer',
]
