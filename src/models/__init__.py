"""Model architectures."""

from .vae import AudioVAE, Encoder, Decoder
from .vae_dac import DACVAE, DACEncoder, DACDecoder
from .conditioned_dac import (
    ConditionedDAC,
    FiLMConditionedDAC,
    create_conditioned_dac,
    load_pretrained_dac
)

# Phase 4 models:
from .diffusion import LatentDiffusion
from .unet import ConditionalUNet, TimestepEmbedding, ConditionEmbedding
from .input_encoder import InputSampleEncoder, SimpleInputEncoder

__all__ = [
    'AudioVAE',
    'Encoder',
    'Decoder',
    'DACVAE',
    'DACEncoder',
    'DACDecoder',
    'ConditionedDAC',
    'FiLMConditionedDAC',
    'create_conditioned_dac',
    'load_pretrained_dac',
    # Phase 4:
    'LatentDiffusion',
    'ConditionalUNet',
    'TimestepEmbedding',
    'ConditionEmbedding',
    'InputSampleEncoder',
    'SimpleInputEncoder',
]
