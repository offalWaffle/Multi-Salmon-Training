"""Training loops (active DAC pitch-adapter path).

VQ-VAE / latent-transformer trainers live in legacy/ — see legacy/README.md.
"""

from .dac_adapter_trainer import DACAdapterTrainer

__all__ = ['DACAdapterTrainer']