"""
VQ-VAE for pitch-conditioned audio generation.

Architecture:
    Input audio → Encoder → Vector Quantizer → Decoder (pitch-conditioned) → Output audio

Key features:
- Pitch-invariant latent codes (timbre representation)
- FiLM-based pitch conditioning in decoder
- Discrete codebook for latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import VectorQuantizer, EMAVectorQuantizer
from .film_layers import FiLMConv1d, FiLMResBlock
from .pitch_conditioning import SinusoidalPitchEmbedding


class Encoder(nn.Module):
    """
    CNN-based encoder for audio to latent codes.

    Converts waveform to compressed latent representation.
    Should be pitch-invariant (timbre-focused).
    """

    def __init__(self, in_channels=1, hidden_dim=128, latent_dim=64, num_layers=4):
        """
        Args:
            in_channels: Number of input channels (1 for mono, 2 for stereo)
            hidden_dim: Hidden layer dimension
            latent_dim: Output latent dimension
            num_layers: Number of downsampling layers
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        layers = []

        # Initial conv
        layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3))
        layers.append(nn.ReLU())

        # Downsampling layers (strided convolutions)
        current_dim = hidden_dim
        for i in range(num_layers):
            next_dim = min(hidden_dim * (2 ** (i + 1)), 512)

            # Downsampling conv (stride=2)
            layers.append(nn.Conv1d(current_dim, next_dim, kernel_size=4,
                                   stride=2, padding=1))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.ReLU())

            # Additional conv at this resolution
            layers.append(nn.Conv1d(next_dim, next_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.ReLU())

            current_dim = next_dim

        # Final projection to latent dimension
        layers.append(nn.Conv1d(current_dim, latent_dim, kernel_size=1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, length) input audio

        Returns:
            (batch, latent_dim, length // downsampling_factor) latent codes
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Pitch-conditioned decoder with FiLM layers.

    Reconstructs audio from latent codes, conditioned on target pitch.
    """

    def __init__(self, latent_dim=64, hidden_dim=128, out_channels=1,
                 num_layers=4, pitch_embed_dim=128):
        """
        Args:
            latent_dim: Input latent dimension
            hidden_dim: Hidden layer dimension
            out_channels: Number of output channels (1 for mono)
            num_layers: Number of upsampling layers
            pitch_embed_dim: Pitch embedding dimension
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.pitch_embed_dim = pitch_embed_dim

        # Pitch embedding
        self.pitch_embedding = SinusoidalPitchEmbedding(pitch_embed_dim)

        # Initial projection
        start_dim = min(hidden_dim * (2 ** num_layers), 512)
        self.initial = nn.Conv1d(latent_dim, start_dim, kernel_size=1)

        # Upsampling layers with FiLM conditioning
        self.upsample_layers = nn.ModuleList()
        current_dim = start_dim

        for i in range(num_layers):
            # Cap at start_dim so the decoder never exceeds the encoder's max channels.
            # Without this, the first upsample step would expand to hidden_dim * 2^num_layers
            # (e.g. 1024 with hidden_dim=128, num_layers=4) — larger than the encoder's 512.
            next_dim = max(min(hidden_dim * (2 ** (num_layers - i - 1)), start_dim), hidden_dim)

            # FiLM residual block
            self.upsample_layers.append(
                FiLMResBlock(current_dim, cond_dim=pitch_embed_dim)
            )

            # Upsampling (transposed conv)
            self.upsample_layers.append(
                nn.ConvTranspose1d(current_dim, next_dim, kernel_size=4,
                                  stride=2, padding=1)
            )
            self.upsample_layers.append(nn.BatchNorm1d(next_dim))
            self.upsample_layers.append(nn.ReLU())

            # FiLM conv
            self.upsample_layers.append(
                FiLMConv1d(next_dim, next_dim, kernel_size=3, padding=1,
                          cond_dim=pitch_embed_dim, activation=nn.ReLU)
            )

            current_dim = next_dim

        # Final output projection
        self.output = nn.Sequential(
            nn.Conv1d(current_dim, out_channels, kernel_size=7, padding=3),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, z, pitch):
        """
        Args:
            z: (batch, latent_dim, time) latent codes
            pitch: (batch,) MIDI note numbers for conditioning

        Returns:
            (batch, out_channels, length) reconstructed audio
        """
        # Get pitch embeddings
        pitch_emb = self.pitch_embedding(pitch)  # (batch, pitch_embed_dim)

        # Initial projection
        x = self.initial(z)

        # Apply upsampling layers with pitch conditioning
        for layer in self.upsample_layers:
            if isinstance(layer, (FiLMResBlock, FiLMConv1d)):
                x = layer(x, pitch_emb)
            else:
                x = layer(x)

        # Final output
        x = self.output(x)

        return x


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model for pitch-conditioned audio generation.

    Combines encoder, vector quantizer, and pitch-conditioned decoder.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration dict with model parameters
        """
        super().__init__()

        # Extract config
        self.in_channels = config.get('in_channels', 1)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.latent_dim = config.get('latent_dim', 64)
        self.num_layers = config.get('num_layers', 4)
        self.pitch_embed_dim = config.get('pitch_embed_dim', 128)

        # Quantizer config
        self.num_embeddings = config.get('num_embeddings', 512)
        self.commitment_cost = config.get('commitment_cost', 0.25)
        self.use_ema = config.get('use_ema', False)

        # Build model
        self.encoder = Encoder(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers
        )

        # Vector Quantizer
        if self.use_ema:
            self.quantizer = EMAVectorQuantizer(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.latent_dim,
                commitment_cost=self.commitment_cost
            )
        else:
            self.quantizer = VectorQuantizer(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.latent_dim,
                commitment_cost=self.commitment_cost
            )

        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            out_channels=self.in_channels,
            num_layers=self.num_layers,
            pitch_embed_dim=self.pitch_embed_dim
        )

    def forward(self, x, pitch):
        """
        Forward pass through VQ-VAE.

        Args:
            x: (batch, in_channels, length) input audio
            pitch: (batch,) MIDI note numbers

        Returns:
            reconstruction: Reconstructed audio
            vq_loss: Vector quantization loss
            perplexity: Codebook usage metric
            encoding_indices: Discrete codes
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, vq_loss, perplexity, encoding_indices = self.quantizer(z)

        # Decode with pitch conditioning
        reconstruction = self.decoder(z_q, pitch)

        # Strided convolutions use floor division, so the decoder output can
        # differ from the input length by a few samples.  Trim or pad to match.
        if reconstruction.shape[-1] != x.shape[-1]:
            diff = x.shape[-1] - reconstruction.shape[-1]
            if diff > 0:
                reconstruction = F.pad(reconstruction, (0, diff))
            else:
                reconstruction = reconstruction[..., :x.shape[-1]]

        # Return both z (pre-quantization) and z_q (post-quantization).
        # z_q is needed by the adversarial pitch classifier in the trainer.
        return reconstruction, vq_loss, perplexity, encoding_indices, z, z_q

    def encode(self, x):
        """
        Encode audio to discrete codes.

        Args:
            x: (batch, in_channels, length) input audio

        Returns:
            encoding_indices: (batch, code_length) discrete codes
        """
        z = self.encoder(x)
        _, _, _, encoding_indices = self.quantizer(z)
        return encoding_indices

    def decode(self, encoding_indices, pitch):
        """
        Decode discrete codes to audio.

        Args:
            encoding_indices: (batch, code_length) discrete codes
            pitch: (batch,) MIDI note numbers

        Returns:
            (batch, in_channels, length) reconstructed audio
        """
        z_q = self.quantizer.quantize(encoding_indices)
        reconstruction = self.decoder(z_q, pitch)
        return reconstruction

    def reconstruct(self, x, pitch):
        """
        Reconstruct audio (encode then decode).

        Args:
            x: (batch, in_channels, length) input audio
            pitch: (batch,) MIDI note numbers

        Returns:
            reconstruction: Reconstructed audio
        """
        encoding_indices = self.encode(x)
        reconstruction = self.decode(encoding_indices, pitch)
        return reconstruction

    @property
    def downsampling_factor(self):
        """Calculate total downsampling factor of encoder."""
        return 2 ** self.num_layers


def create_vqvae(config_path=None, **kwargs):
    """
    Factory function to create VQ-VAE model.

    Args:
        config_path: Path to YAML config file (optional)
        **kwargs: Override config parameters

    Returns:
        VQVAE model instance
    """
    if config_path is not None:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override with kwargs
    config.update(kwargs)

    # Default configuration
    default_config = {
        'in_channels': 1,
        'hidden_dim': 128,
        'latent_dim': 64,
        'num_layers': 4,
        'pitch_embed_dim': 128,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'use_ema': False,
    }

    # Merge with defaults
    for key, value in default_config.items():
        config.setdefault(key, value)

    return VQVAE(config)
