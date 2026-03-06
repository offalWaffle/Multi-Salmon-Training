"""
Variational Autoencoder for audio compression.
Encodes mel-spectrogram to latent space, decodes back to mel-spectrogram.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """CNN encoder for mel-spectrogram."""

    def __init__(self, in_channels: int = 1, channels: list = [64, 128, 256, 512], latent_dim: int = 128):
        """
        Initialize encoder.

        Args:
            in_channels: Number of input channels (1 for mono mel-spectrogram)
            channels: List of channel sizes for each layer
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Build encoder layers with progressive downsampling
        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(channels):
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            prev_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size after convolutions (depends on input size)
        # For mel-spectrogram of shape [128, 345], after 4 downsampling layers (2x each):
        # 128 -> 64 -> 32 -> 16 -> 8
        # 345 -> 172 -> 86 -> 43 -> 21
        # So final size is approximately [512, 8, 21] = 512 * 8 * 21 = 86016
        # We'll use adaptive pooling to make this more flexible

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        final_size = channels[-1] * 4 * 4

        # Projection to latent space (mu and logvar)
        self.fc_mu = nn.Linear(final_size, latent_dim)
        self.fc_logvar = nn.Linear(final_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input mel-spectrogram [batch, 1, n_mels, time]

        Returns:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance of latent distribution [batch, latent_dim]
        """
        # Apply convolutional layers
        h = self.conv_layers(x)

        # Adaptive pooling to fixed size
        # Note: MPS has issues with adaptive pooling, so we move to CPU if needed
        if h.device.type == 'mps':
            h_cpu = h.cpu()
            h_pooled = self.adaptive_pool(h_cpu)
            h = h_pooled.to(h.device)
        else:
            h = self.adaptive_pool(h)

        # Flatten
        h = h.view(h.size(0), -1)

        # Project to mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """CNN decoder for mel-spectrogram."""

    def __init__(self, latent_dim: int = 128, channels: list = [512, 256, 128, 64],
                 out_channels: int = 1, output_size: Tuple[int, int] = (128, 345)):
        """
        Initialize decoder.

        Args:
            latent_dim: Dimension of latent space
            channels: List of channel sizes for each layer (reversed from encoder)
            out_channels: Number of output channels (1 for mono mel-spectrogram)
            output_size: Target output size (n_mels, time_frames)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        # Start with small spatial size
        self.init_size = 4
        self.fc = nn.Linear(latent_dim, channels[0] * self.init_size * self.init_size)

        # Build decoder layers with progressive upsampling
        layers = []
        prev_channels = channels[0]

        for i, out_ch in enumerate(channels[1:]):
            layers.extend([
                nn.ConvTranspose2d(prev_channels, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            prev_channels = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Final layer to reconstruct mel-spectrogram
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Mel-spectrograms are typically in log scale, so tanh output is reasonable
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to mel-spectrogram.

        Args:
            z: Latent vector [batch, latent_dim]

        Returns:
            Reconstructed mel-spectrogram [batch, 1, n_mels, time]
        """
        # Project latent to spatial dimensions
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.init_size, self.init_size)

        # Apply deconvolutional layers
        h = self.conv_layers(h)
        h = self.final_conv(h)

        # Resize to exact output size if needed
        if h.shape[2:] != self.output_size:
            h = F.interpolate(h, size=self.output_size, mode='bilinear', align_corners=False)

        return h


class AudioVAE(nn.Module):
    """
    Variational Autoencoder for audio compression.
    Encodes mel-spectrogram to latent space, decodes back to mel-spectrogram.
    """

    def __init__(self, config):
        """
        Initialize VAE.

        Args:
            config: Configuration dict with keys:
                - in_channels: Number of input channels
                - encoder_channels: List of encoder channel sizes
                - decoder_channels: List of decoder channel sizes
                - latent_dim: Latent dimension
                - output_size: Optional tuple (n_mels, time_frames)
        """
        super().__init__()

        # Get config parameters
        in_channels = config.get('in_channels', 1)
        encoder_channels = config.get('encoder_channels', [64, 128, 256, 512])
        decoder_channels = config.get('decoder_channels', [512, 256, 128, 64])
        latent_dim = config.get('latent_dim', 128)

        # Calculate output size from audio config if available
        if 'output_size' in config:
            output_size = config['output_size']
        else:
            # Default: assume 128 mel bins and calculate time frames
            # With 4 seconds at 44.1kHz and hop_length=512: (4 * 44100) / 512 ≈ 345
            output_size = (128, 345)

        self.encoder = Encoder(
            in_channels=in_channels,
            channels=encoder_channels,
            latent_dim=latent_dim
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            channels=decoder_channels,
            out_channels=in_channels,
            output_size=output_size
        )

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution."""
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input mel-spectrogram [batch, channels, n_mels, time]

        Returns:
            recon: Reconstructed mel-spectrogram
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from prior distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated mel-spectrograms
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
