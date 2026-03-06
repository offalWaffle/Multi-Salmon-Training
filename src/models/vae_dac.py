"""
DAC-style Variational Autoencoder for audio compression.
Based on Descript Audio Codec architecture with continuous latent space.

Key improvements over basic VAE:
- Residual connections for better gradient flow
- Group normalization (better for audio and small batches)
- Deeper convolutional architecture
- Multi-scale mel loss support (implemented in losses.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ResidualBlock(nn.Module):
    """
    Residual block with group normalization.
    Used in both encoder and decoder.
    """

    def __init__(self, channels: int, dilation: int = 1):
        """
        Initialize residual block.

        Args:
            channels: Number of channels
            dilation: Dilation factor for convolution
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=1
        )

        # Group normalization (8 groups is standard)
        # If channels < 8, use channels as num_groups
        num_groups = min(8, channels)
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.norm2 = nn.GroupNorm(num_groups, channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        # Residual connection
        x = x + residual
        x = self.activation(x)

        return x


class EncoderBlock(nn.Module):
    """
    Encoder block with downsampling.
    Consists of residual unit + strided convolution for downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        """
        Initialize encoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for downsampling (default: 2 for 2x downsampling)
        """
        super().__init__()

        # Residual unit (processes at input channel count)
        self.residual = ResidualBlock(in_channels)

        # Downsampling layer (strided convolution)
        self.downsample = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=4, stride=stride, padding=1
        )

        num_groups = min(8, out_channels)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: residual -> downsample."""
        x = self.residual(x)
        x = self.downsample(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling.
    Consists of transposed convolution for upsampling + residual unit.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        """
        Initialize decoder block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for upsampling (default: 2 for 2x upsampling)
        """
        super().__init__()

        # Upsampling layer (transposed convolution)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=stride, padding=1
        )

        num_groups = min(8, out_channels)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Residual unit (processes at output channel count)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: upsample -> residual."""
        x = self.upsample(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.residual(x)
        return x


class DACEncoder(nn.Module):
    """
    DAC-style encoder for mel-spectrogram.
    Progressive downsampling with residual connections.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [64, 128, 256, 512],
        latent_dim: int = 128
    ):
        """
        Initialize DAC encoder.

        Args:
            in_channels: Number of input channels (1 for mono mel-spectrogram)
            channels: List of channel sizes for each encoder block
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Initial convolution (no downsampling)
        self.input_conv = nn.Conv2d(
            in_channels, channels[0],
            kernel_size=7, padding=3
        )
        num_groups = min(8, channels[0])
        self.input_norm = nn.GroupNorm(num_groups, channels[0])
        self.input_activation = nn.LeakyReLU(0.2, inplace=True)

        # Build encoder blocks with progressive downsampling
        self.encoder_blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            block = EncoderBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                stride=2
            )
            self.encoder_blocks.append(block)

        # Add one more residual block at the end (no downsampling)
        self.final_residual = ResidualBlock(channels[-1])

        # Adaptive pooling to fixed size before latent projection
        # This makes the architecture flexible to different input sizes
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
        # Initial convolution
        h = self.input_conv(x)
        h = self.input_norm(h)
        h = self.input_activation(h)

        # Progressive downsampling through encoder blocks
        for block in self.encoder_blocks:
            h = block(h)

        # Final residual processing
        h = self.final_residual(h)

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


class DACDecoder(nn.Module):
    """
    DAC-style decoder for mel-spectrogram.
    Progressive upsampling with residual connections.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        channels: List[int] = [512, 256, 128, 64],
        out_channels: int = 1,
        output_size: Tuple[int, int] = (128, 345)
    ):
        """
        Initialize DAC decoder.

        Args:
            latent_dim: Dimension of latent space
            channels: List of channel sizes for each decoder block (high to low)
            out_channels: Number of output channels (1 for mono mel-spectrogram)
            output_size: Target output size (n_mels, time_frames)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        # Start with small spatial size
        self.init_size = 4
        self.fc = nn.Linear(latent_dim, channels[0] * self.init_size * self.init_size)

        # Initial processing
        self.initial_residual = ResidualBlock(channels[0])

        # Build decoder blocks with progressive upsampling
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            block = DecoderBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                stride=2
            )
            self.decoder_blocks.append(block)

        # Final convolution to output
        self.output_conv = nn.Conv2d(
            channels[-1], out_channels,
            kernel_size=7, padding=3
        )
        # Use tanh activation for normalized mel-spectrogram output
        self.output_activation = nn.Tanh()

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

        # Initial residual processing
        h = self.initial_residual(h)

        # Progressive upsampling through decoder blocks
        for block in self.decoder_blocks:
            h = block(h)

        # Final convolution to output channels
        h = self.output_conv(h)
        h = self.output_activation(h)

        # Resize to exact output size if needed
        if h.shape[2:] != self.output_size:
            h = F.interpolate(h, size=self.output_size, mode='bilinear', align_corners=False)

        return h


class DACVAE(nn.Module):
    """
    DAC-style Variational Autoencoder for audio compression.

    Key features:
    - Residual connections throughout
    - Group normalization (better for audio)
    - Deep convolutional architecture
    - Designed for multi-scale mel loss
    """

    def __init__(self, config):
        """
        Initialize DAC-VAE.

        Args:
            config: Configuration dict with keys:
                - in_channels: Number of input channels (default: 1)
                - encoder_channels: List of encoder channel sizes (default: [64, 128, 256, 512])
                - decoder_channels: List of decoder channel sizes (default: [512, 256, 128, 64])
                - latent_dim: Latent dimension (default: 128)
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
            # Default: 128 mel bins, 345 time frames (4s at 44.1kHz)
            output_size = (128, 345)

        self.encoder = DACEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            latent_dim=latent_dim
        )

        self.decoder = DACDecoder(
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
