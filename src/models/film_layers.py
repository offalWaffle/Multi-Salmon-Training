"""
Feature-wise Linear Modulation (FiLM) layers for conditional generation.

FiLM allows the decoder to adapt its processing based on pitch conditioning,
enabling pitch-controlled audio generation.
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Applies affine transformation to features based on conditioning:
        output = gamma * input + beta

    where gamma and beta are predicted from the conditioning signal.
    """

    def __init__(self, num_features, cond_dim):
        """
        Args:
            num_features: Number of feature channels to modulate
            cond_dim: Dimension of conditioning vector
        """
        super().__init__()
        self.num_features = num_features

        # Linear layers to predict gamma (scale) and beta (shift)
        self.scale_shift = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, conditioning):
        """
        Args:
            x: (batch, channels, *) features to modulate
            conditioning: (batch, cond_dim) conditioning vector

        Returns:
            Modulated features with same shape as x
        """
        # Predict scale and shift
        scale_shift = self.scale_shift(conditioning)  # (batch, num_features * 2)

        # Split into scale and shift
        scale, shift = scale_shift.chunk(2, dim=1)  # Each: (batch, num_features)

        # Reshape for broadcasting with feature maps
        # x shape: (batch, channels, *spatial_dims)
        # We need: (batch, channels, 1, 1, ...)
        shape = [x.shape[0], x.shape[1]] + [1] * (x.ndim - 2)
        scale = scale.view(*shape)
        shift = shift.view(*shape)

        # Apply affine transformation
        return scale * x + shift


class FiLMConv1d(nn.Module):
    """
    1D Convolution with FiLM conditioning.

    Combines convolution with feature-wise modulation for
    pitch-conditioned audio generation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, cond_dim=128, activation=nn.ReLU):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding)
        self.film = FiLM(out_channels, cond_dim)
        self.activation = activation() if activation is not None else nn.Identity()

    def forward(self, x, conditioning):
        """
        Args:
            x: (batch, in_channels, length) input features
            conditioning: (batch, cond_dim) conditioning vector

        Returns:
            (batch, out_channels, length') modulated features
        """
        x = self.conv(x)
        x = self.film(x, conditioning)
        x = self.activation(x)
        return x


class FiLMResBlock(nn.Module):
    """
    Residual block with FiLM conditioning.

    Useful for decoder networks that need to maintain
    skip connections while applying conditioning.
    """

    def __init__(self, channels, cond_dim=128, kernel_size=3, dilation=1):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.film1 = FiLM(channels, cond_dim)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.film2 = FiLM(channels, cond_dim)
        self.activation2 = nn.ReLU()

    def forward(self, x, conditioning):
        """
        Args:
            x: (batch, channels, length) input features
            conditioning: (batch, cond_dim) conditioning vector

        Returns:
            (batch, channels, length) output features
        """
        residual = x

        # First conv + FiLM
        out = self.conv1(x)
        out = self.film1(out, conditioning)
        out = self.activation1(out)

        # Second conv + FiLM
        out = self.conv2(out)
        out = self.film2(out, conditioning)

        # Residual connection
        out = out + residual
        out = self.activation2(out)

        return out


class AdaptiveInstanceNorm1d(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) for 1D signals.

    Alternative to FiLM that normalizes features before applying
    the conditioning-dependent affine transformation.
    """

    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.scale_shift = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, conditioning):
        """
        Args:
            x: (batch, channels, length) features
            conditioning: (batch, cond_dim) conditioning vector

        Returns:
            Normalized and modulated features
        """
        # Normalize
        x = self.norm(x)

        # Predict scale and shift from conditioning
        scale_shift = self.scale_shift(conditioning)
        scale, shift = scale_shift.chunk(2, dim=1)

        # Reshape for broadcasting
        scale = scale.view(x.shape[0], x.shape[1], 1)
        shift = shift.view(x.shape[0], x.shape[1], 1)

        # Apply affine transformation
        return scale * x + shift
