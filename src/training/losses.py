"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    VAE loss = Reconstruction loss + KL divergence.

    Args:
        recon_x: Reconstructed input [batch, ...]
        x: Original input [batch, ...]
        mu: Mean of latent distribution [batch, latent_dim]
        logvar: Log variance of latent distribution [batch, latent_dim]
        beta: Weight for KL term (beta-VAE for disentanglement)

    Returns:
        Total loss scalar
    """
    # Reconstruction loss (MSE)
    # Use mean reduction to be batch-size independent
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss


def vae_loss_with_components(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                              logvar: torch.Tensor, beta: float = 1.0) -> tuple:
    """
    VAE loss with individual components returned for logging.

    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def spectral_convergence_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Spectral convergence loss for better audio quality.

    Args:
        recon_x: Reconstructed spectrogram
        x: Original spectrogram

    Returns:
        Spectral convergence loss
    """
    return torch.norm(x - recon_x, p='fro') / torch.norm(x, p='fro')


def log_magnitude_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Log magnitude loss for better perceptual quality.

    Args:
        recon_x: Reconstructed spectrogram
        x: Original spectrogram

    Returns:
        Log magnitude loss
    """
    return F.l1_loss(torch.log(recon_x.abs() + 1e-5), torch.log(x.abs() + 1e-5))


def vae_loss_with_free_bits(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                             logvar: torch.Tensor, beta: float = 1.0,
                             free_bits: float = 0.0) -> tuple:
    """
    VAE loss with free bits to prevent posterior collapse.

    Free bits: Ignore KL divergence when it's below a threshold per dimension.
    This prevents the model from collapsing the latent space too aggressively.

    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution [batch, latent_dim]
        logvar: Log variance of latent distribution [batch, latent_dim]
        beta: Weight for KL term
        free_bits: Minimum KL per dimension (typical: 0.5)

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence per dimension
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [batch, latent_dim]

    if free_bits > 0:
        # Apply free bits: max(kl_per_dim, free_bits)
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Average over batch and sum over dimensions
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def multi_scale_mel_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    scales: List[int] = [128, 64, 32]
) -> torch.Tensor:
    """
    Multi-scale mel-spectrogram loss.

    Computes L1 loss at multiple frequency resolutions by downsampling
    the mel-spectrogram along the frequency dimension. This captures both
    coarse and fine-grained spectral features.

    Based on DAC-VAE and MelCap approaches for better high-frequency reconstruction.

    Args:
        recon_x: Reconstructed mel-spectrogram [batch, 1, n_mels, time]
        x: Original mel-spectrogram [batch, 1, n_mels, time]
        scales: List of frequency bin counts to evaluate at (default: [128, 64, 32])

    Returns:
        Multi-scale loss (average of losses at all scales)
    """
    total_loss = 0.0

    for scale in scales:
        # Downsample to target scale along frequency dimension (dim=2)
        if scale == x.shape[2]:
            # No downsampling needed
            x_scaled = x
            recon_scaled = recon_x
        else:
            # Interpolate to target frequency resolution
            x_scaled = F.interpolate(
                x, size=(scale, x.shape[3]),
                mode='bilinear', align_corners=False
            )
            recon_scaled = F.interpolate(
                recon_x, size=(scale, recon_x.shape[3]),
                mode='bilinear', align_corners=False
            )

        # Compute L1 loss at this scale
        scale_loss = F.l1_loss(recon_scaled, x_scaled)
        total_loss += scale_loss

    # Average over scales
    return total_loss / len(scales)


def dac_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
    mel_loss_weight: float = 1.0,
    mel_scales: List[int] = [128, 64, 32]
) -> tuple:
    """
    DAC-VAE loss combining multi-scale mel loss and KL divergence.

    This is the recommended loss for Phase 1 of DAC-VAE implementation.
    Multi-scale mel loss provides better perceptual guidance than simple MSE.

    Args:
        recon_x: Reconstructed mel-spectrogram
        x: Original mel-spectrogram
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term (default: 0.1, lower than standard VAE)
        mel_loss_weight: Weight for mel reconstruction loss (default: 1.0)
        mel_scales: Frequency resolutions for multi-scale loss

    Returns:
        Tuple of (total_loss, mel_loss, kl_loss)
    """
    # Multi-scale mel reconstruction loss
    mel_loss = multi_scale_mel_loss(recon_x, x, scales=mel_scales)

    # KL divergence
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # Total loss
    total_loss = mel_loss_weight * mel_loss + beta * kl_loss

    return total_loss, mel_loss, kl_loss


def encoder_feature_matching_loss(
    encoder: nn.Module,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    feature_layers: List[str] = None
) -> torch.Tensor:
    """
    Feature-matching loss using encoder intermediate features.

    Compares intermediate activations when the encoder processes the original
    vs reconstructed samples. This prevents the decoder from ignoring latents
    by ensuring reconstructions pass through similar feature representations.

    This is a simplified version of feature-matching that doesn't require
    a discriminator. For full adversarial feature-matching, use a discriminator.

    Args:
        encoder: The encoder network
        original: Original input [batch, channels, height, width]
        reconstruction: Reconstructed input [batch, channels, height, width]
        feature_layers: List of layer names to extract features from
                       If None, uses all convolutional layers

    Returns:
        Feature matching loss (L1 distance between feature activations)
    """
    # Hook to capture intermediate features
    features_original = {}
    features_recon = {}

    def get_activation(name, feature_dict):
        def hook(module, input, output):
            feature_dict[name] = output.detach()
        return hook

    # Register hooks for convolutional layers
    hooks = []
    if hasattr(encoder, 'encoder_blocks'):
        # DAC-style encoder
        for i, block in enumerate(encoder.encoder_blocks):
            name = f'encoder_block_{i}'
            hooks.append(block.register_forward_hook(
                get_activation(name, features_original)
            ))
    elif hasattr(encoder, 'conv_layers'):
        # Standard encoder
        for i, layer in enumerate(encoder.conv_layers):
            if isinstance(layer, nn.Conv2d):
                name = f'conv_{i}'
                hooks.append(layer.register_forward_hook(
                    get_activation(name, features_original)
                ))

    # Forward pass with original
    with torch.no_grad():
        _ = encoder(original)

    # Update hooks to capture reconstruction features
    for hook in hooks:
        hook.remove()

    hooks = []
    if hasattr(encoder, 'encoder_blocks'):
        for i, block in enumerate(encoder.encoder_blocks):
            name = f'encoder_block_{i}'
            hooks.append(block.register_forward_hook(
                get_activation(name, features_recon)
            ))
    elif hasattr(encoder, 'conv_layers'):
        for i, layer in enumerate(encoder.conv_layers):
            if isinstance(layer, nn.Conv2d):
                name = f'conv_{i}'
                hooks.append(layer.register_forward_hook(
                    get_activation(name, features_recon)
                ))

    # Forward pass with reconstruction
    _ = encoder(reconstruction)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute L1 loss between features
    total_loss = 0.0
    num_layers = 0

    for name in features_original.keys():
        if name in features_recon:
            feat_orig = features_original[name]
            feat_recon = features_recon[name]
            total_loss += F.l1_loss(feat_recon, feat_orig)
            num_layers += 1

    # Average over layers
    if num_layers > 0:
        total_loss = total_loss / num_layers

    return total_loss


def perceptual_reconstruction_loss(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    p: int = 2
) -> torch.Tensor:
    """
    Perceptual reconstruction loss in frequency domain.

    Computes loss on different frequency bands separately, giving more weight
    to perceptually important frequency ranges.

    Args:
        original: Original mel-spectrogram [batch, channels, n_mels, time]
        reconstruction: Reconstructed mel-spectrogram
        p: Lp norm to use (1 for L1, 2 for L2)

    Returns:
        Perceptual reconstruction loss
    """
    # Split into frequency bands (low, mid, high)
    n_mels = original.shape[2]

    # Low frequencies (0-1kHz region, roughly first 1/4 of mels)
    low_end = n_mels // 4
    low_orig = original[:, :, :low_end, :]
    low_recon = reconstruction[:, :, :low_end, :]

    # Mid frequencies (1-4kHz region, middle half)
    mid_start = low_end
    mid_end = 3 * n_mels // 4
    mid_orig = original[:, :, mid_start:mid_end, :]
    mid_recon = reconstruction[:, :, mid_start:mid_end, :]

    # High frequencies (4kHz+, top quarter)
    high_orig = original[:, :, mid_end:, :]
    high_recon = reconstruction[:, :, mid_end:, :]

    # Compute losses with different weights
    # Higher weight for mid frequencies (most perceptually important)
    if p == 1:
        low_loss = F.l1_loss(low_recon, low_orig)
        mid_loss = F.l1_loss(mid_recon, mid_orig)
        high_loss = F.l1_loss(high_recon, high_orig)
    else:
        low_loss = F.mse_loss(low_recon, low_orig)
        mid_loss = F.mse_loss(mid_recon, mid_orig)
        high_loss = F.mse_loss(high_recon, high_orig)

    # Weighted combination (mid frequencies more important)
    total_loss = 0.3 * low_loss + 0.5 * mid_loss + 0.2 * high_loss

    return total_loss
