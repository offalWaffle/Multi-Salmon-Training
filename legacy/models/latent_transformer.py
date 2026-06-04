"""Latent Transformer model for direct pitch control in latent space."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LatentTransformer(nn.Module):
    """
    Direct latent transformation model for pitch control.

    Instead of diffusion-based generation, this model learns to directly transform
    VAE latents from one pitch to another while preserving timbre.

    Architecture:
        - Takes input latent + target MIDI note
        - Outputs transformed latent at target pitch
        - Preserves timbre information from input latent
        - Single forward pass (no iterative denoising)
    """

    def __init__(
        self,
        latent_channels: int = 64,
        latent_time_steps: int = 86,
        hidden_dim: int = 512,
        num_layers: int = 6,
        midi_embed_dim: int = 128,
        velocity_embed_dim: int = 64,
        use_residual: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize LatentTransformer.

        Args:
            latent_channels: Number of latent channels (64 for Stable Audio VAE)
            latent_time_steps: Temporal dimension of latents (86 for 2s at 44.1kHz)
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformation layers
            midi_embed_dim: MIDI embedding dimension
            velocity_embed_dim: Velocity embedding dimension
            use_residual: Whether to use residual connections
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_time_steps = latent_time_steps
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # MIDI note embedding (normalized 0-1 → embed_dim)
        self.midi_embed = nn.Sequential(
            nn.Linear(1, midi_embed_dim),
            nn.ReLU(),
            nn.Linear(midi_embed_dim, midi_embed_dim)
        )

        # Velocity embedding (normalized 0-1 → embed_dim)
        self.velocity_embed = nn.Sequential(
            nn.Linear(1, velocity_embed_dim),
            nn.ReLU(),
            nn.Linear(velocity_embed_dim, velocity_embed_dim)
        )

        # Input latent encoder (extract timbre features)
        self.input_encoder = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU()
        )

        # Combined conditioning projection (MIDI + velocity → hidden_dim)
        self.conditioning_proj = nn.Linear(midi_embed_dim + velocity_embed_dim, hidden_dim)

        # Transformation layers
        self.layers = nn.ModuleList([
            ResidualBlock(
                channels=hidden_dim + latent_channels,  # Concat original latent
                hidden_channels=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim + latent_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, latent_channels, kernel_size=3, padding=1)
        )

    def forward(
        self,
        input_latent: torch.Tensor,
        target_midi: torch.Tensor,
        target_velocity: torch.Tensor,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Transform input latent to target pitch and velocity.

        Args:
            input_latent: [B, C, T] - VAE latent from reference sample
            target_midi: [B] - Target MIDI note (normalized 0-1)
            target_velocity: [B] - Target velocity (normalized 0-1)
            return_intermediates: If True, return intermediate activations

        Returns:
            output_latent: [B, C, T] - Transformed latent at target pitch/velocity
            (optionally) intermediates: Dict of intermediate activations
        """
        B, C, T = input_latent.shape

        # Embed target MIDI note and velocity
        midi_emb = self.midi_embed(target_midi.unsqueeze(-1))  # [B, midi_embed_dim]
        velocity_emb = self.velocity_embed(target_velocity.unsqueeze(-1))  # [B, velocity_embed_dim]

        # Concatenate MIDI and velocity embeddings
        combined_emb = torch.cat([midi_emb, velocity_emb], dim=-1)  # [B, midi_embed_dim + velocity_embed_dim]

        # Project to hidden dimension
        cond = self.conditioning_proj(combined_emb)  # [B, hidden_dim]
        cond = cond.unsqueeze(-1).expand(-1, -1, T)  # [B, hidden_dim, T]

        # Encode input latent (timbre features)
        features = self.input_encoder(input_latent)  # [B, hidden_dim, T]

        # Add combined conditioning to features
        features = features + cond

        # Concatenate with original latent (for residual path)
        x = torch.cat([features, input_latent], dim=1)  # [B, hidden_dim + C, T]

        # Apply transformation layers
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            if return_intermediates:
                intermediates.append(x)

        # Project to output latent
        output_latent = self.output_proj(x)  # [B, C, T]

        # Optional residual connection with input
        if self.use_residual:
            output_latent = output_latent + input_latent

        if return_intermediates:
            return output_latent, {'intermediates': intermediates}

        return output_latent


class ResidualBlock(nn.Module):
    """Residual convolutional block with normalization."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = F.relu(x)

        return x


class MultiSampleLatentTransformer(nn.Module):
    """
    Extended version that can condition on multiple input samples.

    Uses attention over multiple reference samples to extract timbre information.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        latent_time_steps: int = 86,
        hidden_dim: int = 512,
        num_layers: int = 6,
        midi_embed_dim: int = 128,
        velocity_embed_dim: int = 64,
        max_input_samples: int = 5,
        use_residual: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize MultiSampleLatentTransformer.

        Args:
            latent_channels: Number of latent channels
            latent_time_steps: Temporal dimension of latents
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformation layers
            midi_embed_dim: MIDI embedding dimension
            velocity_embed_dim: Velocity embedding dimension
            max_input_samples: Maximum number of input reference samples
            use_residual: Whether to use residual connections
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.max_input_samples = max_input_samples

        # MIDI embedding
        self.midi_embed = nn.Sequential(
            nn.Linear(1, midi_embed_dim),
            nn.ReLU(),
            nn.Linear(midi_embed_dim, midi_embed_dim)
        )

        # Velocity embedding
        self.velocity_embed = nn.Sequential(
            nn.Linear(1, velocity_embed_dim),
            nn.ReLU(),
            nn.Linear(velocity_embed_dim, velocity_embed_dim)
        )

        # Encode each input sample
        self.sample_encoder = nn.Conv1d(latent_channels, hidden_dim, kernel_size=3, padding=1)

        # Attention over input samples
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Main transformer (same as single-sample version)
        self.conditioning_proj = nn.Linear(midi_embed_dim + velocity_embed_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ResidualBlock(
                channels=hidden_dim,
                hidden_channels=hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, latent_channels, kernel_size=3, padding=1)
        )

        self.use_residual = use_residual

    def forward(
        self,
        input_latents: torch.Tensor,
        target_midi: torch.Tensor,
        target_velocity: torch.Tensor,
        num_input_samples: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform multiple input latents to target pitch and velocity.

        Args:
            input_latents: [B, N, C, T] - N reference sample latents
            target_midi: [B] - Target MIDI note (normalized 0-1)
            target_velocity: [B] - Target velocity (normalized 0-1)
            num_input_samples: [B] - Actual number of valid input samples per batch

        Returns:
            output_latent: [B, C, T] - Transformed latent at target pitch/velocity
        """
        B, N, C, T = input_latents.shape

        # Embed target MIDI and velocity
        midi_emb = self.midi_embed(target_midi.unsqueeze(-1))  # [B, midi_embed_dim]
        velocity_emb = self.velocity_embed(target_velocity.unsqueeze(-1))  # [B, velocity_embed_dim]

        # Concatenate and project
        combined_emb = torch.cat([midi_emb, velocity_emb], dim=-1)
        cond = self.conditioning_proj(combined_emb)  # [B, hidden_dim]

        # Encode all input samples
        # Reshape to [B*N, C, T]
        input_flat = input_latents.reshape(B * N, C, T)
        encoded = self.sample_encoder(input_flat)  # [B*N, hidden_dim, T]

        # Global average pooling over time → [B*N, hidden_dim]
        encoded_pooled = encoded.mean(dim=-1)

        # Reshape to [B, N, hidden_dim]
        encoded_samples = encoded_pooled.reshape(B, N, -1)

        # Create attention mask if num_input_samples provided
        if num_input_samples is not None:
            # Mask out padding samples
            mask = torch.arange(N, device=input_latents.device).unsqueeze(0) >= num_input_samples.unsqueeze(1)
        else:
            mask = None

        # Attention over input samples (query = combined conditioning)
        query = cond.unsqueeze(1)  # [B, 1, hidden_dim]
        attended, _ = self.attention(
            query,
            encoded_samples,
            encoded_samples,
            key_padding_mask=mask
        )  # [B, 1, hidden_dim]

        # Broadcast to time dimension
        features = attended.squeeze(1).unsqueeze(-1).expand(-1, -1, T)  # [B, hidden_dim, T]

        # Apply transformation layers
        x = features
        for layer in self.layers:
            x = layer(x)

        # Project to output latent
        output_latent = self.output_proj(x)  # [B, C, T]

        # Optional residual with first input sample
        if self.use_residual:
            output_latent = output_latent + input_latents[:, 0]

        return output_latent


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test single-sample transformer
    print("Testing LatentTransformer...")
    model = LatentTransformer(
        latent_channels=64,
        latent_time_steps=86,
        hidden_dim=512,
        num_layers=6
    )

    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    input_latent = torch.randn(batch_size, 64, 86)
    target_midi = torch.rand(batch_size)  # Normalized MIDI
    target_velocity = torch.rand(batch_size)  # Normalized velocity

    output = model(input_latent, target_midi, target_velocity)
    print(f"Input shape: {input_latent.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == input_latent.shape

    # Test multi-sample transformer
    print("\nTesting MultiSampleLatentTransformer...")
    multi_model = MultiSampleLatentTransformer(
        latent_channels=64,
        latent_time_steps=86,
        hidden_dim=512,
        num_layers=6,
        max_input_samples=5
    )

    print(f"Parameters: {count_parameters(multi_model):,}")

    # Test forward pass
    input_latents = torch.randn(batch_size, 5, 64, 86)  # 5 reference samples
    num_samples = torch.tensor([5, 3, 4, 5])  # Variable number of valid samples

    output = multi_model(input_latents, target_midi, target_velocity, num_samples)
    print(f"Input shape: {input_latents.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 64, 86)

    print("\n✅ All tests passed!")
