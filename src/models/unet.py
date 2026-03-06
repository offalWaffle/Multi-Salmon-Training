"""
U-Net architecture for latent diffusion model.
Operates on DAC latent representations [batch, 1024, 345].
"""

import torch
import torch.nn as nn
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (like in Transformer)."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # MLP to project timestep embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps):
        """
        Args:
            timesteps: Tensor of shape [batch] with timestep values

        Returns:
            Tensor of shape [batch, dim]
        """
        # Create sinusoidal embeddings
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # Project through MLP
        embeddings = self.mlp(embeddings)

        return embeddings


class ConditionEmbedding(nn.Module):
    """Embedding for MIDI note and input sample (velocity removed)."""

    def __init__(self, dim, input_embedding_dim=128):
        super().__init__()
        self.dim = dim

        # Calculate sizes for each component (only MIDI and input, no velocity)
        midi_dim = dim // 2
        input_dim = dim - midi_dim  # Remaining to ensure exact sum

        # Separate embeddings for each condition
        self.midi_embed = nn.Sequential(
            nn.Linear(1, midi_dim),
            nn.SiLU(),
            nn.Linear(midi_dim, midi_dim),
        )

        self.input_embed = nn.Sequential(
            nn.Linear(input_embedding_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, midi_note, input_embedding):
        """
        Args:
            midi_note: [batch] or [batch, 1] normalized MIDI note (0-1)
            input_embedding: [batch, input_embedding_dim]

        Returns:
            Tensor of shape [batch, dim]
        """
        # Ensure midi_note is [batch, 1] regardless of input shape
        if midi_note.dim() == 1:
            midi_note = midi_note.unsqueeze(-1)
        elif midi_note.dim() > 2:
            midi_note = midi_note.reshape(-1, 1)
        # If already [batch, 1], use as-is

        midi_emb = self.midi_embed(midi_note)
        input_emb = self.input_embed(input_embedding)

        # Concatenate embeddings
        return torch.cat([midi_emb, input_emb], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with group normalization and conditioning."""

    def __init__(self, in_channels, out_channels, embed_dim, dropout=0.1, num_groups=32):
        super().__init__()

        # First conv block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # Embedding projection (for timestep + condition)
        self.embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, emb):
        """
        Args:
            x: [batch, in_channels, time]
            emb: [batch, embed_dim] - combined timestep + condition embedding

        Returns:
            [batch, out_channels, time]
        """
        residual = self.residual_conv(x)

        # First conv
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add embedding
        emb_out = self.embed_proj(emb)[:, :, None]  # [batch, out_channels, 1]
        h = h + emb_out

        # Second conv
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """Self-attention block for better long-range dependencies."""

    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [batch, channels, time]

        Returns:
            [batch, channels, time]
        """
        batch, channels, time = x.shape
        residual = x

        # Normalize
        h = self.norm(x)

        # QKV projection
        qkv = self.qkv(h)  # [batch, channels*3, time]
        qkv = qkv.reshape(batch, 3, self.num_heads, channels // self.num_heads, time)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: [batch, num_heads, head_dim, time]

        # Attention
        q = q.permute(0, 1, 3, 2)  # [batch, num_heads, time, head_dim]
        k = k.permute(0, 1, 3, 2)  # [batch, num_heads, time, head_dim]
        v = v.permute(0, 1, 3, 2)  # [batch, num_heads, time, head_dim]

        scale = (channels // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)  # [batch, num_heads, time, head_dim]

        # Reshape and project
        h = h.permute(0, 1, 3, 2).reshape(batch, channels, time)
        h = self.proj(h)

        return h + residual


class Downsample(nn.Module):
    """Downsampling by 2x."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling by 2x."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, target_size=None):
        """
        Args:
            x: Input tensor
            target_size: Optional target size for upsampling (for matching skip connections)
        """
        if target_size is not None:
            x = nn.functional.interpolate(x, size=target_size, mode='nearest')
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """
    U-Net for latent diffusion, conditioned on:
    - Timestep (diffusion step)
    - MIDI note
    - Velocity layer
    - Input sample embedding (for timbral consistency)

    Operates on DAC latent space: [batch, 1024, 345]
    """

    def __init__(
        self,
        in_channels=1024,
        model_channels=256,
        out_channels=1024,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(2,),
        dropout=0.1,
        embed_dim=512,
        input_embedding_dim=128,
        num_heads=4,
        num_groups=32,
    ):
        """
        Args:
            in_channels: Input channels (DAC latent channels = 1024)
            model_channels: Base channel count
            out_channels: Output channels (same as input)
            num_res_blocks: Number of residual blocks per level
            channel_mult: Channel multiplier per level
            attention_resolutions: Which levels to use attention (by downsampling factor)
            dropout: Dropout probability
            embed_dim: Embedding dimension for timestep + conditions
            input_embedding_dim: Dimension of input sample embedding
            num_heads: Number of attention heads
            num_groups: Number of groups for group normalization
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = attention_resolutions
        self.embed_dim = embed_dim

        # Timestep embedding
        self.time_embed = TimestepEmbedding(embed_dim)

        # Condition embedding
        self.cond_embed = ConditionEmbedding(embed_dim, input_embedding_dim)

        # Input projection
        self.input_conv = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()

        ch = model_channels
        input_block_channels = [ch]
        ds = 1  # Downsampling factor

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, out_ch, embed_dim * 2, dropout, num_groups)
                ]

                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads, num_groups))

                self.down_blocks.append(nn.ModuleList(layers))
                ch = out_ch
                input_block_channels.append(ch)

            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_channels.append(ch)
                ds *= 2

        # Middle block
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, embed_dim * 2, dropout, num_groups),
            AttentionBlock(ch, num_heads, num_groups),
            ResidualBlock(ch, ch, embed_dim * 2, dropout, num_groups),
        ])

        # Upsampling path
        self.up_blocks = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                # Skip connection from downsampling
                ich = input_block_channels.pop()

                layers = [
                    ResidualBlock(ch + ich, out_ch, embed_dim * 2, dropout, num_groups)
                ]

                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads, num_groups))

                self.up_blocks.append(nn.ModuleList(layers))
                ch = out_ch

            # Upsample (except last level)
            if level != len(channel_mult) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                ds //= 2

        # Output projection
        self.output_norm = nn.GroupNorm(num_groups, ch)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep, midi_note, input_embedding):
        """
        Args:
            x: Latent representation [batch, 1024, 345]
            timestep: Diffusion timestep [batch]
            midi_note: MIDI note normalized [batch]
            input_embedding: Embedding from input samples [batch, input_embedding_dim]

        Returns:
            Predicted noise [batch, 1024, 345]
        """
        # Embed timestep and conditions
        t_emb = self.time_embed(timestep)
        c_emb = self.cond_embed(midi_note, input_embedding)

        # Combine embeddings
        emb = torch.cat([t_emb, c_emb], dim=-1)

        # Input projection
        h = self.input_conv(x)

        # Downsampling path with skip connections
        skips = [h]

        for blocks in self.down_blocks:
            for layer in blocks:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Downsample):
                    h = layer(h)

            skips.append(h)

        # Middle block
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, emb)
            else:  # AttentionBlock
                h = layer(h)

        # Upsampling path with skip connections
        for blocks in self.up_blocks:
            # Check if this is an upsample block
            if len(blocks) == 1 and isinstance(blocks[0], Upsample):
                # Get target size from next skip connection
                if len(skips) > 0:
                    target_size = skips[-1].shape[-1]
                    h = blocks[0](h, target_size=target_size)
                else:
                    h = blocks[0](h)
            else:
                # Concatenate skip connection
                skip = skips.pop()

                # Handle dimension mismatch by padding if needed
                if h.shape[-1] != skip.shape[-1]:
                    diff = skip.shape[-1] - h.shape[-1]
                    h = nn.functional.pad(h, (0, diff))

                h = torch.cat([h, skip], dim=1)

                for layer in blocks:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, AttentionBlock):
                        h = layer(h)

        # Output projection
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)

        return h


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = 'cpu'

    print("Testing ConditionalUNet...")
    model = ConditionalUNet(
        in_channels=1024,
        model_channels=256,
        out_channels=1024,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(2,),
        embed_dim=512,
        input_embedding_dim=128,
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1024, 345).to(device)
    timestep = torch.randint(0, 1000, (batch_size,)).to(device)
    midi_note = torch.rand(batch_size).to(device)
    velocity = torch.rand(batch_size).to(device)
    input_embedding = torch.randn(batch_size, 128).to(device)

    print(f"Input shape: {x.shape}")
    output = model(x, timestep, midi_note, velocity, input_embedding)
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass successful!")
