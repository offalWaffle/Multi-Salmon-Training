"""
Multi-Instrument Pitch Transformation Model

Direct audio-to-audio transformation with instrument, pitch, and velocity conditioning.
Uses stride convolution for compression and transformers for pitch transformation.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model, max_len=50000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]

        Returns:
            x with positional encoding added: [B, T, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class MultiInstrumentPitchTransformer(nn.Module):
    """
    Multi-instrument pitch transformation model.

    Architecture:
        Input: [B, 1, 176400] (44.1kHz, 4 seconds)
        ↓ Encoder (8x compression via stride conv)
        → [B, 256, 22050]
        ↓ + Instrument/Pitch/Velocity conditioning
        ↓ Transformer (12 layers, 8 heads)
        → [B, 256, 22050]
        ↓ Decoder (8x upsampling via stride transpose conv)
        → [B, 1, 176400]

    Note: Uses 8x compression (same as validated autoencoder).
    Results in 22,050 time steps with Nyquist frequency of 2.75 kHz.
    Self-attention memory: ~7-8 GiB (should fit on MPS).
    """

    def __init__(
        self,
        n_instruments=600,
        d_model=256,
        nhead=8,
        num_layers=12,
        dim_feedforward=1024,
        dropout=0.1
    ):
        """
        Initialize model.

        Args:
            n_instruments: Number of unique instruments
            d_model: Model dimension (must match encoder output channels)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension in transformer
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.n_instruments = n_instruments

        # ========================================
        # Encoder: 8x compression
        # 176400 → 88200 → 44100 → 22050
        # ========================================
        self.encoder = nn.Sequential(
            # Layer 1: 176400 → 88200 (stride 2)
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),

            # Layer 2: 88200 → 44100 (stride 2)
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            # Layer 3: 44100 → 22050 (stride 2)
            nn.Conv1d(128, d_model, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(d_model),
        )

        # ========================================
        # Conditioning embeddings
        # ========================================

        # Instrument embedding: 600 instruments → d_model
        self.instrument_embed = nn.Embedding(n_instruments, d_model)

        # Pitch embedding: continuous MIDI note (0-127) → 128-dim
        self.pitch_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Velocity embedding: continuous velocity (0-127) → 128-dim
        self.velocity_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Combine all conditioning: (d_model + 128 + 128) → d_model
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model + 128 + 128, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # ========================================
        # Transformer
        # ========================================

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=50000)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # [B, T, d_model]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ========================================
        # Decoder: 8x upsampling
        # 22050 → 44100 → 88200 → 176400
        # ========================================
        self.decoder = nn.Sequential(
            # Layer 1: 22050 → 44100 (stride 2)
            nn.ConvTranspose1d(d_model, 128, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            # Layer 2: 44100 → 88200 (stride 2)
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),

            # Layer 3: 88200 → 176400 (stride 2)
            nn.ConvTranspose1d(64, 1, kernel_size=16, stride=2, padding=7),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, audio, instrument_id, midi_note, velocity):
        """
        Forward pass.

        Args:
            audio: Input audio [B, 1, 176400]
            instrument_id: Instrument IDs [B] (long tensor, 0-599)
            midi_note: MIDI notes [B] (float tensor, 0-127)
            velocity: Velocities [B] (float tensor, 0-127)

        Returns:
            output_audio: Transformed audio [B, 1, 176400]
        """
        batch_size = audio.size(0)

        # ========================================
        # 1. Encode audio: [B, 1, 176400] → [B, d_model, 22050]
        # ========================================
        encoded = self.encoder(audio)  # [B, d_model, 22050]

        # ========================================
        # 2. Create conditioning vector
        # ========================================

        # Instrument embedding [B, d_model]
        instr_emb = self.instrument_embed(instrument_id)

        # Pitch embedding [B, 128]
        pitch_input = midi_note.unsqueeze(-1)  # [B, 1]
        pitch_emb = self.pitch_embed(pitch_input)  # [B, 128]

        # Velocity embedding [B, 128]
        vel_input = velocity.unsqueeze(-1)  # [B, 1]
        vel_emb = self.velocity_embed(vel_input)  # [B, 128]

        # Concatenate all conditioning
        cond = torch.cat([instr_emb, pitch_emb, vel_emb], dim=-1)  # [B, d_model+128+128]

        # Project to d_model
        cond = self.cond_proj(cond)  # [B, d_model]

        # ========================================
        # 3. Add conditioning to encoded audio
        # ========================================

        # Transpose for transformer: [B, d_model, T] → [B, T, d_model]
        encoded = encoded.transpose(1, 2)  # [B, 22050, d_model]

        # Add conditioning (broadcast across time dimension)
        encoded = encoded + cond.unsqueeze(1)  # [B, 22050, d_model]

        # ========================================
        # 4. Apply positional encoding
        # ========================================
        encoded = self.pos_encoder(encoded)  # [B, 22050, d_model]

        # ========================================
        # 5. Transformer processing
        # ========================================
        transformed = self.transformer(encoded)  # [B, 22050, d_model]

        # ========================================
        # 6. Decode: [B, 22050, d_model] → [B, 1, 176400]
        # ========================================

        # Transpose back: [B, T, d_model] → [B, d_model, T]
        transformed = transformed.transpose(1, 2)  # [B, d_model, 22050]

        # Decode to audio
        output_audio = self.decoder(transformed)  # [B, 1, ~176400]

        # Ensure exact output size matches input (crop/pad if needed)
        target_length = audio.size(2)
        current_length = output_audio.size(2)

        if current_length > target_length:
            # Crop
            output_audio = output_audio[:, :, :target_length]
        elif current_length < target_length:
            # Pad
            pad_amount = target_length - current_length
            output_audio = torch.nn.functional.pad(output_audio, (0, pad_amount))

        return output_audio

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    print("Testing MultiInstrumentPitchTransformer...")

    # Create model
    model = MultiInstrumentPitchTransformer(
        n_instruments=600,
        d_model=256,
        nhead=8,
        num_layers=12,
        dim_feedforward=1024,
        dropout=0.1
    )

    # Count parameters
    n_params = model.count_parameters()
    print(f"\nTotal parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Test forward pass
    batch_size = 2
    audio = torch.randn(batch_size, 1, 176400)
    instrument_id = torch.randint(0, 600, (batch_size,))
    midi_note = torch.randint(21, 108, (batch_size,)).float()
    velocity = torch.randint(0, 127, (batch_size,)).float()

    print(f"\nInput shapes:")
    print(f"  audio: {audio.shape}")
    print(f"  instrument_id: {instrument_id.shape}")
    print(f"  midi_note: {midi_note.shape}")
    print(f"  velocity: {velocity.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(audio, instrument_id, midi_note, velocity)

    print(f"\nOutput shape: {output.shape}")

    # Check output range
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\n✅ Model test passed!")
