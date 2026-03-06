"""
Pitch conditioning modules for VQ-VAE.

Provides embeddings and utilities for pitch-conditioned generation.
"""

import torch
import torch.nn as nn
import math


class PitchEmbedding(nn.Module):
    """
    Learned embeddings for MIDI pitch values.

    Maps MIDI notes (0-127) to continuous embedding vectors.
    """

    def __init__(self, embed_dim=128, num_pitches=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_pitches = num_pitches

        # Learned embedding table
        self.embedding = nn.Embedding(num_pitches, embed_dim)

    def forward(self, midi_notes):
        """
        Args:
            midi_notes: (batch_size,) tensor of MIDI note numbers

        Returns:
            (batch_size, embed_dim) pitch embeddings
        """
        return self.embedding(midi_notes)


class SinusoidalPitchEmbedding(nn.Module):
    """
    Sinusoidal pitch embeddings (similar to positional encodings).

    Provides continuous pitch representations that generalize better
    to unseen pitches and pitch interpolation.
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, midi_notes):
        """
        Args:
            midi_notes: (batch_size,) tensor of MIDI note numbers

        Returns:
            (batch_size, embed_dim) pitch embeddings
        """
        device = midi_notes.device
        batch_size = midi_notes.shape[0]

        # Normalize MIDI to [0, 1] range
        normalized = midi_notes.float() / 127.0

        # Create frequency bands
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)

        # Apply sinusoidal encoding
        emb = normalized.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # Handle odd embedding dimensions
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(batch_size, 1, device=device)], dim=1)

        return emb


class PitchDeltaEmbedding(nn.Module):
    """
    Embedding for pitch deltas (for transformations).

    Used by LatentTransformer to specify target pitch relative to source.
    """

    def __init__(self, embed_dim=128, max_delta=48):
        """
        Args:
            embed_dim: Embedding dimension
            max_delta: Maximum pitch shift in semitones (±max_delta)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_delta = max_delta

        # Embedding for delta values: -max_delta to +max_delta
        num_deltas = 2 * max_delta + 1
        self.embedding = nn.Embedding(num_deltas, embed_dim)

        # Offset to map [-max_delta, +max_delta] to [0, num_deltas-1]
        self.offset = max_delta

    def forward(self, pitch_deltas):
        """
        Args:
            pitch_deltas: (batch_size,) tensor of pitch shifts in semitones

        Returns:
            (batch_size, embed_dim) delta embeddings
        """
        # Clamp to valid range
        pitch_deltas = torch.clamp(pitch_deltas, -self.max_delta, self.max_delta)

        # Shift to positive indices
        indices = pitch_deltas + self.offset

        return self.embedding(indices)


class HybridPitchEmbedding(nn.Module):
    """
    Combines learned and sinusoidal pitch embeddings.

    Provides both precise learned representations and continuous
    generalization capabilities.
    """

    def __init__(self, embed_dim=128, num_pitches=128):
        super().__init__()
        self.embed_dim = embed_dim

        # Half learned, half sinusoidal
        half_dim = embed_dim // 2

        self.learned = PitchEmbedding(half_dim, num_pitches)
        self.sinusoidal = SinusoidalPitchEmbedding(embed_dim - half_dim)

    def forward(self, midi_notes):
        """
        Args:
            midi_notes: (batch_size,) tensor of MIDI note numbers

        Returns:
            (batch_size, embed_dim) hybrid embeddings
        """
        learned_emb = self.learned(midi_notes)
        sinusoidal_emb = self.sinusoidal(midi_notes)

        return torch.cat([learned_emb, sinusoidal_emb], dim=1)


def midi_to_frequency(midi_notes):
    """
    Convert MIDI note numbers to frequencies in Hz.

    Args:
        midi_notes: Tensor of MIDI note numbers

    Returns:
        Frequencies in Hz
    """
    # A4 (MIDI 69) = 440 Hz
    return 440.0 * (2.0 ** ((midi_notes - 69) / 12.0))


def frequency_to_midi(frequencies):
    """
    Convert frequencies in Hz to MIDI note numbers.

    Args:
        frequencies: Tensor of frequencies in Hz

    Returns:
        MIDI note numbers (float)
    """
    return 69 + 12 * torch.log2(frequencies / 440.0)
