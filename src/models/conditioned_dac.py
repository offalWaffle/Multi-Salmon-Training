"""
Conditioned DAC (Descript Audio Codec) with MIDI note and velocity control.

This module provides wrappers around pretrained DAC models that add conditioning
for MIDI note and velocity, enabling controllable audio generation.

Two approaches are provided:
1. ConditionedDAC: Simple feature injection (recommended for initial testing)
2. FiLMConditionedDAC: Feature-wise Linear Modulation (more powerful)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import dac


class ConditionedDAC(nn.Module):
    """
    DAC with MIDI note and velocity conditioning via feature injection.

    This is the simpler approach that adds conditioning embeddings directly
    to the latent codes before decoding.

    Args:
        dac_model: Pretrained DAC model
        latent_dim: Dimension of DAC latent space (default: 1024)
        embed_dim: Dimension of MIDI and velocity embeddings (default: 256)
    """

    def __init__(
        self,
        dac_model: dac.DAC,
        latent_dim: int = 1024,
        embed_dim: int = 256
    ):
        super().__init__()
        self.dac = dac_model
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Conditioning embeddings
        # MIDI notes: 0-127 (128 possible values)
        self.midi_embed = nn.Embedding(128, embed_dim)
        # Velocity: 0-127 (128 possible values)
        self.velocity_embed = nn.Embedding(128, embed_dim)

        # Conditioning projection (map to latent space dimension)
        # Concatenate MIDI + velocity embeddings (2 * embed_dim) -> latent_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Initialize embeddings with normal distribution
        nn.init.normal_(self.midi_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.velocity_embed.weight, mean=0.0, std=0.02)

    @property
    def sample_rate(self):
        """Pass through DAC sample rate."""
        return self.dac.sample_rate

    def preprocess(self, audio_data, sample_rate):
        """Pass through to DAC preprocess."""
        return self.dac.preprocess(audio_data, sample_rate)

    def encode(self, audio_data):
        """
        Encode audio to latent codes using DAC encoder.

        Args:
            audio_data: Audio tensor [batch, 1, samples]

        Returns:
            z: Continuous latent codes [batch, latent_dim, time]
            codes: Discrete VQ codes [batch, num_codebooks, time]
            latents: Quantizer latents [batch, latent_dim, time]
            commitment_loss: VQ commitment loss
            codebook_loss: VQ codebook loss
        """
        return self.dac.encode(audio_data)

    def decode(
        self,
        z: torch.Tensor,
        midi_note: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent codes to audio with optional conditioning.

        Args:
            z: Latent codes [batch, latent_dim, time]
            midi_note: MIDI note numbers [batch] (0-127), optional
            velocity: Velocity values [batch] (0-127), optional

        Returns:
            audio: Reconstructed audio [batch, 1, samples]
        """
        # If no conditioning provided, use default (middle C at medium velocity)
        if midi_note is None:
            batch_size = z.shape[0]
            midi_note = torch.full((batch_size,), 60, dtype=torch.long, device=z.device)
        if velocity is None:
            batch_size = z.shape[0]
            velocity = torch.full((batch_size,), 80, dtype=torch.long, device=z.device)

        # Get conditioning embeddings
        midi_emb = self.midi_embed(midi_note)  # [batch, embed_dim]
        vel_emb = self.velocity_embed(velocity)  # [batch, embed_dim]

        # Concatenate embeddings
        cond = torch.cat([midi_emb, vel_emb], dim=-1)  # [batch, 2*embed_dim]

        # Project to latent space dimension
        cond_latent = self.cond_proj(cond)  # [batch, latent_dim]

        # Broadcast conditioning across time dimension and add to latent codes
        # z: [batch, latent_dim, time]
        # cond_latent: [batch, latent_dim, 1] (after unsqueeze)
        z_cond = z + cond_latent.unsqueeze(-1)  # [batch, latent_dim, time]

        # Decode with DAC decoder
        return self.dac.decode(z_cond)

    def forward(
        self,
        audio_data: torch.Tensor,
        midi_note: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full forward pass: encode -> decode with conditioning.

        Args:
            audio_data: Input audio [batch, 1, samples]
            midi_note: MIDI note numbers [batch] (0-127), optional
            velocity: Velocity values [batch] (0-127), optional

        Returns:
            recon: Reconstructed audio [batch, 1, samples]
            info: Dict with encoding information (z, codes, losses)
        """
        # Encode
        z, codes, latents, commitment_loss, codebook_loss = self.encode(audio_data)

        # Decode with conditioning
        recon = self.decode(z, midi_note, velocity)

        info = {
            'z': z,
            'codes': codes,
            'latents': latents,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss
        }

        return recon, info


class FiLMConditionedDAC(nn.Module):
    """
    DAC with MIDI note and velocity conditioning via FiLM (Feature-wise Linear Modulation).

    FiLM applies learned scale and shift parameters to the latent codes based on
    the conditioning information. This can be more powerful than simple addition.

    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)

    Args:
        dac_model: Pretrained DAC model
        latent_dim: Dimension of DAC latent space (default: 1024)
        hidden_dim: Hidden dimension for conditioning network (default: 512)
    """

    def __init__(
        self,
        dac_model: dac.DAC,
        latent_dim: int = 1024,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.dac = dac_model
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Conditioning network
        # Takes normalized MIDI note and velocity as input
        self.cond_net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2 inputs: normalized MIDI and velocity
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM layers: generate scale (gamma) and shift (beta) parameters
        self.film_scale = nn.Linear(hidden_dim, latent_dim)
        self.film_shift = nn.Linear(hidden_dim, latent_dim)

        # Initialize FiLM layers
        # Scale initialized near 1.0 (slight modulation)
        nn.init.normal_(self.film_scale.weight, mean=0.0, std=0.01)
        nn.init.ones_(self.film_scale.bias)

        # Shift initialized near 0.0 (no shift initially)
        nn.init.normal_(self.film_shift.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.film_shift.bias)

    @property
    def sample_rate(self):
        """Pass through DAC sample rate."""
        return self.dac.sample_rate

    def preprocess(self, audio_data, sample_rate):
        """Pass through to DAC preprocess."""
        return self.dac.preprocess(audio_data, sample_rate)

    def encode(self, audio_data):
        """
        Encode audio to latent codes using DAC encoder.

        Args:
            audio_data: Audio tensor [batch, 1, samples]

        Returns:
            z: Continuous latent codes [batch, latent_dim, time]
            codes: Discrete VQ codes [batch, num_codebooks, time]
            latents: Quantizer latents [batch, latent_dim, time]
            commitment_loss: VQ commitment loss
            codebook_loss: VQ codebook loss
        """
        return self.dac.encode(audio_data)

    def decode(
        self,
        z: torch.Tensor,
        midi_note: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent codes to audio with FiLM conditioning.

        Args:
            z: Latent codes [batch, latent_dim, time]
            midi_note: MIDI note numbers [batch] (0-127), optional
            velocity: Velocity values [batch] (0-127), optional

        Returns:
            audio: Reconstructed audio [batch, 1, samples]
        """
        batch_size = z.shape[0]

        # If no conditioning provided, use default (middle C at medium velocity)
        if midi_note is None:
            midi_note = torch.full((batch_size,), 60, dtype=torch.long, device=z.device)
        if velocity is None:
            velocity = torch.full((batch_size,), 80, dtype=torch.long, device=z.device)

        # Normalize conditioning to [0, 1]
        midi_norm = midi_note.float() / 127.0  # [batch]
        vel_norm = velocity.float() / 127.0    # [batch]

        # Stack as input to conditioning network
        cond_input = torch.stack([midi_norm, vel_norm], dim=-1)  # [batch, 2]

        # Get conditioning features
        cond_features = self.cond_net(cond_input)  # [batch, hidden_dim]

        # Generate FiLM parameters (scale and shift)
        gamma = self.film_scale(cond_features)  # [batch, latent_dim]
        beta = self.film_shift(cond_features)   # [batch, latent_dim]

        # Apply FiLM to latent codes
        # z: [batch, latent_dim, time]
        # gamma, beta: [batch, latent_dim, 1] (after unsqueeze)
        z_cond = z * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

        # Decode with DAC decoder
        return self.dac.decode(z_cond)

    def forward(
        self,
        audio_data: torch.Tensor,
        midi_note: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full forward pass: encode -> decode with FiLM conditioning.

        Args:
            audio_data: Input audio [batch, 1, samples]
            midi_note: MIDI note numbers [batch] (0-127), optional
            velocity: Velocity values [batch] (0-127), optional

        Returns:
            recon: Reconstructed audio [batch, 1, samples]
            info: Dict with encoding information (z, codes, losses)
        """
        # Encode
        z, codes, latents, commitment_loss, codebook_loss = self.encode(audio_data)

        # Decode with FiLM conditioning
        recon = self.decode(z, midi_note, velocity)

        info = {
            'z': z,
            'codes': codes,
            'latents': latents,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss
        }

        return recon, info


def load_pretrained_dac(model_type: str = "44khz", device: str = "cpu") -> dac.DAC:
    """
    Load pretrained DAC model.

    Args:
        model_type: Model type ("44khz", "24khz", "16khz")
        device: Device to load model on ("cpu", "cuda", "mps")

    Returns:
        Pretrained DAC model
    """
    model_path = dac.utils.download(model_type=model_type)
    model = dac.DAC.load(model_path)
    model.to(device)
    model.eval()
    return model


def create_conditioned_dac(
    conditioning_type: str = "feature_injection",
    model_type: str = "44khz",
    device: str = "cpu",
    **kwargs
) -> nn.Module:
    """
    Factory function to create a conditioned DAC model.

    Args:
        conditioning_type: Type of conditioning ("feature_injection" or "film")
        model_type: DAC model type ("44khz", "24khz", "16khz")
        device: Device to load model on
        **kwargs: Additional arguments for conditioning model

    Returns:
        Conditioned DAC model
    """
    # Load pretrained DAC
    dac_model = load_pretrained_dac(model_type=model_type, device=device)

    # Create conditioned wrapper
    if conditioning_type == "feature_injection":
        model = ConditionedDAC(dac_model, **kwargs)
    elif conditioning_type == "film":
        model = FiLMConditionedDAC(dac_model, **kwargs)
    else:
        raise ValueError(
            f"Unknown conditioning_type: {conditioning_type}. "
            f"Choose 'feature_injection' or 'film'."
        )

    model.to(device)
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing ConditionedDAC...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model
    model = create_conditioned_dac(
        conditioning_type="feature_injection",
        device=device
    )

    # Create dummy input
    batch_size = 2
    audio_length = 4 * 44100  # 4 seconds at 44.1kHz
    dummy_audio = torch.randn(batch_size, 1, audio_length).to(device)
    dummy_midi = torch.tensor([60, 72], dtype=torch.long).to(device)  # C4, C5
    dummy_velocity = torch.tensor([64, 96], dtype=torch.long).to(device)

    # Forward pass
    print("\nForward pass...")
    with torch.no_grad():
        recon, info = model(dummy_audio, dummy_midi, dummy_velocity)

    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape: {recon.shape}")
    print(f"Latent shape: {info['z'].shape}")
    print(f"Codes shape: {info['codes'].shape}")

    print("\n✓ ConditionedDAC test passed!")
