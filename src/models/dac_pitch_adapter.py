"""
DAC Pitch Adapter — train two small networks inside DAC's frozen latent space.

Architecture:
    audio → [frozen DAC encoder] → z_dac
        z_dac → [PitchStripper] → z_timbre  (adversarially forced to be pitch-free)
        z_timbre + target_pitch → [PitchInjector FiLM] → z_modified
        z_modified → [frozen DAC decoder] → audio_out

Both adapters are residual with zero-init output → identity at initialisation,
so epoch-0 reconstruction equals DAC's native quality.
"""

import torch
import torch.nn as nn

from .film_layers import FiLMResBlock
from .pitch_conditioning import SinusoidalPitchEmbedding


# ---------------------------------------------------------------------------
# Plain residual block (no conditioning) — used by PitchStripper
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.act2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        return self.act2(out + residual)


# ---------------------------------------------------------------------------
# PitchStripper
# ---------------------------------------------------------------------------

class PitchStripper(nn.Module):
    """
    Residual pitch-removal network.

    z_timbre = z_dac − pitch_delta,  where pitch_delta is predicted from z_dac.
    Zero-init on proj_out ensures the network starts as the identity.

    x_inner (output of proj_in, before residual blocks) is returned so the
    caller can attach GRL + PitchClassifier there.
    """

    def __init__(self, dac_latent_dim=1024, inner_dim=256, num_residual_blocks=3):
        super().__init__()
        self.proj_in = nn.Conv1d(dac_latent_dim, inner_dim, 1)
        self.blocks = nn.Sequential(
            *[ResBlock1d(inner_dim) for _ in range(num_residual_blocks)]
        )
        self.proj_out = nn.Conv1d(inner_dim, dac_latent_dim, 1)

        # Zero-init: at t=0, proj_out(x) = 0 → tanh(0)*0.1 = 0 → z_timbre = z_dac
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z_dac):
        """
        Args:
            z_dac: [B, 1024, T]

        Returns:
            z_timbre: [B, 1024, T]
            x_inner:  [B, inner_dim, T]  — hook point for GRL + classifier
        """
        x_inner = self.proj_in(z_dac)          # [B, 256, T]
        x = self.blocks(x_inner)               # [B, 256, T]
        pitch_delta = torch.tanh(self.proj_out(x)) * 0.1
        z_timbre = z_dac - pitch_delta
        return z_timbre, x_inner


# ---------------------------------------------------------------------------
# PitchInjector
# ---------------------------------------------------------------------------

class PitchInjector(nn.Module):
    """
    FiLM-conditioned pitch-injection network.

    z_modified = z_timbre + pitch_delta,  where pitch_delta is predicted from
    z_timbre conditioned on target_midi.
    Zero-init on proj_out ensures the network starts as the identity.
    """

    def __init__(self, dac_latent_dim=1024, inner_dim=256,
                 num_residual_blocks=3, pitch_embed_dim=128):
        super().__init__()
        self.pitch_embed = SinusoidalPitchEmbedding(embed_dim=pitch_embed_dim)
        self.proj_in = nn.Conv1d(dac_latent_dim, inner_dim, 1)
        self.blocks = nn.ModuleList([
            FiLMResBlock(inner_dim, cond_dim=pitch_embed_dim)
            for _ in range(num_residual_blocks)
        ])
        self.proj_out = nn.Conv1d(inner_dim, dac_latent_dim, 1)

        # Zero-init → identity at t=0
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z_timbre, target_midi):
        """
        Args:
            z_timbre:    [B, 1024, T]
            target_midi: [B] MIDI note numbers

        Returns:
            z_modified: [B, 1024, T]
        """
        pitch_emb = self.pitch_embed(target_midi)   # [B, pitch_embed_dim]
        x = self.proj_in(z_timbre)                  # [B, 256, T]
        for block in self.blocks:
            x = block(x, pitch_emb)
        pitch_delta = torch.tanh(self.proj_out(x)) * 0.1
        z_modified = z_timbre + pitch_delta
        return z_modified


# ---------------------------------------------------------------------------
# DACPitchAdapter — top-level wrapper
# ---------------------------------------------------------------------------

class DACPitchAdapter(nn.Module):
    """
    Wraps frozen DAC encoder+decoder with trainable PitchStripper and PitchInjector.

    Only PitchStripper and PitchInjector parameters are trained.
    DAC remains completely frozen.
    """

    def __init__(self, dac_model, inner_dim=256, num_residual_blocks=3,
                 pitch_embed_dim=128, dac_latent_dim=1024):
        super().__init__()

        # Freeze DAC
        self.dac = dac_model
        for p in self.dac.parameters():
            p.requires_grad_(False)

        self.stripper = PitchStripper(dac_latent_dim, inner_dim, num_residual_blocks)
        self.injector = PitchInjector(dac_latent_dim, inner_dim, num_residual_blocks,
                                      pitch_embed_dim)

    def strip_pitch(self, z_dac):
        """
        Args:
            z_dac: [B, 1024, T]

        Returns:
            z_timbre: [B, 1024, T]
            x_inner:  [B, inner_dim, T]
        """
        return self.stripper(z_dac)

    def inject_pitch(self, z_timbre, target_midi):
        """
        Args:
            z_timbre:    [B, 1024, T]
            target_midi: [B]

        Returns:
            z_modified: [B, 1024, T]
        """
        return self.injector(z_timbre, target_midi)

    def forward(self, z_dac, src_midi, tgt_midi):
        """
        Full adapter pipeline (training forward pass).

        Strips src pitch, injects tgt pitch.  For reconstruction training,
        call with src_midi == tgt_midi.

        Args:
            z_dac:    [B, 1024, T]
            src_midi: [B]  (unused in current residual design but kept for API symmetry)
            tgt_midi: [B]

        Returns:
            z_modified: [B, 1024, T]
            x_inner:    [B, inner_dim, T]  — for GRL + adversarial loss
            z_timbre:   [B, 1024, T]
        """
        z_timbre, x_inner = self.stripper(z_dac)
        z_modified = self.injector(z_timbre, tgt_midi)
        return z_modified, x_inner, z_timbre

    @torch.no_grad()
    def transfer_pitch(self, audio, target_midi_note):
        """
        One-shot inference: encode, strip, inject new pitch, decode.

        Args:
            audio:            [B, 1, samples] input audio (44.1 kHz)
            target_midi_note: int or [B] tensor — MIDI note(s) for output

        Returns:
            audio_out: [B, 1, samples'] reconstructed audio at new pitch
        """
        self.eval()

        # Encode with frozen DAC
        z_dac, _codes, _latents, _cl, _ql = self.dac.encode(audio)

        # Strip pitch
        z_timbre, _ = self.stripper(z_dac)

        # Build target midi tensor
        if isinstance(target_midi_note, int):
            tgt = torch.full((audio.shape[0],), target_midi_note,
                             dtype=torch.long, device=audio.device)
        else:
            tgt = target_midi_note.to(audio.device)

        # Inject new pitch
        z_modified = self.injector(z_timbre, tgt)

        # Decode with frozen DAC
        audio_out = self.dac.decode(z_modified)
        return audio_out

    def adapter_parameters(self):
        """Return only the trainable adapter parameters (for main optimizer)."""
        return list(self.stripper.parameters()) + list(self.injector.parameters())
