"""
DAC Pitch Adapter — pitch translation in frozen DAC latent space.

Architecture:
    audio → [frozen DAC encoder] → z_dac
        z_dac → [PitchStripper] → z_timbre  (stays near-identity; no adversarial)
        z_timbre + src_midi + tgt_midi → [PitchInjector FiLM] → z_modified
        z_modified → [frozen DAC decoder] → audio_out

The injector is conditioned on BOTH source and target MIDI notes because the
latent-space pitch transformation is not interval-invariant: a C2→C3 shift
looks nothing like a C6→C7 shift in DAC's representation.
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
    Residual network — stays near-identity (no adversarial pressure).
    Kept in the pipeline for architectural symmetry; effectively passes
    z_dac through unchanged.
    """

    def __init__(self, dac_latent_dim=1024, inner_dim=512, num_residual_blocks=6):
        super().__init__()
        self.proj_in  = nn.Conv1d(dac_latent_dim, inner_dim, 1)
        self.blocks   = nn.Sequential(
            *[ResBlock1d(inner_dim) for _ in range(num_residual_blocks)]
        )
        self.proj_out = nn.Conv1d(inner_dim, dac_latent_dim, 1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z_dac):
        x = self.proj_in(z_dac)
        x = self.blocks(x)
        pitch_delta = torch.tanh(self.proj_out(x)) * 1.0
        z_timbre = z_dac - pitch_delta
        return z_timbre, None   # None: x_inner no longer needed


# ---------------------------------------------------------------------------
# PitchInjector
# ---------------------------------------------------------------------------

class PitchProbe(nn.Module):
    """
    Lightweight pitch classifier on z_modified.

    Global-average-pool the latent → linear → 88 MIDI class logits.
    Provides a direct gradient signal that forces the injector to encode
    target pitch information (no DAC decoder needed).
    """
    def __init__(self, dac_latent_dim=1024, num_classes=88):
        super().__init__()
        self.linear = nn.Linear(dac_latent_dim, num_classes)

    def forward(self, z):
        # z: [B, dac_latent_dim, T]
        return self.linear(z.mean(-1))   # [B, num_classes]


class PitchInjector(nn.Module):
    """
    FiLM-conditioned pitch-translation network.

    Conditioned on BOTH src_midi and tgt_midi so the network can account for
    the register-dependent acoustic structure of the source latent.

    Uses exponentially increasing dilations [1,2,4,8,16,32] across blocks to
    give a receptive field of ~132/172 frames — wide enough to see the global
    pitch period structure of the 2 s window.

    z_modified = z_timbre + pitch_delta(z_timbre, src_midi, tgt_midi)
    Zero-init on proj_out → identity at initialisation.
    """

    # Dilation schedule: doubles each block, giving wide temporal context
    DILATIONS = [1, 2, 4, 8, 16, 32]

    def __init__(self, dac_latent_dim=1024, inner_dim=512,
                 num_residual_blocks=6, pitch_embed_dim=256):
        super().__init__()
        self.src_embed = SinusoidalPitchEmbedding(embed_dim=pitch_embed_dim)
        self.tgt_embed = SinusoidalPitchEmbedding(embed_dim=pitch_embed_dim)
        cond_dim = pitch_embed_dim * 2          # concatenated src + tgt embeddings
        self.proj_in  = nn.Conv1d(dac_latent_dim, inner_dim, 1)

        dilations = self.DILATIONS
        # Cycle the dilation list if num_residual_blocks > len(DILATIONS)
        dilations = [dilations[i % len(dilations)] for i in range(num_residual_blocks)]
        self.blocks = nn.ModuleList([
            FiLMResBlock(inner_dim, cond_dim=cond_dim, dilation=d)
            for d in dilations
        ])
        self.proj_out = nn.Conv1d(inner_dim, dac_latent_dim, 1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z_timbre, src_midi, tgt_midi):
        """
        Args:
            z_timbre:  [B, 1024, T]
            src_midi:  [B] — MIDI note of the source latent
            tgt_midi:  [B] — MIDI note to translate to

        Returns:
            z_modified: [B, 1024, T]
        """
        src_emb   = self.src_embed(src_midi)                    # [B, pitch_embed_dim]
        tgt_emb   = self.tgt_embed(tgt_midi)                    # [B, pitch_embed_dim]
        pitch_cond = torch.cat([src_emb, tgt_emb], dim=-1)     # [B, pitch_embed_dim*2]
        x = self.proj_in(z_timbre)
        for block in self.blocks:
            x = block(x, pitch_cond)
        pitch_delta = self.proj_out(x)      # unbounded — L1 loss bounds magnitude
        return z_timbre + pitch_delta


# ---------------------------------------------------------------------------
# DACPitchAdapter — top-level wrapper
# ---------------------------------------------------------------------------

class DACPitchAdapter(nn.Module):
    """
    Wraps frozen DAC encoder+decoder with trainable PitchStripper and PitchInjector.
    Only adapter parameters are trained; DAC remains frozen.
    """

    def __init__(self, dac_model, inner_dim=512, num_residual_blocks=6,
                 pitch_embed_dim=256, dac_latent_dim=1024):
        super().__init__()

        self.dac = dac_model
        for p in self.dac.parameters():
            p.requires_grad_(False)

        self.stripper = PitchStripper(dac_latent_dim, inner_dim, num_residual_blocks)
        self.injector = PitchInjector(dac_latent_dim, inner_dim, num_residual_blocks,
                                      pitch_embed_dim)
        self.probe    = PitchProbe(dac_latent_dim, num_classes=88)

    def forward(self, z_dac, src_midi, tgt_midi):
        z_timbre, _ = self.stripper(z_dac)
        z_modified  = self.injector(z_timbre, src_midi, tgt_midi)
        return z_modified, z_timbre

    @torch.no_grad()
    def transfer_pitch(self, audio, src_midi_note, target_midi_note):
        """
        One-shot inference: encode, translate pitch, decode.

        Args:
            audio:            [B, 1, samples] input audio (44.1 kHz)
            src_midi_note:    int or [B] — MIDI note of the source audio
            target_midi_note: int or [B] — desired output MIDI note

        Returns:
            audio_out: [B, 1, samples'] reconstructed audio at new pitch
        """
        self.eval()

        z_dac, _codes, _latents, _cl, _ql = self.dac.encode(audio)
        z_timbre, _ = self.stripper(z_dac)

        def _to_tensor(note, B, device):
            if isinstance(note, int):
                return torch.full((B,), note, dtype=torch.long, device=device)
            return note.to(device)

        B, device = audio.shape[0], audio.device
        src = _to_tensor(src_midi_note, B, device)
        tgt = _to_tensor(target_midi_note, B, device)

        z_modified = self.injector(z_timbre, src, tgt)
        return self.dac.decode(z_modified)

    def adapter_parameters(self):
        return (list(self.stripper.parameters()) +
                list(self.injector.parameters()) +
                list(self.probe.parameters()))
