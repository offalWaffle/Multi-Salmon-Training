"""
Adversarial pitch disentanglement module.

GradientReversalLayer + PitchClassifier for forcing the VQ-VAE encoder to
produce pitch-invariant (timbre-only) latent codes.

The classifier tries to predict pitch from z_q.  The GRL makes the encoder
receive sign-flipped gradients, so it learns to hide pitch from z_q.
Equilibrium: the classifier performs at chance; all pitch information is
routed through the decoder's FiLM conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin et al., 2015).

    Forward pass: identity.
    Backward pass: negate gradient, scaled by alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class PitchClassifier(nn.Module):
    """
    Small MLP that predicts MIDI note from a pooled z_q summary.

    Architecture: global avg pool → Linear(64,256) → ReLU → Linear(256,128)
                  → ReLU → Linear(128, num_midi_classes)

    ~50k parameters — negligible overhead.
    """

    def __init__(self, latent_dim=64, num_midi_classes=88, midi_offset=21):
        """
        Args:
            latent_dim: Channel dimension of z_q (matches VQ-VAE latent_dim).
            num_midi_classes: Number of pitch classes (88 for piano: MIDI 21–108).
            midi_offset: Lowest MIDI note in the range (21 = A0 for piano).
        """
        super().__init__()
        self.midi_offset = midi_offset
        self.num_midi_classes = num_midi_classes

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_midi_classes),
        )

    def forward(self, z_q, midi_note):
        """
        Args:
            z_q: [B, latent_dim, T] — quantized latent codes (possibly passed
                 through GradientReversal.apply before calling this).
            midi_note: [B] — raw MIDI note numbers (e.g. 21–108 for piano).

        Returns:
            loss: CrossEntropyLoss scalar.
            logits: [B, num_midi_classes] raw prediction logits.
        """
        # Global average pool over time — pitch is a global property
        pooled = z_q.mean(dim=-1)          # [B, latent_dim]
        logits = self.net(pooled)           # [B, num_midi_classes]

        class_idx = (midi_note - self.midi_offset).long().clamp(
            0, self.num_midi_classes - 1
        )
        loss = F.cross_entropy(logits, class_idx)
        return loss, logits
