"""
Disentanglement losses for VQ-VAE training.

These losses encourage the encoder to learn pitch-invariant timbre codes,
separating pitch information from timbre/instrument identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning pitch-invariant representations.

    Encourages samples from the same instrument to have similar latent codes
    regardless of pitch, while pushing different instruments apart.
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Temperature parameter for softmax (lower = harder contrasts)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z, instrument_ids, pitch):
        """
        Compute contrastive loss.

        Args:
            z: (batch, latent_dim, time) latent codes from encoder
            instrument_ids: (batch,) instrument identifiers
            pitch: (batch,) MIDI note numbers (for negative sampling)

        Returns:
            Contrastive loss value
        """
        batch_size = z.shape[0]

        # Pool latent codes to single vector per sample
        # Global average pooling over time dimension
        z_pooled = z.mean(dim=2)  # (batch, latent_dim)

        # L2 normalize for cosine similarity
        z_norm = F.normalize(z_pooled, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(z_norm, z_norm.t()) / self.temperature
        # (batch, batch)

        # Create positive pairs mask (same instrument, different or same pitch)
        # We want to pull together samples of the same instrument
        instrument_ids = instrument_ids.unsqueeze(0)  # (1, batch)
        positive_mask = (instrument_ids == instrument_ids.t()).float()
        # (batch, batch)

        # Remove self-similarity (diagonal)
        positive_mask = positive_mask * (1 - torch.eye(batch_size, device=z.device))

        # Create negative pairs mask (different instruments)
        negative_mask = 1 - positive_mask - torch.eye(batch_size, device=z.device)

        # InfoNCE loss
        # For each anchor, compute loss against positives and negatives
        loss = 0.0
        num_positives = 0

        for i in range(batch_size):
            # Get positives for this anchor
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0:
                continue  # No positives for this anchor

            # Numerator: similarity to positives
            pos_sim = similarity[i, pos_indices]

            # Denominator: similarity to all (excluding self)
            mask = 1 - torch.eye(batch_size, device=z.device)[i]
            all_sim = similarity[i][mask.bool()]

            # Log-sum-exp for numerical stability
            denominator = torch.logsumexp(all_sim, dim=0)

            # Loss for each positive
            for pos_sim_i in pos_sim:
                loss += -pos_sim_i + denominator
                num_positives += 1

        # Average over all positive pairs
        if num_positives > 0:
            loss = loss / num_positives
        else:
            loss = torch.tensor(0.0, device=z.device)

        return loss


class InstrumentConsistencyLoss(nn.Module):
    """
    Instrument consistency loss.

    Encourages the quantized codes for the same instrument at different pitches
    to use similar codebook entries.
    """

    def __init__(self):
        super().__init__()

    def forward(self, encoding_indices, instrument_ids):
        """
        Compute instrument consistency loss.

        Args:
            encoding_indices: (batch, code_length) discrete code indices
            instrument_ids: (batch,) instrument identifiers

        Returns:
            Consistency loss value
        """
        batch_size = encoding_indices.shape[0]

        # For each unique instrument, compute variance of code usage
        unique_instruments = torch.unique(instrument_ids)

        total_variance = 0.0
        num_instruments = 0

        for inst_id in unique_instruments:
            # Get all samples of this instrument
            mask = (instrument_ids == inst_id)
            inst_indices = encoding_indices[mask]  # (n_samples, code_length)

            if inst_indices.shape[0] < 2:
                continue  # Need at least 2 samples to compute variance

            # Compute code distribution for this instrument
            # Flatten all codes from this instrument
            flat_codes = inst_indices.reshape(-1)  # (n_samples * code_length,)

            # Count code frequency
            code_counts = torch.bincount(flat_codes, minlength=encoding_indices.max() + 1)
            code_probs = code_counts.float() / code_counts.sum()

            # Entropy of code distribution (higher = more diverse codes)
            # We want LOW entropy (consistent codes per instrument)
            entropy = -torch.sum(code_probs * torch.log(code_probs + 1e-10))

            total_variance += entropy
            num_instruments += 1

        # Average entropy across instruments
        if num_instruments > 0:
            loss = total_variance / num_instruments
        else:
            loss = torch.tensor(0.0, device=encoding_indices.device)

        return loss


class PitchInvarianceLoss(nn.Module):
    """
    Pitch invariance loss.

    Explicitly encourages the encoder to produce similar latents for
    the same instrument at different pitches.

    Requires paired samples (same instrument, different pitches).
    """

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2, instrument_ids1, instrument_ids2):
        """
        Compute pitch invariance loss for paired samples.

        Args:
            z1: (batch, latent_dim, time) latents for pitch 1
            z2: (batch, latent_dim, time) latents for pitch 2
            instrument_ids1: (batch,) instrument IDs for z1
            instrument_ids2: (batch,) instrument IDs for z2

        Returns:
            Invariance loss value
        """
        # Only compute loss for pairs with same instrument ID
        same_instrument = (instrument_ids1 == instrument_ids2)

        if not same_instrument.any():
            return torch.tensor(0.0, device=z1.device)

        # Pool latents
        z1_pooled = z1.mean(dim=2)  # (batch, latent_dim)
        z2_pooled = z2.mean(dim=2)

        # L2 distance between paired latents (should be small)
        distances = torch.norm(z1_pooled - z2_pooled, p=2, dim=1)  # (batch,)

        # Average distance for same-instrument pairs
        loss = distances[same_instrument].mean()

        return loss


class DisentanglementLoss(nn.Module):
    """
    Combined disentanglement loss.

    Combines multiple losses to encourage pitch-invariant timbre learning.
    """

    def __init__(self, contrastive_weight=1.0, consistency_weight=0.5,
                 temperature=0.07):
        """
        Args:
            contrastive_weight: Weight for contrastive loss
            consistency_weight: Weight for consistency loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight

        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.consistency_loss = InstrumentConsistencyLoss()

    def forward(self, z, encoding_indices, instrument_ids, pitch):
        """
        Compute combined disentanglement loss.

        Args:
            z: (batch, latent_dim, time) encoder latents
            encoding_indices: (batch, code_length) quantized code indices
            instrument_ids: (batch,) instrument identifiers
            pitch: (batch,) MIDI note numbers

        Returns:
            Dictionary with individual losses and total loss
        """
        # Contrastive loss
        contrastive = self.contrastive_loss(z, instrument_ids, pitch)

        # Consistency loss
        consistency = self.consistency_loss(encoding_indices, instrument_ids)

        # Total loss
        total_loss = (
            self.contrastive_weight * contrastive +
            self.consistency_weight * consistency
        )

        return {
            'loss': total_loss,
            'contrastive_loss': contrastive.item() if isinstance(contrastive, torch.Tensor) else contrastive,
            'consistency_loss': consistency.item() if isinstance(consistency, torch.Tensor) else consistency,
        }


class CodebookUsageLoss(nn.Module):
    """
    Codebook usage loss.

    Encourages balanced usage of the codebook to avoid "dead codes"
    that are never used.
    """

    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, encoding_indices):
        """
        Compute codebook usage loss.

        Args:
            encoding_indices: (batch, code_length) discrete code indices

        Returns:
            Usage loss (higher when codes are imbalanced)
        """
        # Flatten all codes
        flat_codes = encoding_indices.reshape(-1)

        # Count usage of each code
        code_counts = torch.bincount(flat_codes, minlength=self.num_embeddings)
        code_probs = code_counts.float() / code_counts.sum()

        # Maximum entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(self.num_embeddings, dtype=torch.float32,
                                            device=encoding_indices.device))

        # Actual entropy
        entropy = -torch.sum(code_probs * torch.log(code_probs + 1e-10))

        # Loss: encourage high entropy (balanced usage)
        # Return negative entropy so higher is worse
        loss = max_entropy - entropy

        return loss
