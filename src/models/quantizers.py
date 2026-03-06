"""
Vector Quantization layers for VQ-VAE.

Implements discrete codebook learning with various quantization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer.

    Maps continuous latent codes to discrete codebook entries.
    Uses straight-through estimator for gradients.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Args:
            num_embeddings: Size of codebook (number of discrete codes)
            embedding_dim: Dimension of each code vector
            commitment_cost: Weight for commitment loss (encoder commits to codes)
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: learnable embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Initialize codebook with uniform distribution
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # Exponential moving average of per-code usage for dead-code reset.
        # Initialised to 1.0 so codes are not immediately considered dead on the
        # first batch; they decay below the reset threshold after ~70 unused batches
        # (0.99^70 ≈ 0.50, 0.99^100 ≈ 0.37 …  0.99^200 ≈ 0.13).
        self.register_buffer('ema_usage', torch.ones(num_embeddings))

    def forward(self, z):
        """
        Quantize latent codes to nearest codebook entries.

        Args:
            z: (batch, embedding_dim, *) continuous latent codes

        Returns:
            quantized: Quantized latent codes (same shape as z)
            loss: VQ loss (codebook + commitment)
            perplexity: Codebook usage metric
            encodings: One-hot encoding of selected codes
        """
        # Reshape z: (batch, dim, *) -> (batch, *, dim)
        z = z.movedim(1, -1)
        flat_z = z.reshape(-1, self.embedding_dim)  # (batch * ..., dim)

        # Calculate distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z · e
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)  # ||z||^2
            + torch.sum(self.embedding.weight ** 2, dim=1)  # ||e||^2
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())  # 2 * z · e
        )  # (batch * ..., num_embeddings)

        # Find nearest codebook entry for each latent
        encoding_indices = torch.argmin(distances, dim=1)  # (batch * ...,)

        # One-hot encoding
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        # (batch * ..., num_embeddings)

        # Quantize by looking up codebook entries
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        # (batch * ..., dim)

        # Reshape back
        quantized = quantized_flat.view_as(z)  # (batch, *, dim)
        quantized = quantized.movedim(-1, 1)  # (batch, dim, *)

        # Compute losses (following Oord et al. 2017 VQ-VAE paper).
        # Codebook loss: pulls codebook entries toward encoder outputs (weight 1.0).
        #   gradient flows through quantized (the codebook entry), z is stopped.
        codebook_loss = F.mse_loss(quantized, z.movedim(-1, 1).detach())

        # Commitment loss: pulls encoder outputs toward codebook (weight β).
        #   gradient flows through z (the encoder output), quantized is stopped.
        commitment_loss = F.mse_loss(quantized.detach(), z.movedim(-1, 1))

        # Total VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from quantized to z
        quantized = z.movedim(-1, 1) + (quantized - z.movedim(-1, 1)).detach()

        # Calculate perplexity (measure of codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Dead-code reset: codes whose EMA usage decays below 0.5 are replaced with a
        # random encoder output from the current batch.  This prevents the collapse
        # where a handful of codes absorb all assignments while the rest never move.
        if self.training:
            batch_usage = encodings.sum(0)                # (num_embeddings,)
            self.ema_usage.mul_(0.99).add_(batch_usage, alpha=0.01)

            dead = self.ema_usage < 0.5
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                # Sample random z vectors from the current batch as replacement values.
                # Using actual encoder outputs ensures the new codes land in the real
                # data distribution rather than at the random initialisation origin.
                rand_idx = torch.randint(flat_z.size(0), (n_dead,), device=flat_z.device)
                self.embedding.weight.data[dead] = flat_z[rand_idx].detach()
                # Warm-start EMA so the reset code isn't immediately reset again.
                self.ema_usage[dead] = 1.0

        return quantized, vq_loss, perplexity, encoding_indices.view(z.shape[0], -1)

    def quantize(self, encoding_indices):
        """
        Convert encoding indices back to quantized vectors.

        Args:
            encoding_indices: (batch, *) indices into codebook

        Returns:
            Quantized vectors
        """
        flat_indices = encoding_indices.reshape(-1)
        quantized_flat = self.embedding(flat_indices)

        # Reshape to include spatial dimensions
        quantized = quantized_flat.view(*encoding_indices.shape, self.embedding_dim)

        # Move channel dimension to position 1
        quantized = quantized.movedim(-1, 1)

        return quantized


class EMAVectorQuantizer(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.

    Updates codebook using EMA instead of gradient descent.
    More stable training, especially with large codebooks.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA buffers (not trained with gradient descent)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, z):
        """
        Quantize with EMA codebook updates.

        Args:
            z: (batch, embedding_dim, *) continuous latent codes

        Returns:
            quantized, loss, perplexity, encoding_indices
        """
        # Reshape z
        z = z.movedim(1, -1)
        flat_z = z.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        # Find nearest codes
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view_as(z).movedim(-1, 1)

        # EMA codebook update (only during training)
        if self.training:
            # Update cluster sizes
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                   (1 - self.decay) * torch.sum(encodings, dim=0)

            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )

            # Update embeddings
            dw = torch.matmul(encodings.t(), flat_z)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            # Normalize — clamp to ≥1 to avoid dividing by near-zero cluster
            # sizes for unused codes (which would amplify ema_w by ~200× → NaN)
            self.embedding.weight.data = self.ema_w / self.ema_cluster_size.clamp(min=1.0).unsqueeze(1)

        # Commitment loss only (codebook is updated via EMA, not gradients).
        # Pulls encoder outputs toward codebook — gradient flows through z, quantized is stopped.
        commitment_loss = F.mse_loss(quantized.detach(), z.movedim(-1, 1))
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z.movedim(-1, 1) + (quantized - z.movedim(-1, 1)).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity, encoding_indices.view(z.shape[0], -1)

    def quantize(self, encoding_indices):
        """Convert indices to quantized vectors."""
        flat_indices = encoding_indices.reshape(-1)
        quantized_flat = self.embedding(flat_indices)
        quantized = quantized_flat.view(*encoding_indices.shape, self.embedding_dim)
        quantized = quantized.movedim(-1, 1)
        return quantized


class GumbelVectorQuantizer(nn.Module):
    """
    Vector Quantizer using Gumbel-Softmax for differentiable sampling.

    Provides soft quantization during training and hard quantization during inference.
    """

    def __init__(self, num_embeddings, embedding_dim, temperature=1.0,
                 commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """Quantize using Gumbel-Softmax trick."""
        # Reshape
        z = z.movedim(1, -1)
        flat_z = z.reshape(-1, self.embedding_dim)

        # Calculate logits (negative distances)
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )
        logits = -distances  # Higher is better

        if self.training:
            # Gumbel-Softmax during training (differentiable)
            from torch.nn.functional import gumbel_softmax
            soft_encodings = gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            # Hard argmax during inference
            encoding_indices = torch.argmax(logits, dim=1)
            soft_encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize
        quantized_flat = torch.matmul(soft_encodings, self.embedding.weight)
        quantized = quantized_flat.view_as(z).movedim(-1, 1)

        # Losses
        codebook_loss = F.mse_loss(quantized.detach(), z.movedim(-1, 1))
        commitment_loss = F.mse_loss(quantized, z.movedim(-1, 1).detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through
        quantized = z.movedim(-1, 1) + (quantized - z.movedim(-1, 1)).detach()

        # Perplexity
        avg_probs = torch.mean(soft_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = torch.argmax(logits, dim=1)
        return quantized, vq_loss, perplexity, encoding_indices.view(z.shape[0], -1)

    def quantize(self, encoding_indices):
        """Convert indices to quantized vectors."""
        flat_indices = encoding_indices.reshape(-1)
        quantized_flat = self.embedding(flat_indices)
        quantized = quantized_flat.view(*encoding_indices.shape, self.embedding_dim)
        quantized = quantized.movedim(-1, 1)
        return quantized
