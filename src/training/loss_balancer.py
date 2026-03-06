"""
Loss Balancer for automatic gradient balancing.

Based on EnCodec's loss balancer mechanism (Défossez et al., 2022).
Automatically balances gradient contributions from multiple losses,
eliminating the need for manual loss weight tuning.

Key idea: Each loss defines the target fraction of overall gradient it should represent.
The balancer automatically adjusts multipliers to achieve this, regardless of loss magnitudes.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import defaultdict


class LossBalancer:
    """
    Balances gradients from multiple losses automatically.

    Instead of manually tuning loss weights (alpha, beta, etc.), this balancer
    ensures each loss contributes a fixed fraction of the overall gradient magnitude.

    Example:
        If target_fractions = {'mel': 0.9, 'kl': 0.1}, then:
        - Mel loss will contribute 90% of the total gradient
        - KL loss will contribute 10% of the total gradient

    This is achieved by dynamically adjusting loss weights based on gradient magnitudes.
    """

    def __init__(
        self,
        target_fractions: Dict[str, float],
        ema_decay: float = 0.99,
        epsilon: float = 1e-8
    ):
        """
        Initialize loss balancer.

        Args:
            target_fractions: Dict mapping loss name to target gradient fraction
                             Example: {'mel': 0.9, 'kl': 0.1}
                             Fractions must sum to 1.0
            ema_decay: Decay factor for exponential moving average of gradient norms
                       Higher = more stable but slower to adapt (default: 0.99)
            epsilon: Small constant for numerical stability
        """
        # Validate fractions sum to 1.0
        total_fraction = sum(target_fractions.values())
        if not (0.99 <= total_fraction <= 1.01):  # Allow small numerical error
            raise ValueError(
                f"Target fractions must sum to 1.0, got {total_fraction}. "
                f"Fractions: {target_fractions}"
            )

        self.target_fractions = target_fractions
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # EMA of gradient norms for each loss
        self.grad_norm_ema = {}
        for name in target_fractions.keys():
            self.grad_norm_ema[name] = None

    def balance(
        self,
        losses: Dict[str, torch.Tensor],
        parameters: nn.Parameter,
        update_ema: bool = True
    ) -> torch.Tensor:
        """
        Compute balanced loss by adjusting weights based on gradient magnitudes.

        Args:
            losses: Dict mapping loss name to loss tensor
                   Example: {'mel': mel_loss, 'kl': kl_loss}
            parameters: Model parameters to compute gradients against
                       Typically a single representative parameter (e.g., first layer weight)
            update_ema: Whether to update EMA of gradient norms (set False during validation)

        Returns:
            Balanced total loss (single scalar tensor)
        """
        # Ensure all losses are in target_fractions
        for name in losses.keys():
            if name not in self.target_fractions:
                raise ValueError(
                    f"Loss '{name}' not in target_fractions. "
                    f"Available: {list(self.target_fractions.keys())}"
                )

        # Compute gradient norm for each loss
        grad_norms = {}
        for name, loss in losses.items():
            # Compute gradient of this loss w.r.t. parameters
            # retain_graph=True because we need to compute multiple gradients
            grads = torch.autograd.grad(
                loss, parameters,
                retain_graph=True,
                create_graph=True  # Allow second-order gradients
            )

            # Compute L2 norm of gradients
            grad_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
            grad_norms[name] = grad_norm

        # Update EMA of gradient norms
        if update_ema:
            for name, grad_norm in grad_norms.items():
                if self.grad_norm_ema[name] is None:
                    # First time: initialize with current value
                    self.grad_norm_ema[name] = grad_norm.item()
                else:
                    # Update EMA
                    self.grad_norm_ema[name] = (
                        self.ema_decay * self.grad_norm_ema[name] +
                        (1 - self.ema_decay) * grad_norm.item()
                    )

        # Compute loss weights to match target fractions
        # We want: weight[i] * grad_norm[i] / total_weighted_grad = target_fraction[i]
        # Solving for weights:
        #   weight[i] = target_fraction[i] * total_weighted_grad / grad_norm[i]
        #
        # But total_weighted_grad depends on weights, so we use EMA grad norms
        # to approximate the steady-state solution

        # Use EMA grad norms for weight calculation (more stable)
        ema_norms = {
            name: self.grad_norm_ema[name] if self.grad_norm_ema[name] is not None else 1.0
            for name in losses.keys()
        }

        # Calculate weights to achieve target fractions
        # Total gradient magnitude (sum of weighted norms)
        # We normalize by total target fraction to handle cases where not all losses are used
        total_target = sum(self.target_fractions[name] for name in losses.keys())

        weights = {}
        for name in losses.keys():
            # Weight = (target_fraction / grad_norm) * normalization
            # Normalization ensures weights are scaled appropriately
            weights[name] = (
                self.target_fractions[name] / (ema_norms[name] + self.epsilon)
            )

        # Normalize weights so they average to a reasonable scale
        # This prevents weights from becoming extremely large or small
        weight_sum = sum(weights.values())
        weight_norm = len(weights) / (weight_sum + self.epsilon)
        weights = {name: w * weight_norm for name, w in weights.items()}

        # Compute balanced loss
        balanced_loss = sum(weights[name] * losses[name] for name in losses.keys())

        return balanced_loss

    def get_weights(self) -> Dict[str, float]:
        """
        Get current loss weights based on EMA gradient norms.

        Returns:
            Dict mapping loss name to current weight
        """
        if any(v is None for v in self.grad_norm_ema.values()):
            # Not initialized yet
            return {name: 1.0 for name in self.target_fractions.keys()}

        # Calculate current weights (same logic as in balance())
        total_target = sum(self.target_fractions.values())
        weights = {}
        for name in self.target_fractions.keys():
            weights[name] = (
                self.target_fractions[name] /
                (self.grad_norm_ema[name] + self.epsilon)
            )

        # Normalize
        weight_sum = sum(weights.values())
        weight_norm = len(weights) / (weight_sum + self.epsilon)
        weights = {name: w * weight_norm for name, w in weights.items()}

        return weights

    def get_grad_norms(self) -> Dict[str, float]:
        """
        Get current EMA gradient norms.

        Returns:
            Dict mapping loss name to EMA gradient norm
        """
        return {
            name: norm if norm is not None else 0.0
            for name, norm in self.grad_norm_ema.items()
        }


class SimpleLossBalancer:
    """
    Simplified loss balancer that doesn't require gradient computation.

    Uses loss magnitudes instead of gradient norms as a proxy for balancing.
    Less accurate than full gradient-based balancer, but much faster.

    Useful for initial experiments or when compute budget is limited.
    """

    def __init__(
        self,
        target_fractions: Dict[str, float],
        ema_decay: float = 0.99,
        epsilon: float = 1e-8
    ):
        """Initialize simplified balancer."""
        # Validate fractions
        total_fraction = sum(target_fractions.values())
        if not (0.99 <= total_fraction <= 1.01):
            raise ValueError(
                f"Target fractions must sum to 1.0, got {total_fraction}"
            )

        self.target_fractions = target_fractions
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # EMA of loss magnitudes
        self.loss_mag_ema = {name: None for name in target_fractions.keys()}

    def balance(
        self,
        losses: Dict[str, torch.Tensor],
        update_ema: bool = True
    ) -> torch.Tensor:
        """
        Compute balanced loss using loss magnitudes as proxy for gradients.

        Args:
            losses: Dict mapping loss name to loss tensor
            update_ema: Whether to update EMA

        Returns:
            Balanced total loss
        """
        # Update EMA of loss magnitudes
        if update_ema:
            for name, loss in losses.items():
                loss_mag = loss.detach().item()
                if self.loss_mag_ema[name] is None:
                    self.loss_mag_ema[name] = loss_mag
                else:
                    self.loss_mag_ema[name] = (
                        self.ema_decay * self.loss_mag_ema[name] +
                        (1 - self.ema_decay) * loss_mag
                    )

        # Compute weights based on loss magnitudes
        ema_mags = {
            name: self.loss_mag_ema[name] if self.loss_mag_ema[name] is not None else 1.0
            for name in losses.keys()
        }

        weights = {}
        for name in losses.keys():
            weights[name] = (
                self.target_fractions[name] / (ema_mags[name] + self.epsilon)
            )

        # Normalize weights
        weight_sum = sum(weights.values())
        weight_norm = len(weights) / (weight_sum + self.epsilon)
        weights = {name: w * weight_norm for name, w in weights.items()}

        # Compute balanced loss
        balanced_loss = sum(weights[name] * losses[name] for name in losses.keys())

        return balanced_loss

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        if any(v is None for v in self.loss_mag_ema.values()):
            return {name: 1.0 for name in self.target_fractions.keys()}

        weights = {}
        for name in self.target_fractions.keys():
            weights[name] = (
                self.target_fractions[name] /
                (self.loss_mag_ema[name] + self.epsilon)
            )

        weight_sum = sum(weights.values())
        weight_norm = len(weights) / (weight_sum + self.epsilon)
        weights = {name: w * weight_norm for name, w in weights.items()}

        return weights
