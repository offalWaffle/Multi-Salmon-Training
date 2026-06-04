"""
VQ-VAE Trainer.

Handles training loop, loss computation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import yaml
import json
from datetime import datetime

from ..models.vqvae import VQVAE
from ..models.pitch_adversary import GradientReversal, PitchClassifier
from ..losses.spectral_loss import AudioReconstructionLoss
from ..losses.disentanglement_loss import DisentanglementLoss


class VQVAETrainer:
    """
    Trainer for VQ-VAE model.

    Combines all loss functions and handles training/validation loops.
    """

    def __init__(self, config, device='cuda'):
        """
        Args:
            config: Configuration dictionary
            device: Device to train on ('cuda', 'mps', or 'cpu')
        """
        self.config = config
        self.device = device

        # Create model
        self.model = VQVAE(config['model']).to(device)

        # Loss functions
        self.reconstruction_loss = AudioReconstructionLoss(
            sample_rate=config['audio']['sample_rate'],
            time_weight=1.0,
            stft_weight=1.0,
            mel_weight=1.0
        ).to(device)

        self.disentanglement_loss = DisentanglementLoss(
            contrastive_weight=1.0,
            consistency_weight=0.5,
            temperature=0.07
        ).to(device)

        # Loss weights
        self.recon_weight = config['training']['reconstruction_weight']
        self.vq_weight = config['training']['vq_weight']
        self.spectral_weight = config['training']['spectral_weight']
        self.perceptual_weight = config['training'].get('perceptual_weight', 0.1)

        # Adversarial pitch disentanglement
        adv_cfg = config.get('adversarial', {})
        self.adv_enabled = adv_cfg.get('enabled', False)
        self.adv_lambda = adv_cfg.get('lambda', 0.1)
        self.adv_warmup_epochs = adv_cfg.get('alpha_warmup_epochs', 10)

        if self.adv_enabled:
            latent_dim = config['model']['latent_dim']
            self.pitch_classifier = PitchClassifier(
                latent_dim=latent_dim,
                num_midi_classes=88,
                midi_offset=21,
            ).to(device)
            self.classifier_optimizer = optim.Adam(
                self.pitch_classifier.parameters(),
                lr=adv_cfg.get('classifier_lr', 1e-3),
            )
        else:
            self.pitch_classifier = None
            self.classifier_optimizer = None

        # Optimizer
        lr = config['training']['learning_rate']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=config['training']['betas'],
            weight_decay=config['training']['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/vqvae'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs/vqvae'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = config['logging']['log_every_n_steps']

        # Gradient clipping
        self.gradient_clip = config['training'].get('gradient_clip', 1.0)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        schedule_type = self.config['training'].get('lr_schedule', 'cosine')

        if schedule_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=self.config['training'].get('min_lr', 1e-6)
            )
        elif schedule_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        else:
            return None

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Dictionary of average losses
        """
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_disentangle_loss = 0.0
        total_adv_loss = 0.0
        total_perplexity = 0.0

        # Alpha ramp: 0 → 1 over the first adv_warmup_epochs
        adv_alpha = min(1.0, self.epoch / max(1, self.adv_warmup_epochs))

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio = batch['audio'].to(self.device)
            midi_note = batch['midi_note'].to(self.device)
            instrument_id = batch['instrument_id'].to(self.device)

            # Guard: skip batches where audio data itself is corrupt
            if not torch.isfinite(audio).all():
                print(f"\n[skip] NaN/Inf in audio at batch {batch_idx}")
                continue

            # Forward pass — z (pre-quant) and z_q (post-quant) are both returned
            reconstruction, vq_loss, perplexity, encoding_indices, z, z_q = \
                self.model(audio, midi_note)

            # Reconstruction loss
            recon_loss_dict = self.reconstruction_loss(reconstruction, audio)
            recon_loss = recon_loss_dict['loss']

            # Disentanglement loss (reuses z from the forward pass above)
            if self.perceptual_weight > 0:
                disentangle_loss_dict = self.disentanglement_loss(
                    z.detach(), encoding_indices, instrument_id, midi_note
                )
                disentangle_loss = disentangle_loss_dict['loss']
            else:
                disentangle_loss = torch.tensor(0.0, device=self.device)

            # Adversarial pitch disentanglement
            # Step 1: Update classifier using detached z_q (no encoder gradient)
            adv_loss_for_main = torch.tensor(0.0, device=self.device)
            if self.adv_enabled and adv_alpha > 0:
                cls_loss_plain, _ = self.pitch_classifier(z_q.detach(), midi_note)
                self.classifier_optimizer.zero_grad()
                cls_loss_plain.backward()
                self.classifier_optimizer.step()

                # Step 2: Adversarial signal for encoder via GRL
                z_q_grl = GradientReversal.apply(z_q, adv_alpha)
                adv_loss_for_main, _ = self.pitch_classifier(z_q_grl, midi_note)

            # Total loss
            loss = (
                self.recon_weight * recon_loss +
                self.vq_weight * vq_loss +
                self.perceptual_weight * disentangle_loss +
                self.adv_lambda * adv_loss_for_main
            )

            # NaN guard: diagnose which component is NaN and skip the step
            # to avoid corrupting model parameters.
            if not torch.isfinite(loss):
                nan_recon = not torch.isfinite(recon_loss)
                nan_vq   = not torch.isfinite(vq_loss)
                nan_z    = not torch.isfinite(z).all()
                nan_rec  = not torch.isfinite(reconstruction).all()
                print(
                    f"\n[NaN] batch {batch_idx} | "
                    f"recon_loss={nan_recon} vq_loss={nan_vq} "
                    f"z={nan_z} reconstruction={nan_rec} | "
                    f"perp={perplexity.item():.1f}"
                )
                self.optimizer.zero_grad()
                continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Zero classifier grads accumulated from the adversarial backward
            # (the main optimizer only covers model params, but we clean up anyway)
            if self.adv_enabled and self.pitch_classifier is not None:
                for p in self.pitch_classifier.parameters():
                    p.grad = None

            # Gradient clipping — must check for Inf/NaN grad norm BEFORE stepping.
            # clip_grad_norm_ computes total_norm; if any grad is Inf, total_norm=Inf,
            # scale = max_norm/Inf = 0.0, then 0.0 * Inf = NaN → corrupts all params.
            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                if not torch.isfinite(grad_norm):
                    print(f"\n[skip] Inf/NaN grad norm ({grad_norm:.2f}) at batch {batch_idx}")
                    self.optimizer.zero_grad()
                    continue

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_disentangle_loss += disentangle_loss.item()
            total_adv_loss += adv_loss_for_main.item()
            total_perplexity += perplexity.item()

            # Update progress bar
            postfix = {
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'vq': f'{vq_loss.item():.4f}',
                'perp': f'{perplexity.item():.1f}',
            }
            if self.adv_enabled:
                postfix['adv'] = f'{adv_loss_for_main.item():.4f}'
            pbar.set_postfix(postfix)

            self.global_step += 1

            # Logging
            if self.global_step % self.log_every == 0:
                self._log_step({
                    'train/loss': loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/vq_loss': vq_loss.item(),
                    'train/disentangle_loss': disentangle_loss.item(),
                    'train/adv_loss': adv_loss_for_main.item(),
                    'train/adv_alpha': adv_alpha,
                    'train/perplexity': perplexity.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                })

        # Average losses
        num_batches = len(train_loader)
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches,
            'disentangle_loss': total_disentangle_loss / num_batches,
            'adv_loss': total_adv_loss / num_batches,
            'perplexity': total_perplexity / num_batches,
        }

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary of average losses
        """
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0

        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device
            audio = batch['audio'].to(self.device)
            midi_note = batch['midi_note'].to(self.device)

            # Forward pass
            reconstruction, vq_loss, perplexity, _, _z, _z_q = \
                self.model(audio, midi_note)

            # Reconstruction loss
            recon_loss_dict = self.reconstruction_loss(reconstruction, audio)
            recon_loss = recon_loss_dict['loss']

            # Total loss
            loss = self.recon_weight * recon_loss + self.vq_weight * vq_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()

        # Average losses
        num_batches = len(val_loader)
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches,
            'perplexity': total_perplexity / num_batches,
        }

    def train(self, train_loader, val_loader, num_epochs):
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs to train
        """
        # Release any MPS memory retained from previous runs
        if self.device == 'mps':
            torch.mps.empty_cache()

        print(f"\n{'='*70}")
        print(f"Starting VQ-VAE Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch(train_loader)

            # Validate
            val_losses = self.validate(val_loader)

            # Log epoch results
            print(f"\nEpoch {epoch} Results:")
            adv_suffix = (f" | Adv: {train_losses['adv_loss']:.4f}"
                          if self.adv_enabled else "")
            print(f"  Train Loss: {train_losses['loss']:.4f} | "
                  f"Recon: {train_losses['recon_loss']:.4f} | "
                  f"VQ: {train_losses['vq_loss']:.4f} | "
                  f"Perplexity: {train_losses['perplexity']:.1f}"
                  f"{adv_suffix}")
            print(f"  Val Loss:   {val_losses['loss']:.4f} | "
                  f"Recon: {val_losses['recon_loss']:.4f} | "
                  f"VQ: {val_losses['vq_loss']:.4f} | "
                  f"Perplexity: {val_losses['perplexity']:.1f}")

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_every_n_epochs'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Save best model
            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model saved! (val_loss: {val_losses['loss']:.4f})")

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.adv_enabled and self.pitch_classifier is not None:
            checkpoint['pitch_classifier_state_dict'] = self.pitch_classifier.state_dict()
            checkpoint['classifier_optimizer_state_dict'] = self.classifier_optimizer.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.adv_enabled and self.pitch_classifier is not None:
            if 'pitch_classifier_state_dict' in checkpoint:
                self.pitch_classifier.load_state_dict(
                    checkpoint['pitch_classifier_state_dict']
                )
                self.classifier_optimizer.load_state_dict(
                    checkpoint['classifier_optimizer_state_dict']
                )

        print(f"Loaded checkpoint from epoch {self.epoch}")

    def _log_step(self, metrics):
        """Log metrics for current step."""
        # Simple file-based logging
        # You can replace with wandb/tensorboard
        log_file = self.log_dir / 'training.log'

        with open(log_file, 'a') as f:
            f.write(f"Step {self.global_step}: {metrics}\n")
