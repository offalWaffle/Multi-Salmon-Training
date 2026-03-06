"""Trainer classes for model training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm

from .losses import vae_loss_with_components


class VAETrainer:
    """Trainer for VAE model."""

    def __init__(self, model: nn.Module, config: dict, device: str = 'mps'):
        """
        Initialize VAE trainer.

        Args:
            model: VAE model to train
            config: Training configuration dict with keys:
                - learning_rate: Learning rate
                - beta: Beta parameter for VAE loss
                - gradient_clip: Optional gradient clipping value
            device: Device to train on ('mps', 'cuda', or 'cpu')
        """
        self.model = model.to(device)
        self.device = device

        # Get training config
        if 'training' in config:
            train_config = config['training']
        else:
            train_config = config

        learning_rate = train_config.get('learning_rate', 1e-4)
        self.beta = train_config.get('beta', 1.0)
        self.gradient_clip = train_config.get('gradient_clip', None)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

        # Scheduler (optional)
        scheduler_type = train_config.get('scheduler', None)
        if scheduler_type == 'cosine':
            num_epochs = train_config.get('num_epochs', 100)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs
            )
        else:
            self.scheduler = None

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader, mel_transform: Optional[callable] = None) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            mel_transform: Optional transform to convert audio to mel-spectrogram

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            audio = batch['audio'].to(self.device)

            # Convert audio to mel-spectrogram if transform provided
            if mel_transform is not None:
                with torch.no_grad():
                    mel_spec = mel_transform(audio)
            else:
                mel_spec = audio  # Assume already mel-spectrogram

            # Forward pass
            recon, mu, logvar = self.model(mel_spec)
            loss, recon_loss, kl_loss = vae_loss_with_components(
                recon, mel_spec, mu, logvar, beta=self.beta
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        self.train_losses.append(avg_loss)

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }

    def validate(self, dataloader: DataLoader, mel_transform: Optional[callable] = None) -> Dict[str, float]:
        """
        Validation loop.

        Args:
            dataloader: Validation data loader
            mel_transform: Optional transform to convert audio to mel-spectrogram

        Returns:
            Dictionary of average validation losses
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for batch in pbar:
                audio = batch['audio'].to(self.device)

                # Convert audio to mel-spectrogram if transform provided
                if mel_transform is not None:
                    mel_spec = mel_transform(audio)
                else:
                    mel_spec = audio

                # Forward pass
                recon, mu, logvar = self.model(mel_spec)
                loss, recon_loss, kl_loss = vae_loss_with_components(
                    recon, mel_spec, mu, logvar, beta=self.beta
                )

                # Track metrics
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        self.val_losses.append(avg_loss)

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }

    def save_checkpoint(self, filepath: str, epoch: int, best: bool = False):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best': best
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch']

    def step_scheduler(self):
        """Step the learning rate scheduler if it exists."""
        if self.scheduler is not None:
            self.scheduler.step()
