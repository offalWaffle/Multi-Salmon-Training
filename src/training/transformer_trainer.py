"""
LatentTransformer Trainer.

Trains the LatentTransformer with a frozen VQ-VAE.

Forward pass per batch:
    1. Encode source audio  →  z_q_src  (frozen VQ-VAE encoder + quantizer, no grad)
    2. Encode target audio  →  z_q_tgt  (frozen, used only for latent reg loss)
    3. Transform            →  z_pred = LatentTransformer(z_q_src, target_midi, target_vel)
    4. Decode               →  audio_pred = VQ-VAE_decoder(z_pred, target_midi)
                               (gradients flow through frozen decoder back to z_pred)
    5. Loss                 →  audio_loss(audio_pred, audio_tgt)
                              + latent_weight * MSE(z_pred, z_q_tgt)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from pathlib import Path
from tqdm import tqdm

from ..models.vqvae import VQVAE
from ..models.latent_transformer import LatentTransformer
from ..losses.spectral_loss import AudioReconstructionLoss


class LatentTransformerTrainer:

    def __init__(self, config, device='mps'):
        """
        Args:
            config: Parsed YAML config dict.
            device: 'mps', 'cuda', or 'cpu'.
        """
        self.config = config
        self.device = device
        train_cfg = config['training']

        # ------------------------------------------------------------------
        # Load VQ-VAE (frozen)
        # ------------------------------------------------------------------
        self.vqvae = self._load_vqvae(
            config['vqvae']['checkpoint'],
            config['vqvae']['config'],
        )
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # Build LatentTransformer
        # ------------------------------------------------------------------
        m = config['model']
        self.model = LatentTransformer(
            latent_channels=m['latent_channels'],
            latent_time_steps=m['latent_time_steps'],
            hidden_dim=m['hidden_dim'],
            num_layers=m['num_layers'],
            midi_embed_dim=m['midi_embed_dim'],
            velocity_embed_dim=m.get('velocity_embed_dim', 64),
            use_residual=m['use_residual'],
            dropout=m['dropout'],
        ).to(device)

        # ------------------------------------------------------------------
        # Losses
        # ------------------------------------------------------------------
        self.reconstruction_loss = AudioReconstructionLoss(
            sample_rate=train_cfg['sample_rate'],
            time_weight=1.0,
            stft_weight=1.0,
            mel_weight=1.0,
        ).to(device)

        self.audio_weight  = train_cfg.get('audio_weight', 1.0)
        self.latent_weight = train_cfg.get('latent_weight', 0.1)

        # ------------------------------------------------------------------
        # Optimizer (only LatentTransformer parameters)
        # ------------------------------------------------------------------
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg.get('weight_decay', 0.01),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['num_epochs'],
            eta_min=train_cfg.get('min_learning_rate', 1e-6),
        )
        self.grad_clip = train_cfg.get('grad_clip', 1.0)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.epoch       = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        self.checkpoint_dir = Path(train_cfg['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._target_len = int(train_cfg['sample_rate'] * train_cfg['duration'])

    # ------------------------------------------------------------------
    # VQ-VAE helpers
    # ------------------------------------------------------------------

    def _load_vqvae(self, checkpoint_path, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model = VQVAE(cfg['model']).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded VQ-VAE from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
        return model

    def _encode(self, audio):
        """Encode audio → z_q using frozen VQ-VAE (no gradients tracked)."""
        with torch.no_grad():
            z = self.vqvae.encoder(audio)
            z_q, _, _, _ = self.vqvae.quantizer(z)
        return z_q

    def _decode(self, z, midi):
        """
        Decode latent → audio using the frozen VQ-VAE decoder.

        Gradients are NOT blocked here so that loss.backward() can update
        the LatentTransformer through the (frozen) decoder computations.
        The decoder params have requires_grad=False, so they won't be updated.
        """
        audio = self.vqvae.decoder(z, midi)
        # Match target length (strided convolutions may drift by a few samples)
        if audio.shape[-1] != self._target_len:
            diff = self._target_len - audio.shape[-1]
            if diff > 0:
                audio = F.pad(audio, (0, diff))
            else:
                audio = audio[..., :self._target_len]
        return audio

    # ------------------------------------------------------------------
    # Training / validation loops
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader):
        self.model.train()
        self.vqvae.eval()

        total_loss    = 0.0
        total_audio   = 0.0
        total_latent  = 0.0
        n_batches     = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            src_audio    = batch['source_audio'].to(self.device)
            src_midi     = batch['source_midi'].to(self.device)
            tgt_audio    = batch['target_audio'].to(self.device)
            tgt_midi     = batch['target_midi'].to(self.device)
            tgt_velocity = batch['target_velocity'].to(self.device)

            if not (torch.isfinite(src_audio).all() and torch.isfinite(tgt_audio).all()):
                continue

            # Step 1–2: encode source & target (no grad)
            z_q_src = self._encode(src_audio)
            z_q_tgt = self._encode(tgt_audio)

            # Pitch delta: how many semitones to shift (range -127..+127 → normalised to -1..+1)
            # This gives the transformer a direct "how much to transform" signal.
            pitch_delta_norm = (tgt_midi.float() - src_midi.float()) / 127.0
            tgt_vel_norm     = tgt_velocity.float() / 127.0

            # Step 3: transform
            z_pred = self.model(z_q_src, pitch_delta_norm, tgt_vel_norm)

            # Step 4: decode (gradients flow back through decoder to z_pred)
            audio_pred = self._decode(z_pred, tgt_midi)

            # Step 5: losses
            audio_loss  = self.reconstruction_loss(audio_pred, tgt_audio)['loss']
            latent_loss = F.mse_loss(z_pred, z_q_tgt.detach())
            loss = self.audio_weight * audio_loss + self.latent_weight * latent_loss

            if not torch.isfinite(loss):
                print(
                    f"\n[NaN] batch {batch_idx}: "
                    f"audio={audio_loss.item():.4f} latent={latent_loss.item():.4f}"
                )
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    print(f"\n[skip] Inf/NaN grad norm at batch {batch_idx}")
                    self.optimizer.zero_grad()
                    continue

            self.optimizer.step()

            total_loss   += loss.item()
            total_audio  += audio_loss.item()
            total_latent += latent_loss.item()
            n_batches    += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss':   f'{loss.item():.4f}',
                'audio':  f'{audio_loss.item():.4f}',
                'latent': f'{latent_loss.item():.4f}',
            })

        if n_batches == 0:
            return {'loss': float('nan'), 'audio_loss': float('nan'), 'latent_loss': float('nan')}

        return {
            'loss':        total_loss   / n_batches,
            'audio_loss':  total_audio  / n_batches,
            'latent_loss': total_latent / n_batches,
        }

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        self.vqvae.eval()

        total_loss    = 0.0
        total_audio   = 0.0
        total_latent  = 0.0
        n_batches     = 0

        for batch in tqdm(val_loader, desc="Validation"):
            src_audio    = batch['source_audio'].to(self.device)
            src_midi     = batch['source_midi'].to(self.device)
            tgt_audio    = batch['target_audio'].to(self.device)
            tgt_midi     = batch['target_midi'].to(self.device)
            tgt_velocity = batch['target_velocity'].to(self.device)

            z_q_src = self._encode(src_audio)
            z_q_tgt = self._encode(tgt_audio)

            pitch_delta_norm = (tgt_midi.float() - src_midi.float()) / 127.0
            tgt_vel_norm     = tgt_velocity.float() / 127.0

            z_pred     = self.model(z_q_src, pitch_delta_norm, tgt_vel_norm)
            audio_pred = self._decode(z_pred, tgt_midi)

            audio_loss  = self.reconstruction_loss(audio_pred, tgt_audio)['loss']
            latent_loss = F.mse_loss(z_pred, z_q_tgt)
            loss = self.audio_weight * audio_loss + self.latent_weight * latent_loss

            if not torch.isfinite(loss):
                continue

            total_loss   += loss.item()
            total_audio  += audio_loss.item()
            total_latent += latent_loss.item()
            n_batches    += 1

        if n_batches == 0:
            return {'loss': float('nan'), 'audio_loss': float('nan'), 'latent_loss': float('nan')}

        return {
            'loss':        total_loss   / n_batches,
            'audio_loss':  total_audio  / n_batches,
            'latent_loss': total_latent / n_batches,
        }

    def train(self, train_loader, val_loader, num_epochs):
        if self.device == 'mps':
            torch.mps.empty_cache()

        print(f"\n{'='*70}")
        print("Training LatentTransformer")
        print(f"{'='*70}")
        print(f"Device : {self.device}")
        print(f"Epochs : {num_epochs}")
        print(f"Transformer params : {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"VQ-VAE params (frozen) : {sum(p.numel() for p in self.vqvae.parameters()):,}")
        print(f"{'='*70}\n")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            train_losses = self.train_epoch(train_loader)
            val_losses   = self.validate(val_loader)

            print(
                f"\nEpoch {epoch}:\n"
                f"  Train: loss={train_losses['loss']:.4f} | "
                f"audio={train_losses['audio_loss']:.4f} | "
                f"latent={train_losses['latent_loss']:.4f}\n"
                f"  Val:   loss={val_losses['loss']:.4f} | "
                f"audio={val_losses['audio_loss']:.4f} | "
                f"latent={val_losses['latent_loss']:.4f}"
            )

            self.scheduler.step()

            save_every = self.config['training'].get('save_every', 10)
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model (val_loss: {val_losses['loss']:.4f})")

        self.save_checkpoint('final_model.pt')
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch':                self.epoch,
            'global_step':          self.global_step,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss':        self.best_val_loss,
            'config':               self.config,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch         = ckpt['epoch'] + 1  # start at next epoch
        self.global_step   = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        print(f"Resumed from epoch {ckpt['epoch']} → starting epoch {self.epoch} (best val loss: {self.best_val_loss:.4f})")
