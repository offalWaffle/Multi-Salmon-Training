"""
DAC Pitch Adapter Trainer.

Two-step adversarial training loop:
  1. Update PitchClassifier on detached x_inner (plain cross-entropy).
  2. Update PitchStripper + PitchInjector with:
       total = recon_weight * recon_loss + adv_lambda * adv_loss(GRL)

NaN guards, alpha ramp, checkpoint save/load (including classifier state).

DAC encoder is NOT used during training — latents are precomputed.
DAC decoder IS used to decode z_modified for the reconstruction loss.
"""

import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from ..models.dac_pitch_adapter import DACPitchAdapter
from ..models.pitch_adversary import GradientReversal, PitchClassifier
from ..losses.spectral_loss import AudioReconstructionLoss


class DACAdapterTrainer:
    def __init__(self, config, dac_model, device='mps'):
        """
        Args:
            config:    Full config dict (from YAML).
            dac_model: Pretrained, frozen DAC model.
            device:    Training device.
        """
        self.config = config
        self.device = device

        # --- Build adapter (wraps frozen DAC) ---
        model_cfg = config['model']
        self.adapter = DACPitchAdapter(
            dac_model=dac_model,
            inner_dim=model_cfg['inner_dim'],
            num_residual_blocks=model_cfg['num_residual_blocks'],
            pitch_embed_dim=model_cfg['pitch_embed_dim'],
            dac_latent_dim=model_cfg['dac_latent_dim'],
        ).to(device)

        # --- Reconstruction loss ---
        self.recon_loss_fn = AudioReconstructionLoss(
            sample_rate=config['audio']['sample_rate'],
            time_weight=1.0,
            stft_weight=1.0,
            mel_weight=1.0,
        ).to(device)
        self.recon_weight = config['training']['reconstruction_weight']

        # --- Adversarial setup ---
        adv_cfg = config.get('adversarial', {})
        self.adv_enabled = adv_cfg.get('enabled', True)
        self.adv_lambda = adv_cfg.get('lambda', 0.1)
        self.adv_warmup_epochs = adv_cfg.get('alpha_warmup_epochs', 10)

        if self.adv_enabled:
            self.pitch_classifier = PitchClassifier(
                latent_dim=model_cfg['inner_dim'],  # classifier sees x_inner [B, inner_dim, T]
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

        # --- Main optimizer (adapter params only, NOT DAC) ---
        train_cfg = config['training']
        self.optimizer = optim.Adam(
            self.adapter.adapter_parameters(),
            lr=train_cfg['learning_rate'],
        )

        # --- LR scheduler ---
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['num_epochs'],
            eta_min=train_cfg.get('min_lr', 1e-6),
        )

        # --- State ---
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.gradient_clip = train_cfg.get('gradient_clip', 1.0)

        # --- Checkpointing / logging ---
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/dac_adapter'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.get('log_dir', 'logs/dac_adapter'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader):
        self.adapter.train()

        total_loss = 0.0
        total_recon = 0.0
        total_adv = 0.0
        num_batches = 0

        # Alpha ramp: 0 → 1 over first adv_warmup_epochs
        adv_alpha = min(1.0, self.epoch / max(1, self.adv_warmup_epochs))

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            latent    = batch['latent'].to(self.device)    # [B, 1024, T]
            audio     = batch['audio'].to(self.device)     # [B, 1, T_aud]
            midi_note = batch['midi_note'].to(self.device) # [B]

            # Skip corrupt batches
            if not torch.isfinite(latent).all() or not torch.isfinite(audio).all():
                print(f"\n[skip] NaN/Inf in input at batch {batch_idx}")
                continue

            # ----------------------------------------------------------------
            # Adapter forward: self-reconstruction (tgt_midi == src_midi)
            # ----------------------------------------------------------------
            z_modified, x_inner, _z_timbre = self.adapter(latent, midi_note, midi_note)

            # Decode z_modified with frozen DAC decoder
            audio_pred = self.adapter.dac.decode(z_modified)  # [B, 1, T_dec]

            # Guard: DAC decoder can emit NaN/Inf if z_modified drifts far from its
            # training distribution in early steps.
            if not torch.isfinite(audio_pred).all():
                print(f"\n[skip] NaN/Inf in audio_pred at batch {batch_idx}")
                self.optimizer.zero_grad()
                continue

            # Align lengths: DAC decoder output may differ from crop_audio by a few samples
            min_len = min(audio_pred.shape[-1], audio.shape[-1])
            audio_pred_trim = audio_pred[..., :min_len]
            audio_trim      = audio[..., :min_len]

            # ----------------------------------------------------------------
            # Step 1: Update classifier (detached x_inner, no encoder gradient)
            # ----------------------------------------------------------------
            adv_loss_for_main = torch.tensor(0.0, device=self.device)
            if self.adv_enabled and adv_alpha > 0 and self.pitch_classifier is not None:
                cls_loss, _ = self.pitch_classifier(x_inner.detach(), midi_note)
                self.classifier_optimizer.zero_grad()
                cls_loss.backward()
                self.classifier_optimizer.step()

                # ----------------------------------------------------------------
                # Step 2: Adversarial signal for adapter via GRL
                # ----------------------------------------------------------------
                x_inner_grl = GradientReversal.apply(x_inner, adv_alpha)
                adv_loss_for_main, _ = self.pitch_classifier(x_inner_grl, midi_note)

            # ----------------------------------------------------------------
            # Reconstruction loss
            # ----------------------------------------------------------------
            recon_loss_dict = self.recon_loss_fn(audio_pred_trim, audio_trim)
            recon_loss = recon_loss_dict['loss']

            # Total loss
            loss = self.recon_weight * recon_loss + self.adv_lambda * adv_loss_for_main

            # NaN guard
            if not torch.isfinite(loss):
                print(
                    f"\n[NaN] batch {batch_idx} | "
                    f"recon={not torch.isfinite(recon_loss)} "
                    f"adv={not torch.isfinite(adv_loss_for_main)}"
                )
                self.optimizer.zero_grad()
                continue

            # ----------------------------------------------------------------
            # Backward pass (main optimizer)
            # ----------------------------------------------------------------
            self.optimizer.zero_grad()
            loss.backward()

            # Zero any classifier grads that leaked via GRL backward
            if self.adv_enabled and self.pitch_classifier is not None:
                for p in self.pitch_classifier.parameters():
                    p.grad = None

            # Gradient clipping
            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.adapter.adapter_parameters(),
                    self.gradient_clip,
                )
                if not torch.isfinite(grad_norm):
                    print(f"\n[skip] Inf/NaN grad norm at batch {batch_idx}")
                    self.optimizer.zero_grad()
                    continue

            self.optimizer.step()

            # Accumulate
            total_loss  += loss.item()
            total_recon += recon_loss.item()
            total_adv   += adv_loss_for_main.item()
            num_batches += 1

            postfix = {
                'loss':  f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
            }
            if self.adv_enabled:
                postfix['adv'] = f'{adv_loss_for_main.item():.4f}'
            pbar.set_postfix(postfix)

            self.global_step += 1
            self._log_step({
                'train/loss':       loss.item(),
                'train/recon_loss': recon_loss.item(),
                'train/adv_loss':   adv_loss_for_main.item(),
                'train/adv_alpha':  adv_alpha,
                'train/lr':         self.optimizer.param_groups[0]['lr'],
            })

        if num_batches == 0:
            return {'loss': float('nan'), 'recon_loss': float('nan'), 'adv_loss': float('nan')}

        return {
            'loss':       total_loss  / num_batches,
            'recon_loss': total_recon / num_batches,
            'adv_loss':   total_adv   / num_batches,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader):
        self.adapter.eval()

        total_loss  = 0.0
        total_recon = 0.0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            latent    = batch['latent'].to(self.device)
            audio     = batch['audio'].to(self.device)
            midi_note = batch['midi_note'].to(self.device)

            z_modified, _x_inner, _z_timbre = self.adapter(latent, midi_note, midi_note)
            audio_pred = self.adapter.dac.decode(z_modified)

            min_len = min(audio_pred.shape[-1], audio.shape[-1])
            recon_loss_dict = self.recon_loss_fn(
                audio_pred[..., :min_len], audio[..., :min_len]
            )
            recon_loss = recon_loss_dict['loss']

            if not torch.isfinite(recon_loss):
                continue

            total_loss  += recon_loss.item()
            total_recon += recon_loss.item()
            num_batches += 1

        if num_batches == 0:
            return {'loss': float('nan'), 'recon_loss': float('nan')}

        return {
            'loss':       total_loss  / num_batches,
            'recon_loss': total_recon / num_batches,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, num_epochs):
        if self.device == 'mps':
            torch.mps.empty_cache()

        save_every = self.config.get('checkpoint', {}).get('save_every_n_epochs', 10)

        adapter_params = sum(p.numel() for p in self.adapter.adapter_parameters())
        print(f"\n{'='*70}")
        print("DAC Pitch Adapter Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Adapter parameters: {adapter_params:,}")
        print(f"Adversarial: {self.adv_enabled}")
        print(f"{'='*70}\n")

        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch

            train_losses = self.train_epoch(train_loader)
            val_losses   = self.validate(val_loader)

            adv_sfx = (f" | Adv: {train_losses['adv_loss']:.4f}"
                       if self.adv_enabled else "")
            print(
                f"\nEpoch {epoch} | "
                f"Train: {train_losses['loss']:.4f} "
                f"(recon={train_losses['recon_loss']:.4f}{adv_sfx}) | "
                f"Val: {val_losses['loss']:.4f}"
            )

            self.scheduler.step()

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model (val_loss={val_losses['loss']:.4f})")

        self.save_checkpoint('final_model.pt')
        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch':               self.epoch,
            'global_step':         self.global_step,
            'best_val_loss':       self.best_val_loss,
            'config':              self.config,
            'adapter_state_dict':  self.adapter.stripper.state_dict(),
            'injector_state_dict': self.adapter.injector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if self.adv_enabled and self.pitch_classifier is not None:
            checkpoint['classifier_state_dict'] = self.pitch_classifier.state_dict()
            checkpoint['classifier_optimizer_state_dict'] = self.classifier_optimizer.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.adapter.stripper.load_state_dict(checkpoint['adapter_state_dict'])
        self.adapter.injector.load_state_dict(checkpoint['injector_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch         = checkpoint['epoch'] + 1  # resume from next epoch
        self.global_step   = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.adv_enabled and self.pitch_classifier is not None:
            if 'classifier_state_dict' in checkpoint:
                self.pitch_classifier.load_state_dict(checkpoint['classifier_state_dict'])
                self.classifier_optimizer.load_state_dict(
                    checkpoint['classifier_optimizer_state_dict']
                )

        print(f"Resumed from epoch {self.epoch} (best_val_loss={self.best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(self, metrics):
        log_file = self.log_dir / 'training.log'
        with open(log_file, 'a') as f:
            f.write(f"Step {self.global_step}: {metrics}\n")
