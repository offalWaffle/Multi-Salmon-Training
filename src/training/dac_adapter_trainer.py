"""
DAC Pitch Adapter Trainer.

Loss: per-frame cosine direction + log-magnitude
  - direction_loss = mean over (B, T) of (1 - cos_sim(z_mod, z_tgt) along channel)
  - magnitude_loss = L1(log|z_mod|, log|z_tgt|) per frame
  - recon_loss     = direction_loss + magnitude_loss_weight * magnitude_loss
  Replaces flat L1, which under-weights low-energy tail frames (their absolute
  error is small even when their direction is wrong) and caused a learned bias
  delta to show up as a fixed tonal "buzz" in decay regions.

Pairs share instrument + velocity; only pitch differs (or src == tgt for the
identity-pair subset). No DAC decoder in the gradient path. No adversarial loss.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from ..models.dac_pitch_adapter import DACPitchAdapter


def _recon_loss(z_modified, z_target, magnitude_weight=0.1, eps=1e-6):
    """Per-frame cosine direction + log-magnitude loss.

    Equal gradient pressure across frames regardless of their latent magnitude —
    forces the model to predict tail frames as accurately as attack frames.
    """
    cos_sim = F.cosine_similarity(z_modified, z_target, dim=1)   # [B, T]
    direction_loss = (1.0 - cos_sim).mean()
    mag_mod = z_modified.norm(dim=1)                              # [B, T]
    mag_tgt = z_target.norm(dim=1)
    magnitude_loss = F.l1_loss(torch.log(mag_mod + eps),
                               torch.log(mag_tgt + eps))
    recon = direction_loss + magnitude_weight * magnitude_loss
    return recon, direction_loss, magnitude_loss


class DACAdapterTrainer:
    def __init__(self, config, dac_model, device='mps'):
        self.config = config
        self.device = device

        model_cfg = config['model']
        self.adapter = DACPitchAdapter(
            dac_model=dac_model,
            inner_dim=model_cfg['inner_dim'],
            num_residual_blocks=model_cfg['num_residual_blocks'],
            pitch_embed_dim=model_cfg['pitch_embed_dim'],
            dac_latent_dim=model_cfg['dac_latent_dim'],
        ).to(device)

        train_cfg = config['training']
        self.optimizer = optim.Adam(
            self.adapter.adapter_parameters(),
            lr=train_cfg['learning_rate'],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['num_epochs'],
            eta_min=train_cfg.get('min_lr', 1e-6),
        )

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.gradient_clip     = train_cfg.get('gradient_clip', 1.0)
        self.probe_weight      = train_cfg.get('pitch_probe_weight', 0.1)
        self.magnitude_weight  = train_cfg.get('magnitude_loss_weight', 0.1)
        self.midi_offset       = config.get('midi_offset', 21)  # MIDI 21 = class 0

        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/dac_adapter'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.get('log_dir', 'logs/dac_adapter'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader):
        self.adapter.train()

        total_loss       = 0.0
        total_recon_loss = 0.0
        total_dir_loss   = 0.0
        total_mag_loss   = 0.0
        total_probe_loss = 0.0
        num_batches      = 0
        src_correct      = 0   # tracked but not printed; used to confirm src suppression

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            src_latent = batch['src_latent'].to(self.device)  # [B, 1024, T]
            src_midi   = batch['src_midi'].to(self.device)    # [B]
            tgt_latent = batch['tgt_latent'].to(self.device)  # [B, 1024, T]
            tgt_midi   = batch['tgt_midi'].to(self.device)    # [B]

            if not torch.isfinite(src_latent).all() or not torch.isfinite(tgt_latent).all():
                print(f"\n[skip] NaN/Inf in latent at batch {batch_idx}")
                continue

            # Stripper stays near-identity (no gradient source)
            with torch.no_grad():
                z_timbre, _ = self.adapter.stripper(src_latent)

            z_modified = self.adapter.injector(z_timbre, src_midi, tgt_midi)

            recon_loss, direction_loss, magnitude_loss = _recon_loss(
                z_modified, tgt_latent.detach(), magnitude_weight=self.magnitude_weight
            )

            # Pitch probe — two signals:
            #   (a) target probe:       z_modified must predict tgt_midi  [pull toward target]
            #   (b) source suppression: minimise p(src_class) directly — bounded [0,1], stable grads
            tgt_class    = (tgt_midi - self.midi_offset).long()   # [B], range 0–87
            src_class    = (src_midi - self.midi_offset).long()   # [B], range 0–87
            probe_logits = self.adapter.probe(z_modified)         # [B, 88]
            tgt_probe_loss = F.cross_entropy(probe_logits, tgt_class)
            # Source suppression: directly minimise probability mass on src_class
            src_probs      = torch.softmax(probe_logits, dim=-1)
            src_probe_loss = src_probs.gather(1, src_class.unsqueeze(1)).squeeze(1).mean()
            probe_loss = tgt_probe_loss + 0.5 * src_probe_loss

            loss = recon_loss + self.probe_weight * probe_loss

            if not torch.isfinite(loss):
                print(f"\n[NaN] batch {batch_idx} loss={loss.item():.4f}")
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.adapter.adapter_parameters(), self.gradient_clip
                )
                if not torch.isfinite(grad_norm):
                    print(f"\n[skip] Inf/NaN grad norm at batch {batch_idx}")
                    self.optimizer.zero_grad()
                    continue

            self.optimizer.step()

            total_loss       += loss.item()
            total_recon_loss += recon_loss.item()
            total_dir_loss   += direction_loss.item()
            total_mag_loss   += magnitude_loss.item()
            total_probe_loss += probe_loss.item()
            num_batches      += 1

            pbar.set_postfix({
                'dir':   f'{direction_loss.item():.4f}',
                'mag':   f'{magnitude_loss.item():.4f}',
                'probe': f'{tgt_probe_loss.item():.4f}',
            })
            self.global_step += 1
            self._log_step({
                'train/loss':           loss.item(),
                'train/recon_loss':     recon_loss.item(),
                'train/direction_loss': direction_loss.item(),
                'train/magnitude_loss': magnitude_loss.item(),
                'train/probe_loss':     probe_loss.item(),
                'train/tgt_probe_loss': tgt_probe_loss.item(),
                'train/src_probe_loss': src_probe_loss.item(),
                'train/lr':             self.optimizer.param_groups[0]['lr'],
            })

        if num_batches == 0:
            return {'loss': float('nan'), 'recon_loss': float('nan'),
                    'direction_loss': float('nan'), 'magnitude_loss': float('nan'),
                    'probe_loss': float('nan')}
        return {
            'loss':           total_loss       / num_batches,
            'recon_loss':     total_recon_loss / num_batches,
            'direction_loss': total_dir_loss   / num_batches,
            'magnitude_loss': total_mag_loss   / num_batches,
            'probe_loss':     total_probe_loss / num_batches,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader):
        self.adapter.eval()

        total_loss       = 0.0
        total_recon_loss = 0.0
        total_dir_loss   = 0.0
        total_mag_loss   = 0.0
        total_probe_loss = 0.0
        correct          = 0
        src_correct      = 0
        total_samples    = 0
        num_batches      = 0

        for batch in tqdm(val_loader, desc="Validation"):
            src_latent = batch['src_latent'].to(self.device)
            src_midi   = batch['src_midi'].to(self.device)
            tgt_latent = batch['tgt_latent'].to(self.device)
            tgt_midi   = batch['tgt_midi'].to(self.device)

            z_timbre, _  = self.adapter.stripper(src_latent)
            z_modified   = self.adapter.injector(z_timbre, src_midi, tgt_midi)
            recon_loss, direction_loss, magnitude_loss = _recon_loss(
                z_modified, tgt_latent, magnitude_weight=self.magnitude_weight
            )

            tgt_class      = (tgt_midi - self.midi_offset).long()
            src_class      = (src_midi - self.midi_offset).long()
            probe_logits   = self.adapter.probe(z_modified)
            tgt_probe_loss = F.cross_entropy(probe_logits, tgt_class)
            src_probs      = torch.softmax(probe_logits, dim=-1)
            src_probe_loss = src_probs.gather(1, src_class.unsqueeze(1)).squeeze(1).mean()
            probe_loss     = tgt_probe_loss + 0.5 * src_probe_loss
            loss           = recon_loss + self.probe_weight * probe_loss

            if not torch.isfinite(loss):
                continue

            # Probe accuracy: target (want high) and source (want low = suppressed)
            preds          = probe_logits.argmax(dim=-1)
            correct        += (preds == tgt_class).sum().item()
            src_correct    += (preds == src_class).sum().item()
            total_samples  += tgt_class.shape[0]

            total_loss       += loss.item()
            total_recon_loss += recon_loss.item()
            total_dir_loss   += direction_loss.item()
            total_mag_loss   += magnitude_loss.item()
            total_probe_loss += probe_loss.item()
            num_batches      += 1

        if num_batches == 0:
            return {'loss': float('nan'), 'recon_loss': float('nan'),
                    'direction_loss': float('nan'), 'magnitude_loss': float('nan'),
                    'probe_loss': float('nan'), 'probe_acc': 0.0, 'src_probe_acc': 0.0}
        return {
            'loss':           total_loss       / num_batches,
            'recon_loss':     total_recon_loss / num_batches,
            'direction_loss': total_dir_loss   / num_batches,
            'magnitude_loss': total_mag_loss   / num_batches,
            'probe_loss':     total_probe_loss / num_batches,
            'probe_acc':      correct     / max(1, total_samples),
            'src_probe_acc':  src_correct / max(1, total_samples),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, num_epochs):
        if self.device == 'mps':
            torch.mps.empty_cache()

        save_every    = self.config.get('checkpoint', {}).get('save_every_n_epochs', 10)
        adapter_params = sum(p.numel() for p in self.adapter.adapter_parameters())

        print(f"\n{'='*70}")
        print("DAC Pitch Adapter Training  [cosine direction + log-magnitude]")
        print(f"{'='*70}")
        print(f"Device: {self.device}  |  Epochs: {num_epochs}  |  Params: {adapter_params:,}")
        print(f"Magnitude weight: {self.magnitude_weight}")
        print(f"{'='*70}\n")

        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch

            train_losses = self.train_epoch(train_loader)
            val_losses   = self.validate(val_loader)

            print(
                f"\nEpoch {epoch} | "
                f"train={train_losses['loss']:.4f} "
                f"(dir={train_losses['direction_loss']:.4f} "
                f"mag={train_losses['magnitude_loss']:.4f} "
                f"probe={train_losses['probe_loss']:.4f}) | "
                f"val={val_losses['loss']:.4f} "
                f"(dir={val_losses['direction_loss']:.4f} "
                f"mag={val_losses['magnitude_loss']:.4f} "
                f"tgt_acc={val_losses['probe_acc']*100:.1f}% "
                f"src_acc={val_losses['src_probe_acc']*100:.1f}%)"
            )

            self.scheduler.step()

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best (val={val_losses['loss']:.4f})")

        self.save_checkpoint('final_model.pt')
        print(f"\nBest val loss: {self.best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch':                self.epoch,
            'global_step':          self.global_step,
            'best_val_loss':        self.best_val_loss,
            'config':               self.config,
            'adapter_state_dict':   self.adapter.stripper.state_dict(),
            'injector_state_dict':  self.adapter.injector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
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

        self.epoch         = checkpoint['epoch'] + 1
        self.global_step   = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from epoch {self.epoch} (best_val_loss={self.best_val_loss:.4f})")

    def _log_step(self, metrics):
        log_file = self.log_dir / 'training.log'
        with open(log_file, 'a') as f:
            f.write(f"Step {self.global_step}: {metrics}\n")
