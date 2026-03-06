#!/usr/bin/env python3
"""
Train DAC Pitch Adapter.

Usage:
    python scripts/train_dac_adapter.py [--config config/dac_adapter_config.yaml]
                                        [--resume checkpoints/dac_adapter/best_model.pt]
                                        [--device mps|cuda|cpu]
"""

import sys
import argparse
from pathlib import Path
import yaml
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dac_latent_dataset import create_dataloader
from src.models.conditioned_dac import load_pretrained_dac
from src.training.dac_adapter_trainer import DACAdapterTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train DAC Pitch Adapter')
    parser.add_argument('--config', type=str, default='config/dac_adapter_config.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu); auto-detected if omitted')
    return parser.parse_args()


def get_device(device_arg=None):
    if device_arg is not None:
        return device_arg
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("DAC PITCH ADAPTER TRAINING")
    print("="*70)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load frozen DAC
    print("\nLoading pretrained DAC...")
    dac_model = load_pretrained_dac(
        model_type=config['model']['dac_model_type'],
        device=device,
    )
    dac_model.eval()

    # Build dataloaders
    data_cfg = config['data']
    latent_dir = Path(data_cfg['latent_dir'])

    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        latent_dir=latent_dir / 'train',
        batch_size=config['training']['batch_size'],
        duration=config['audio']['duration'],
        sample_rate=config['audio']['sample_rate'],
        augment=True,
        num_workers=data_cfg['num_workers'],
        shuffle=True,
        pin_memory=data_cfg.get('pin_memory', False),
    )
    val_loader = create_dataloader(
        latent_dir=latent_dir / 'val',
        batch_size=config['training']['batch_size'],
        duration=config['audio']['duration'],
        sample_rate=config['audio']['sample_rate'],
        augment=False,
        num_workers=data_cfg['num_workers'],
        shuffle=False,
        pin_memory=data_cfg.get('pin_memory', False),
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DACAdapterTrainer(config, dac_model=dac_model, device=device)

    if args.resume is not None:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader, num_epochs=config['training']['num_epochs'])

    print(f"\nCheckpoints: {trainer.checkpoint_dir}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print("\nNext steps:")
    print("  python scripts/generate_instrument_dac.py --source <audio.wav> --midi 60")


if __name__ == '__main__':
    main()
