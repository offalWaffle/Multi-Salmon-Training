#!/usr/bin/env python3
"""
Train LatentTransformer for VQ-VAE pitch-conditioned generation (Phase 2).

Usage:
    python scripts/train_latent_transformer.py
    python scripts/train_latent_transformer.py --config config/latent_transformer_config.yaml
    python scripts/train_latent_transformer.py --resume checkpoints/latent_transformer_vqvae/latest.pt
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transformer_dataset import TransformerDataset, collate_fn
from src.training.transformer_trainer import LatentTransformerTrainer


def main():
    parser = argparse.ArgumentParser(description='Train LatentTransformer (Phase 2)')
    parser.add_argument(
        '--config', type=str,
        default='config/latent_transformer_config.yaml',
        help='Path to YAML config file',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device selection
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    train_cfg = config['training']

    # Datasets
    train_dataset = TransformerDataset(
        metadata_path=config['data']['train_path'],
        sample_rate=train_cfg['sample_rate'],
        duration=train_cfg['duration'],
        augment=True,
    )
    val_dataset = TransformerDataset(
        metadata_path=config['data']['val_path'],
        sample_rate=train_cfg['sample_rate'],
        duration=train_cfg['duration'],
        augment=False,
    )

    num_workers = train_cfg.get('num_workers', 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Trainer
    trainer = LatentTransformerTrainer(config, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(train_loader, val_loader, train_cfg['num_epochs'])


if __name__ == '__main__':
    main()
