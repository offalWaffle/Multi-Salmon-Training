#!/usr/bin/env python3
"""
Train VQ-VAE model for one-shot instrument generation.

Usage:
    python scripts/train_vqvae.py [--config config/vqvae_config.yaml] [--resume checkpoint.pt]
"""

import sys
import argparse
from pathlib import Path
import yaml
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.vqvae_dataset import create_dataloader
from src.training.vqvae_trainer import VQVAETrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VQ-VAE model')

    parser.add_argument('--config', type=str, default='config/vqvae_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cuda/mps/cpu), auto-detect if not specified')

    return parser.parse_args()


def get_device(device_arg=None):
    """Determine which device to use."""
    if device_arg is not None:
        return device_arg

    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("VQ-VAE TRAINING")
    print("="*70)

    # Load config
    print(f"\nLoading config from: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        metadata_path=config['data']['train_path'],
        batch_size=config['training']['batch_size'],
        sample_rate=config['audio']['sample_rate'],
        duration=config['audio']['duration'],
        augment=True,  # Enable augmentation for training
        num_workers=config['data']['num_workers'],
        shuffle=True,
        pin_memory=config['data']['pin_memory']
    )

    val_loader = create_dataloader(
        metadata_path=config['data']['val_path'],
        batch_size=config['training']['batch_size'],
        sample_rate=config['audio']['sample_rate'],
        duration=config['audio']['duration'],
        augment=False,  # No augmentation for validation
        num_workers=config['data']['num_workers'],
        shuffle=False,
        pin_memory=config['data']['pin_memory']
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = VQVAETrainer(config, device=device)

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    num_epochs = config['training']['num_epochs']
    trainer.train(train_loader, val_loader, num_epochs)

    # Save final model
    print("\nSaving final model...")
    trainer.save_checkpoint('final_model.pt')

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")

    # Display success criteria
    print("\n" + "-"*70)
    print("SUCCESS CRITERIA (Phase 1):")
    print("-"*70)
    criteria = config.get('success_criteria', {})
    print(f"Target Reconstruction SDR: > {criteria.get('min_reconstruction_sdr', 10)} dB")
    print(f"Target Pitch Accuracy:     > {criteria.get('min_pitch_accuracy', 0.7)*100}%")
    print(f"Target Codebook Perplexity: > {criteria.get('min_codebook_perplexity', 100)}")
    print("-"*70)
    print("\nNext steps:")
    print("  1. Evaluate model: python scripts/evaluate_vqvae.py")
    print("  2. Test one-shot generation: python scripts/test_oneshot.py")
    print()


if __name__ == "__main__":
    main()
