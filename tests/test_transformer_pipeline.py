"""Quick smoke-test for the Phase 2 LatentTransformer pipeline."""

import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transformer_dataset import TransformerDataset, collate_fn
from src.training.transformer_trainer import LatentTransformerTrainer


def test_dataset():
    print("\n--- Step 1: Dataset ---")
    ds = TransformerDataset("data/vqvae_piano/train.json", duration=2.0)
    batch = collate_fn([ds[0], ds[1], ds[2], ds[3]])

    print("source_audio:", batch["source_audio"].shape)   # [4, 1, 88200]
    print("target_audio:", batch["target_audio"].shape)   # [4, 1, 88200]
    print("source_midi: ", batch["source_midi"].tolist())
    print("target_midi: ", batch["target_midi"].tolist())
    print("any same-pitch pairs:", (batch["source_midi"] == batch["target_midi"]).any().item())

    assert batch["source_audio"].shape == (4, 1, 88200)
    assert batch["target_audio"].shape == (4, 1, 88200)
    print("PASS")
    return batch


def test_forward_pass(batch):
    print("\n--- Step 2: Forward pass ---")
    with open("config/latent_transformer_config.yaml") as f:
        config = yaml.safe_load(f)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    trainer = LatentTransformerTrainer(config, device=device)

    src      = batch["source_audio"].to(device)
    tgt_midi = batch["target_midi"].to(device)
    tgt_vel  = batch["target_velocity"].to(device)

    z_q_src = trainer._encode(src)
    print("z_q_src shape:", z_q_src.shape)          # [4, 64, 5512]

    z_pred = trainer.model(z_q_src, tgt_midi.float() / 127.0, tgt_vel.float() / 127.0)
    print("z_pred shape:", z_pred.shape)             # [4, 64, 5512]

    audio_pred = trainer._decode(z_pred, tgt_midi)
    print("audio_pred shape:", audio_pred.shape)     # [4, 1, 88200]
    print("All finite:", torch.isfinite(audio_pred).all().item())

    assert z_q_src.shape == (4, 64, 5512), f"unexpected z shape: {z_q_src.shape}"
    assert z_pred.shape  == z_q_src.shape
    assert audio_pred.shape == (4, 1, 88200)
    assert torch.isfinite(audio_pred).all()
    print("PASS")
    return trainer


def test_two_epochs(trainer):
    print("\n--- Step 3: 2-epoch mini run ---")
    config = trainer.config.copy()
    config["training"]["num_epochs"] = 2
    config["training"]["save_every"] = 999  # skip checkpointing

    ds     = TransformerDataset("data/vqvae_piano/train.json", duration=2.0)
    val_ds = TransformerDataset("data/vqvae_piano/val.json",   duration=2.0)
    loader     = DataLoader(ds,     batch_size=4, shuffle=True,  num_workers=0,
                            collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

    trainer.train(loader, val_loader, num_epochs=2)
    print("PASS")


if __name__ == "__main__":
    batch   = test_dataset()
    trainer = test_forward_pass(batch)
    test_two_epochs(trainer)
    print("\nAll tests passed!")
