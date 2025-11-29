"""
Train a UNet segmentation model on EFC tiles.

Expected dataset layout (relative to data_root):
  data_root/train/images/tile_*.png
  data_root/train/masks/tile_*_mask.png
  data_root/val/images/tile_*.png
  data_root/val/masks/tile_*_mask.png

Masks should use class ids:
  0 = other, 1 = forest, 2 = deforested
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import EFCDataset
from .unet_efc import EfcUNet


def train_unet(
    data_root: str,
    models_dir: str,
    num_epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> None:
    """
    Train a UNet segmentation model on EFC tiles and save the best checkpoint.

    Parameters
    ----------
    data_root: str
        Root directory containing train/val subfolders with images/ and masks/.
    models_dir: str
        Output directory to save efc_unet.pt.
    num_epochs: int
        Number of training epochs.
    batch_size: int
        Batch size for training/validation.
    lr: float
        Learning rate for Adam optimizer.
    device: str | None
        Force device selection; if None, choose CUDA when available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = EFCDataset(data_root, split="train")
    val_ds = EFCDataset(data_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = EfcUNet(n_channels=3, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    os.makedirs(models_dir, exist_ok=True)
    best_path = os.path.join(models_dir, "efc_unet.pt")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  Saved new best model to {best_path}")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}. Model saved to {best_path}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT / "data" / "efc_tiles"
    MODELS_DIR = PROJECT_ROOT / "models"
    train_unet(str(DATA_ROOT), str(MODELS_DIR))
