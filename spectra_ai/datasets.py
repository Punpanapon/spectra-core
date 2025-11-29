"""
Dataset utilities for loading EFC tiles and masks.

Expected structure under root_dir:
  root_dir/<split>/images/tile_*.png
  root_dir/<split>/masks/tile_*_mask.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class EFCDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform: Optional[callable] = None):
        """
        Parameters
        ----------
        root_dir : str
            Base directory containing efc_tiles.
        split : str
            "train" or "val". Directories are expected at root_dir/split/images and masks.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.images_dir = self.root_dir / split / "images"
        self.masks_dir = self.root_dir / split / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self.samples: List[Tuple[Path, Path]] = []
        skipped = 0
        for img_path in sorted(self.images_dir.glob("tile_*.png")):
            mask_path = self.masks_dir / img_path.name.replace(".png", "_mask.png")
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                skipped += 1

        print(f"EFCDataset(split={split}): {len(self.samples)} samples with masks, {skipped} images skipped (no mask)")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"EFCDataset(split={split}) found no image/mask pairs under {self.images_dir} and {self.masks_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0  # H x W x 3 in [0,1]
        img_np = np.transpose(img_np, (2, 0, 1))  # to C x H x W
        img_tensor = torch.from_numpy(img_np)

        # Load mask and remap to binary: 0 = background/other, 1 = deforestation
        mask = Image.open(mask_path)
        mask_np = np.array(mask, dtype="int64")
        mask_np = (mask_np == 2).astype("int64")
        mask_tensor = torch.from_numpy(mask_np)

        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor
