#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path("data/efc_tiles")


def check_split(split: str) -> None:
    img_dir = BASE / split / "images"
    mask_dir = BASE / split / "masks"

    print(f"=== Checking split: {split} ===")
    if not img_dir.exists():
        print(f"  Images dir not found: {img_dir}")
        return
    if not mask_dir.exists():
        print(f"  Masks dir not found: {mask_dir}")
        return

    num_ok = 0
    num_missing = 0
    num_mismatch = 0
    bad_values = 0

    for img_path in sorted(img_dir.glob("tile_*.png")):
        mask_path = mask_dir / img_path.name.replace(".png", "_mask.png")
        if not mask_path.exists():
            print(f"  [MISSING MASK] {mask_path.name}")
            num_missing += 1
            continue

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        if img.shape[:2] != mask.shape[:2]:
            print(f"  [SHAPE MISMATCH] {img_path.name}: img {img.shape}, mask {mask.shape}")
            num_mismatch += 1
            continue

        uniq = np.unique(mask)
        if not set(uniq).issubset({0, 1, 2}):
            print(f"  [BAD LABEL VALUES] {mask_path.name}: {uniq}")
            bad_values += 1
            continue

        num_ok += 1

    print(f"  OK: {num_ok}, missing: {num_missing}, mismatch: {num_mismatch}, bad_values: {bad_values}")
    print()


if __name__ == "__main__":
    for split in ["train", "val"]:
        check_split(split)
