#!/usr/bin/env python3
"""
Generate weak labels (forest / deforested / other) for each EFC tile using Hansen GFC.

Reads tile metadata from data/efc_tiles/tile_metadata.csv, fetches a label mask from
Earth Engine for each tile footprint, and writes PNG masks aligned to the saved tiles.
"""

from __future__ import annotations

import csv
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import ee
import numpy as np
from PIL import Image

# Resolve project root (spectra-core)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "efc_tiles" / "tile_metadata.csv"
GFC_ASSET_ID = "UMD/hansen/global_forest_change_2024_v1_12"
CREDENTIALS_PATH = Path.home() / ".config" / "earthengine" / "credentials"


def _load_project_from_credentials_file() -> str | None:
    """Try to read a default project from ~/.config/earthengine/credentials."""
    if not CREDENTIALS_PATH.exists():
        return None
    try:
        with CREDENTIALS_PATH.open() as f:
            data = json.load(f)
        return data.get("project_id") or data.get("quota_project_id")
    except Exception:
        return None


def _resolve_project(cli_project: str | None) -> str | None:
    """Resolve project precedence: CLI arg -> env vars -> credentials file."""
    if cli_project:
        return cli_project
    return (
        os.getenv("EE_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("CLOUDSDK_CORE_PROJECT")
        or _load_project_from_credentials_file()
    )


def init_earth_engine(cli_project: str | None) -> None:
    """
    Initialize Earth Engine using existing environment / credentials.

    Project resolution order:
      1) --project CLI argument
      2) EE_PROJECT / GOOGLE_CLOUD_PROJECT / CLOUDSDK_CORE_PROJECT env
      3) ~/.config/earthengine/credentials project_id/quota_project_id
    """
    project = _resolve_project(cli_project)
    service_account = os.getenv("EE_SERVICE_ACCOUNT")
    private_key = os.getenv("EE_PRIVATE_KEY")

    if project is None:
        raise RuntimeError(
            "No Earth Engine project ID found. Set --project or EE_PROJECT / "
            "GOOGLE_CLOUD_PROJECT / CLOUDSDK_CORE_PROJECT."
        )

    try:
        if service_account and private_key:
            # Replace escaped newlines so single-line keys work.
            private_key_material = private_key.replace("\\n", "\n")
            creds = ee.ServiceAccountCredentials(
                service_account, key_data=private_key_material
            )
            ee.Initialize(credentials=creds, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001
        hint = (
            "Check EE_SERVICE_ACCOUNT / EE_PRIVATE_KEY and project settings."
            if service_account and private_key
            else "Set EE_PROJECT or GOOGLE_CLOUD_PROJECT."
        )
        raise RuntimeError(
            f"Failed to initialize Earth Engine. {hint} Original error: {exc}"
        ) from exc


def build_label_image() -> ee.Image:
    """Construct the 0/1/2 label image from Hansen GFC."""
    gfc = ee.Image(GFC_ASSET_ID)
    treecover = gfc.select("treecover2000")
    lossyear = gfc.select("lossyear")
    forest = treecover.gt(30)
    loss = lossyear.gt(0)
    forest_now = forest.And(loss.Not())
    deforested = forest.And(loss)
    label_img = (
        deforested.multiply(2)
        .add(forest_now.rename("forest"))
        .rename("label")
    )
    return label_img


def read_metadata_rows() -> list[Dict[str, str]]:
    """Read tile metadata CSV into a list of dicts."""
    if not METADATA_PATH.exists():
        print(f"[WARN] Metadata file not found: {METADATA_PATH}")
        return []
    with METADATA_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in ("", None) else None
    except (TypeError, ValueError):
        return None


def fetch_label_array(
    label_img: ee.Image,
    lat: float,
    lon: float,
    patch_size_m: float,
    target_shape: Tuple[int, int],
) -> np.ndarray | None:
    """
    Sample the label image over the requested region and return a NumPy array.
    Resamples to the target shape if the EE output differs.
    """
    center = ee.Geometry.Point([lon, lat])
    half_size = patch_size_m / 2.0
    region = center.buffer(half_size).bounds()

    # Derive approximate pixel scale (meters per pixel) from the saved tile size.
    height, width = target_shape
    pixel_scale = patch_size_m / max(height, width)
    pixel_scale = max(pixel_scale, 1)  # Avoid zero/negative scale requests.

    # Resample is omitted to avoid unsupported interpolation modes in some EE environments.
    scaled_img = label_img.reproject(
        crs="EPSG:4326",
        scale=pixel_scale,
    )
    rect = scaled_img.sampleRectangle(
        region=region,
        defaultValue=0,
    ).getInfo()

    props = rect.get("properties", rect)
    arr = props.get("label")
    if arr is None:
        return None

    mask_arr = np.array(arr)
    if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
        mask_arr = mask_arr[..., 0]
    if mask_arr.ndim != 2:
        return None

    mask_arr = mask_arr.astype(np.uint8)
    if mask_arr.shape != (height, width):
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
        mask_arr = np.array(mask_img, dtype=np.uint8)

    return mask_arr


def process_tile(row: Dict[str, str], label_img: ee.Image) -> None:
    """Process a single metadata row to generate a mask."""
    tile_id = (row.get("tile_id") or "").strip()
    split = (row.get("split") or "").strip()
    lat = parse_float(row.get("lat"))
    lon = parse_float(row.get("lon"))
    patch_size_m = parse_float(row.get("patch_size_m"))

    if not tile_id or split not in {"train", "val"}:
        print(f"[SKIP] Invalid row (tile_id or split missing): {row}")
        return
    if lat is None or lon is None or patch_size_m is None:
        print(f"[SKIP] Missing lat/lon/patch_size for tile {tile_id}")
        return

    image_path = PROJECT_ROOT / "data" / "efc_tiles" / split / "images" / f"{tile_id}.png"
    mask_dir = PROJECT_ROOT / "data" / "efc_tiles" / split / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{tile_id}_mask.png"

    if mask_path.exists():
        print(f"[SKIP] Mask already exists: {mask_path}")
        return
    if not image_path.exists():
        print(f"[WARN] Image not found for tile {tile_id}: {image_path}")
        return

    with Image.open(image_path) as img:
        width, height = img.size

    mask_arr = fetch_label_array(
        label_img=label_img,
        lat=lat,
        lon=lon,
        patch_size_m=patch_size_m,
        target_shape=(height, width),
    )
    if mask_arr is None:
        print(f"[WARN] No mask returned for tile {tile_id}")
        return

    Image.fromarray(mask_arr, mode="L").save(mask_path)
    print(f"[OK] {tile_id} ({split}) -> {mask_path}")


def main() -> None:
    """Entry point for generating masks for all tiles in tile_metadata.csv."""
    parser = argparse.ArgumentParser(description="Generate weak-label masks from Hansen GFC.")
    parser.add_argument(
        "--project",
        help="GEE project ID (fallback to EE_PROJECT/GOOGLE_CLOUD_PROJECT/CLOUDSDK_CORE_PROJECT or credentials file)",
    )
    args = parser.parse_args()

    init_earth_engine(args.project)
    rows = read_metadata_rows()
    if not rows:
        print("[INFO] No metadata rows to process.")
        return

    label_img = build_label_image()
    for row in rows:
        try:
            process_tile(row, label_img)
        except Exception as exc:  # noqa: BLE001
            tile_id = row.get("tile_id") or "unknown"
            split = row.get("split") or "?"
            print(f"[ERROR] Failed for {tile_id} ({split}): {exc}")


if __name__ == "__main__":
    sys.setrecursionlimit(2000)
    main()
