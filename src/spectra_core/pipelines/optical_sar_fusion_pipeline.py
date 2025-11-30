# Discovery: backend processing is in src/spectra_core (pipeline.py/pipeline_change.py) and Streamlit entry is app/streamlit_app.py; current AI overlay runs spectra_ai.infer_unet.run_unet_on_efc_tile on EFC tiles.
"""
End-to-end optical+SAR fusion inference pipeline.

This module keeps the orchestration separate from the Streamlit UI. The default
tile_fetcher is a stub; wire it to your tiling/GEE export logic to provide
aligned tiles.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from spectra_core.data.optical_sar_alignment import (
    load_aligned_optical_sar_tiles,
    save_prob_raster,
)
from spectra_core.models.optical_sar_fusion import OpticalSarFusionModel, fallback_fusion
from spectra_core.models.optical_sar_wrappers import run_optical_unet_on_tile, run_sar_unet_on_tile
from spectra_core.ndvi import compute_ndvi

TileFetcher = Callable[[Any, Any, Any], Iterable[Dict[str, Any]]]


def _default_tile_fetcher(aoi, t0, t1) -> Iterable[Dict[str, Any]]:
    """
    Placeholder tile fetcher.

    Replace this with a function that yields dicts containing either:
      - {'tile_id': str, 's2_paths': {'B04': ..., 'B03': ..., 'B02': ..., 'B08': ...}, 's1_path': ...}
    or
      - {'tile_id': str, 's2_tile': np.ndarray, 's1_tile': np.ndarray, 'transform': Affine, 'crs': CRS}
    """
    raise NotImplementedError(
        "Plug in a tile_fetcher that uses your existing GEE export/tiling to produce aligned tiles."
    )


def _materialize_tile(
    tile: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Affine, CRS]:
    """Load arrays + metadata from a tile dict."""
    if "s2_tile" in tile and "s1_tile" in tile:
        s2_tile = np.asarray(tile["s2_tile"], dtype=np.float32)
        s1_tile = np.asarray(tile["s1_tile"], dtype=np.float32)
        transform = tile.get("transform", Affine.identity())
        crs = CRS.from_user_input(tile.get("crs", "EPSG:4326"))
    elif "s2_paths" in tile:
        s2_paths = tile["s2_paths"]
        s1_path = tile.get("s1_path")
        s2_tile, s1_tile, transform, crs = load_aligned_optical_sar_tiles(
            s2_paths, s1_path
        )
    else:
        raise ValueError("Tile dict must contain either in-memory tiles or 's2_paths'.")

    if s1_tile is None:
        raise ValueError("SAR tile is required for fusion.")
    assert s1_tile.shape[:2] == s2_tile.shape[:2], "Optical/SAR shapes must match."
    return s2_tile, s1_tile, transform, CRS.from_user_input(crs)


def _compute_extra_features(s2_tile: np.ndarray, s1_tile: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute optional helper rasters (NDVI, VV/VH ratio) if available."""
    extras: Dict[str, np.ndarray] = {}
    try:
        red = s2_tile[..., 0]
        nir = s2_tile[..., 3]
        extras["ndvi"] = compute_ndvi(nir, red).astype(np.float32)
    except Exception:
        pass

    if s1_tile is not None and s1_tile.shape[-1] >= 2:
        vv = s1_tile[..., 0]
        vh = s1_tile[..., 1]
        extras["vv_vh_ratio"] = (vv / (vh + 1e-6)).astype(np.float32)
    return extras


def run_optical_sar_fusion_for_aoi(
    aoi,
    t0,
    t1,
    fusion_model_path: str | None,
    tile_fetcher: Optional[TileFetcher] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
) -> str:
    """
    Run optical+SAR fusion over all tiles intersecting an AOI/time range.

    Returns
    -------
    str
        Directory containing the fused GeoTIFFs.

    Notes
    -----
    If fusion_model_path is None or fails to load, a simple fallback fusion
    (average of optical/SAR probabilities) is used so processing still completes.
    """
    fetcher = tile_fetcher or _default_tile_fetcher
    tiles = list(fetcher(aoi, t0, t1))
    if not tiles:
        raise RuntimeError("Tile fetcher returned no tiles to process.")

    out_dir = Path(
        output_dir
        or Path("outputs")
        / "optical_sar_fusion"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fusion_model = None
    if fusion_model_path:
        try:
            fusion_model = OpticalSarFusionModel(fusion_model_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[fusion_pipeline] Failed to load fusion model at {fusion_model_path}: {exc}")
    saved: List[str] = []

    for idx, tile in enumerate(tiles):
        tile_id = tile.get("tile_id") or f"tile_{idx}"
        s2_tile, s1_tile, transform, crs = _materialize_tile(tile)
        probs_opt = run_optical_unet_on_tile(s2_tile)
        probs_sar = run_sar_unet_on_tile(s1_tile, device=device)
        extras = _compute_extra_features(s2_tile, s1_tile)
        if fusion_model is not None:
            probs_fused = fusion_model.fuse(probs_opt, probs_sar, extras)
        else:
            probs_fused = fallback_fusion(probs_opt, probs_sar)
        out_path = out_dir / f"{tile_id}_fused.tif"
        save_prob_raster(probs_fused, transform, crs, str(out_path))
        saved.append(str(out_path))

    return str(out_dir)
