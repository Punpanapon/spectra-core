"""
Supported input modes for the fusion pipeline:
  - Upload/server file paths: run_pipeline() / run_pipeline_standard()/run_pipeline_windowed()
  - GEE arrays (Sentinel-2/Sentinel-1) can be handled via run_pipeline_arrays() without
    reading any on-disk GeoTIFF inputs (only writes outputs like metrics/PNG).

AI deforestation overlay intent:
  - S2 + S1: run optical UNet + SAR UNet, fuse with 0.5*p_opt + 0.5*p_sar.
  - S2 only: run optical UNet only (optical-only AI deforestation).
"""
import json
import logging
import os

import numpy as np
import rasterio
from PIL import Image

from .fusion import make_efc
from .geo import is_large_raster, load_tif_as_array, read_raster_windowed, resample_to_match
from .models.external_unet_wrappers import (
    get_wrapper_metadata,
    run_optical_unet_defmapping,
    run_sar_unet_sentinel,
)
from .ndvi import compute_ndvi
from .sar import to_db

logger = logging.getLogger(__name__)

def run_pipeline(red_path, nir_path, sar_c_path=None, sar_l_path=None, output_dir="outputs/"):
    """Run the complete SPECTRA fusion pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we need windowed processing
    use_windowed = is_large_raster(red_path) or is_large_raster(nir_path)
    
    if use_windowed:
        return run_pipeline_windowed(red_path, nir_path, sar_c_path, sar_l_path, output_dir)
    else:
        return run_pipeline_standard(red_path, nir_path, sar_c_path, sar_l_path, output_dir)

def run_pipeline_standard(red_path, nir_path, sar_c_path=None, sar_l_path=None, output_dir="outputs/"):
    """Standard pipeline for smaller rasters."""
    # Load reference (RED) for alignment
    with rasterio.open(red_path) as ref_ds:
        red = ref_ds.read(1)
        
    # Load and align NIR
    nir = load_tif_as_array(nir_path)
    with rasterio.open(red_path) as ref_ds:
        nir = resample_to_match(nir, ref_ds)
    
    # Compute NDVI
    ndvi = compute_ndvi(nir, red)
    
    # Process SAR data
    sar_c_db = None
    sar_l_db = None
    
    if sar_c_path:
        sar_c = load_tif_as_array(sar_c_path)
        with rasterio.open(red_path) as ref_ds:
            sar_c = resample_to_match(sar_c, ref_ds)
        sar_c_db = to_db(sar_c)
        
    if sar_l_path:
        sar_l = load_tif_as_array(sar_l_path)
        with rasterio.open(red_path) as ref_ds:
            sar_l = resample_to_match(sar_l, ref_ds)
        sar_l_db = to_db(sar_l)
    
    # Create EFC
    efc_path = os.path.join(output_dir, "efc.png")
    make_efc(efc_path, ndvi, sar_c_db, sar_l_db)
    
    # Generate metrics
    metrics = {
        "ndvi_min": float(ndvi.min()),
        "ndvi_max": float(ndvi.max()),
        "ndvi_mean": float(ndvi.mean()),
        "has_sar_c": sar_c_path is not None,
        "has_sar_l": sar_l_path is not None
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate summary
    summary = f"EFC generated: NDVI range [{metrics['ndvi_min']:.3f}, {metrics['ndvi_max']:.3f}], SAR bands: {int(metrics['has_sar_c']) + int(metrics['has_sar_l'])}"
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    
    return efc_path, metrics, summary

def run_pipeline_windowed(red_path, nir_path, sar_c_path=None, sar_l_path=None, output_dir="outputs/"):
    """Windowed pipeline for large rasters with downscaled output."""
    # Accumulate metrics across windows
    ndvi_values = []
    
    # Create downscaled quicklook (max 2048x2048)
    with rasterio.open(red_path) as src:
        scale_factor = max(src.width / 2048, src.height / 2048, 1.0)
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)
        
        # Read downscaled data
        red_small = src.read(1, out_shape=(new_height, new_width))
    
    with rasterio.open(nir_path) as src:
        nir_small = src.read(1, out_shape=(new_height, new_width))
    
    # Compute NDVI for quicklook
    ndvi_small = compute_ndvi(nir_small, red_small)
    ndvi_values.extend(ndvi_small.flatten())
    
    # Process SAR if available
    sar_c_small = None
    sar_l_small = None
    
    if sar_c_path:
        with rasterio.open(sar_c_path) as src:
            sar_c_raw = src.read(1, out_shape=(new_height, new_width))
            sar_c_small = to_db(sar_c_raw)
    
    if sar_l_path:
        with rasterio.open(sar_l_path) as src:
            sar_l_raw = src.read(1, out_shape=(new_height, new_width))
            sar_l_small = to_db(sar_l_raw)
    
    # Create EFC from downscaled data
    efc_path = os.path.join(output_dir, "efc.png")
    make_efc(efc_path, ndvi_small, sar_c_small, sar_l_small)
    
    # Generate metrics
    ndvi_array = np.array(ndvi_values)
    ndvi_array = ndvi_array[~np.isnan(ndvi_array)]  # Remove NaN values
    
    metrics = {
        "ndvi_min": float(ndvi_array.min()) if len(ndvi_array) > 0 else 0.0,
        "ndvi_max": float(ndvi_array.max()) if len(ndvi_array) > 0 else 0.0,
        "ndvi_mean": float(ndvi_array.mean()) if len(ndvi_array) > 0 else 0.0,
        "has_sar_c": sar_c_path is not None,
        "has_sar_l": sar_l_path is not None,
        "windowed_processing": True,
        "scale_factor": scale_factor
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Generate summary
    summary = f"EFC generated (windowed, {scale_factor:.1f}x downscaled): NDVI range [{metrics['ndvi_min']:.3f}, {metrics['ndvi_max']:.3f}], SAR bands: {int(metrics['has_sar_c']) + int(metrics['has_sar_l'])}"
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    
    return efc_path, metrics, summary


def run_pipeline_arrays(red_arr: np.ndarray, nir_arr: np.ndarray, sar_c_arr: np.ndarray | None = None,
                        sar_l_arr: np.ndarray | None = None, output_dir: str = "outputs/"):
    """
    Array-only version of the pipeline (no input file paths required).
    Computes NDVI and optional SAR dB, writes metrics/summary and EFC PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    red_arr = np.asarray(red_arr, dtype=np.float32)
    nir_arr = np.asarray(nir_arr, dtype=np.float32)
    h = min(red_arr.shape[0], nir_arr.shape[0])
    w = min(red_arr.shape[1], nir_arr.shape[1])
    red_arr = red_arr[:h, :w]
    nir_arr = nir_arr[:h, :w]

    ndvi = compute_ndvi(nir_arr, red_arr)
    sar_c_db = to_db(sar_c_arr[:h, :w]) if sar_c_arr is not None else None
    sar_l_db = to_db(sar_l_arr[:h, :w]) if sar_l_arr is not None else None

    efc_path = os.path.join(output_dir, "efc.png")
    make_efc(efc_path, ndvi, sar_c_db, sar_l_db)

    ndvi_clean = ndvi[~np.isnan(ndvi)]
    ndvi_min = float(ndvi_clean.min()) if ndvi_clean.size > 0 else 0.0
    ndvi_max = float(ndvi_clean.max()) if ndvi_clean.size > 0 else 0.0
    ndvi_mean = float(ndvi_clean.mean()) if ndvi_clean.size > 0 else 0.0

    metrics = {
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "ndvi_mean": ndvi_mean,
        "has_sar_c": sar_c_arr is not None,
        "has_sar_l": sar_l_arr is not None,
        "array_mode": True,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    summary = (
        f"EFC (array mode): NDVI range [{ndvi_min:.3f}, {ndvi_max:.3f}], "
        f"SAR bands: {int(sar_c_arr is not None) + int(sar_l_arr is not None)}"
    )
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)

    return efc_path, metrics, summary


def _align_bands_to_min_shape(arrays: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Crop all arrays to the smallest HxW among non-None entries."""
    valid = [a for a in arrays if a is not None]
    if not valid:
        return arrays
    h = min(a.shape[0] for a in valid)
    w = min(a.shape[1] for a in valid)
    aligned: list[np.ndarray | None] = []
    for arr in arrays:
        if arr is None:
            aligned.append(None)
        else:
            aligned.append(np.asarray(arr, dtype=np.float32)[:h, :w])
    return aligned


def run_optical_and_sar_unets_from_arrays(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    nir: np.ndarray,
    sar: np.ndarray | None,
) -> dict:
    """
    Run external U-Nets and return probabilities + metadata.

    Returns
    -------
    dict
        {
            'p_opt': HxW optical probabilities,
            'p_sar': HxW SAR probabilities (zeros if unavailable),
            'p_fused': HxW fused probabilities,
            'metadata': wrapper metadata (selected weights),
            'messages': list of status strings
        }
    """
    messages: list[str] = []
    red, green, blue, nir, sar = _align_bands_to_min_shape([red, green, blue, nir, sar])
    assert red is not None and green is not None and blue is not None and nir is not None  # for type checking
    h, w = red.shape

    sar_aligned = None
    if sar is not None:
        sar_arr = np.asarray(sar, dtype=np.float32)
        if sar_arr.ndim == 2:
            sar_arr = sar_arr[..., None]
        if sar_arr.shape[0] != h or sar_arr.shape[1] != w:
            try:
                from skimage.transform import resize

                sar_arr = resize(
                    sar_arr,
                    (h, w, sar_arr.shape[-1]),
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.float32)
                messages.append("Resampled SAR to match Sentinel-2 grid for fusion.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to resample SAR to S2 grid: %s", exc)
                messages.append("SAR shape mismatch; using optical-only fusion.")
                sar_arr = None
        sar_aligned = sar_arr

    p_opt = run_optical_unet_defmapping(red, green, blue, nir)
    if sar_aligned is None:
        p_sar = np.zeros_like(p_opt, dtype=np.float32)
        messages.append("SAR unavailable; using optical-only probabilities.")
    else:
        p_sar = run_sar_unet_sentinel(sar_aligned)
        if p_sar.shape != p_opt.shape:
            try:
                from skimage.transform import resize

                p_sar = resize(p_sar, p_opt.shape, preserve_range=True, anti_aliasing=False).astype(np.float32)
                messages.append("Resampled SAR probabilities to optical grid.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to align SAR probabilities; falling back to optical-only: %s", exc)
                p_sar = np.zeros_like(p_opt, dtype=np.float32)
                messages.append("SAR probabilities misaligned; using optical-only.")

    sar_used = sar_aligned is not None and not np.allclose(p_sar, 0.0)
    if sar_used:
        p_fused = 0.5 * p_opt.astype(np.float32) + 0.5 * p_sar.astype(np.float32)
    else:
        if sar_aligned is not None:
            messages.append("SAR model output is zero; falling back to optical-only fusion.")
        p_fused = p_opt.astype(np.float32)
    p_fused = np.clip(p_fused, 0.0, 1.0).astype(np.float32)

    return {
        "p_opt": p_opt.astype(np.float32),
        "p_sar": p_sar.astype(np.float32),
        "p_fused": p_fused,
        "metadata": get_wrapper_metadata(),
        "messages": messages,
    }
