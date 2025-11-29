import json
import os
import rasterio
import numpy as np
from PIL import Image
from .geo import load_tif_as_array, resample_to_match, is_large_raster, read_raster_windowed
from .ndvi import compute_ndvi
from .sar import to_db, normalize01
from .fusion import make_efc

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