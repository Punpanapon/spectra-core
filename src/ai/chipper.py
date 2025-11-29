import os
import json
import numpy as np
import click
from tqdm import tqdm
from .paths import open_raster

def compute_ndvi(nir, red):
    """Compute NDVI from NIR and RED bands."""
    return (nir - red) / (nir + red + 1e-8)

def normalize_band(band):
    """Normalize band to 0-1 range."""
    band_min, band_max = np.nanpercentile(band, [2, 98])
    return np.clip((band - band_min) / (band_max - band_min + 1e-8), 0, 1)

def make_chips(red_path, nir_path, sar_path, label_path, out_dir, 
               chip_size=256, stride=256, nodata_thresh=0.2, min_foreground_frac=0.01):
    """Generate training chips from raster data."""
    
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)
    
    # Load and align rasters
    print("Loading rasters...")
    red = open_raster(red_path)
    nir = open_raster(nir_path)
    
    # Reproject NIR to match RED
    nir = nir.rio.reproject_match(red)
    
    # Compute NDVI
    ndvi = compute_ndvi(nir.values, red.values)
    
    # Stack features
    features = [normalize_band(red.values), normalize_band(nir.values), normalize_band(ndvi)]
    
    # Add SAR if provided
    if sar_path:
        sar = open_raster(sar_path).rio.reproject_match(red)
        features.append(normalize_band(sar.values))
    
    feature_stack = np.stack(features, axis=-1).astype(np.float32)
    
    # Load labels
    labels = open_raster(label_path).rio.reproject_match(red).values.astype(np.uint8)
    
    # Generate chips
    h, w = feature_stack.shape[:2]
    chip_count = 0
    
    print(f"Generating chips from {h}x{w} raster...")
    
    for y in tqdm(range(0, h - chip_size + 1, stride)):
        for x in range(0, w - chip_size + 1, stride):
            # Extract chip
            feat_chip = feature_stack[y:y+chip_size, x:x+chip_size]
            label_chip = labels[y:y+chip_size, x:x+chip_size]
            
            # Check nodata threshold
            nodata_frac = np.isnan(feat_chip).sum() / feat_chip.size
            if nodata_frac > nodata_thresh:
                continue
            
            # Check foreground fraction
            foreground_frac = (label_chip == 1).sum() / label_chip.size
            if foreground_frac < min_foreground_frac:
                continue
            
            # Save chip
            chip_id = f"chip_{chip_count:06d}"
            
            # Replace NaN with 0
            feat_chip = np.nan_to_num(feat_chip, 0)
            
            np.save(f"{out_dir}/images/{chip_id}.npy", feat_chip)
            np.save(f"{out_dir}/masks/{chip_id}.npy", label_chip)
            
            # Save metadata
            metadata = {
                "chip_id": chip_id,
                "bounds": [x, y, x + chip_size, y + chip_size],
                "foreground_frac": float(foreground_frac),
                "channels": len(features)
            }
            
            with open(f"{out_dir}/images/{chip_id}.json", 'w') as f:
                json.dump(metadata, f)
            
            chip_count += 1
    
    print(f"Generated {chip_count} chips in {out_dir}")
    return chip_count

@click.command()
@click.option('--red', required=True, help='RED band path')
@click.option('--nir', required=True, help='NIR band path')
@click.option('--sar', help='SAR band path (optional)')
@click.option('--label', required=True, help='Label raster path')
@click.option('--out', required=True, help='Output directory')
@click.option('--chip-size', default=256, help='Chip size in pixels')
@click.option('--stride', default=256, help='Stride for chip extraction')
@click.option('--nodata-thresh', default=0.2, help='Max nodata fraction')
@click.option('--min-foreground-frac', default=0.01, help='Min foreground fraction')
def main(red, nir, sar, label, out, chip_size, stride, nodata_thresh, min_foreground_frac):
    """Generate training chips from raster data."""
    make_chips(red, nir, sar, label, out, chip_size, stride, nodata_thresh, min_foreground_frac)

if __name__ == '__main__':
    main()