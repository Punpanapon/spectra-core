import os
import json
import numpy as np
import click
from PIL import Image
from tqdm import tqdm
from .paths import open_da, ensure_common_grid

def compute_ndvi(nir, red):
    """Compute NDVI with safe division."""
    return (nir - red) / (nir + red + 1e-8)

def make_chips(red_path, nir_path, label_path, out_dir, sar_path=None, 
               chip_size=256, stride=256, nodata_frac_max=0.2, min_label_frac=0.01, 
               save_png_previews=True):
    """Generate training chips from raster data."""
    
    # Create output directories
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)
    if save_png_previews:
        os.makedirs(f"{out_dir}/thumbnails", exist_ok=True)
    
    print("Loading rasters...")
    
    # Load rasters
    red_da = open_da(red_path)
    nir_da = open_da(nir_path)
    
    # Load SAR if provided
    if sar_path:
        sar_da = open_da(sar_path)
        red_da, aligned = ensure_common_grid(red_da, nir_da, sar_da)
        nir_da, sar_da = aligned
    else:
        red_da, aligned = ensure_common_grid(red_da, nir_da)
        nir_da = aligned[0]
        sar_da = None
    
    # Load labels
    label_da = open_da(label_path)
    red_da, aligned = ensure_common_grid(red_da, label_da)
    label_da = aligned[0]
    
    # Compute NDVI
    ndvi = compute_ndvi(nir_da.values, red_da.values)
    
    # Stack features
    features = [red_da.values, nir_da.values, ndvi]
    if sar_da is not None:
        features.append(sar_da.values)
    
    feature_stack = np.stack(features, axis=-1).astype(np.float32)
    labels = label_da.values.astype(np.uint8)
    
    # Normalize features to 0-1
    for i in range(feature_stack.shape[-1]):
        band = feature_stack[:, :, i]
        p2, p98 = np.nanpercentile(band, [2, 98])
        feature_stack[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    # Generate chips
    h, w = feature_stack.shape[:2]
    chip_count = 0
    
    print(f"Generating chips from {h}x{w} raster...")
    
    for y in tqdm(range(0, h - chip_size + 1, stride)):
        for x in range(0, w - chip_size + 1, stride):
            # Extract chip
            feat_chip = feature_stack[y:y+chip_size, x:x+chip_size]
            label_chip = labels[y:y+chip_size, x:x+chip_size]
            
            # Check nodata fraction
            nodata_frac = np.isnan(feat_chip).sum() / feat_chip.size
            if nodata_frac > nodata_frac_max:
                continue
            
            # Check label coverage
            label_frac = (label_chip == 1).sum() / label_chip.size
            if label_frac < min_label_frac:
                continue
            
            # Replace NaN with 0
            feat_chip = np.nan_to_num(feat_chip, 0)
            
            # Save chip
            chip_id = f"chip_{chip_count:06d}"
            
            np.save(f"{out_dir}/images/{chip_id}.npy", feat_chip)
            np.save(f"{out_dir}/masks/{chip_id}.npy", label_chip)
            
            # Save preview PNG
            if save_png_previews:
                # Create RGB preview (use first 3 channels)
                rgb = feat_chip[:, :, :3]
                rgb_8bit = (rgb * 255).astype(np.uint8)
                Image.fromarray(rgb_8bit).save(f"{out_dir}/thumbnails/{chip_id}.png")
            
            chip_count += 1
    
    # Save manifest
    manifest = {
        "chip_count": chip_count,
        "chip_size": chip_size,
        "channels": len(features),
        "channel_names": ["red", "nir", "ndvi"] + (["sar"] if sar_da else []),
        "grid_info": {
            "crs": str(red_da.rio.crs),
            "transform": list(red_da.rio.transform())[:6]
        }
    }
    
    with open(f"{out_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated {chip_count} chips in {out_dir}")
    return chip_count

@click.command()
@click.option('--red', required=True, help='RED band path')
@click.option('--nir', required=True, help='NIR band path')
@click.option('--label', required=True, help='Label raster path')
@click.option('--out', required=True, help='Output directory')
@click.option('--sar', help='SAR band path (optional)')
@click.option('--chip-size', default=256, help='Chip size in pixels')
@click.option('--stride', default=256, help='Stride for chip extraction')
@click.option('--nodata-frac-max', default=0.2, help='Max nodata fraction')
@click.option('--min-label-frac', default=0.01, help='Min label fraction')
def main(red, nir, label, out, sar, chip_size, stride, nodata_frac_max, min_label_frac):
    """Generate training chips from raster data."""
    make_chips(red, nir, label, out, sar, chip_size, stride, nodata_frac_max, min_label_frac)

if __name__ == '__main__':
    main()