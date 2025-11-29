import click
import numpy as np
from skimage import filters, morphology
from scipy import ndimage
from .paths import open_da, save_tif

def compute_ndvi(nir, red, eps=1e-8):
    """Compute NDVI with safe division."""
    return (nir - red) / (nir + red + eps)

@click.command()
@click.option('--red', required=True, help='RED band path')
@click.option('--nir', required=True, help='NIR band path')
@click.option('--out', required=True, help='Output label path')
@click.option('--method', type=click.Choice(['otsu', 'percentile']), default='otsu', help='Thresholding method')
@click.option('--percentile', default=15, help='Percentile threshold (0-100)')
@click.option('--smooth', default=1, help='Gaussian sigma (0 disables)')
@click.option('--invert', is_flag=True, help='Invert mask')
def main(red, nir, out, method, percentile, smooth, invert):
    """Create dummy labels from NDVI thresholding."""
    
    print(f"Loading {red} and {nir}...")
    
    # Load rasters
    red_da = open_da(red)
    nir_da = open_da(nir)
    
    # Align NIR to RED grid
    if red_da.rio.crs != nir_da.rio.crs or red_da.shape != nir_da.shape:
        nir_da = nir_da.rio.reproject_match(red_da)
    
    # Compute NDVI
    ndvi = compute_ndvi(nir_da.values, red_da.values)
    
    # Optional smoothing
    if smooth > 0:
        ndvi = ndimage.gaussian_filter(ndvi, sigma=smooth)
    
    # Threshold
    valid_mask = ~np.isnan(ndvi)
    valid_ndvi = ndvi[valid_mask]
    
    if method == 'otsu':
        threshold = filters.threshold_otsu(valid_ndvi)
    else:  # percentile
        threshold = np.nanpercentile(ndvi, percentile)
    
    # Create mask
    mask = ndvi < threshold
    
    # Invert if requested
    if invert:
        mask = ~mask
    
    # Convert to uint8 labels
    labels = mask.astype(np.uint8)
    
    # Set invalid areas to 0
    labels[~valid_mask] = 0
    
    # Save with RED profile
    save_tif(out, labels, red_da)
    
    # Summary
    pos_percent = (labels == 1).sum() / labels.size * 100
    print(f"Threshold: {threshold:.3f}")
    print(f"Positive pixels: {pos_percent:.1f}%")
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()