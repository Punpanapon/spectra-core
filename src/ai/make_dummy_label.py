import click
import numpy as np
from .paths import open_raster, save_tif

@click.command()
@click.option('--red', required=True, help='RED band path')
@click.option('--nir', required=True, help='NIR band path')
@click.option('--out', required=True, help='Output label path')
def main(red, nir, out):
    """Create dummy label from NDVI threshold."""
    
    # Load bands
    red_data = open_raster(red)
    nir_data = open_raster(nir)
    
    # Compute NDVI
    ndvi = (nir_data - red_data) / (nir_data + red_data + 1e-8)
    
    # Create dummy labels: 1 where NDVI < 0.3 (potential loss), 0 elsewhere
    labels = (ndvi < 0.3).astype(np.uint8)
    
    # Save with same profile as RED
    profile = {'dtype': 'uint8', 'count': 1}
    save_tif(out, labels, profile)
    
    print(f"Created dummy labels: {out}")

if __name__ == '__main__':
    main()