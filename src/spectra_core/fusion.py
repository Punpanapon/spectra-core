import numpy as np
from PIL import Image
from .ndvi import normalize01

def make_efc(rgb_out_png, ndvi, sar_c=None, sar_l=None):
    """Create Enhanced Forest Composite and save as PNG."""
    # R channel: 1 - NDVI
    r_channel = normalize01(1 - ndvi)
    
    # G channel: NDVI
    g_channel = normalize01(ndvi)
    
    # B channel: SAR dB normalized
    if sar_c is not None and sar_l is not None:
        # Average both bands
        b_channel = normalize01((sar_c + sar_l) / 2)
    elif sar_c is not None:
        b_channel = normalize01(sar_c)
    elif sar_l is not None:
        b_channel = normalize01(sar_l)
    else:
        b_channel = np.zeros_like(ndvi)
    
    # Stack and convert to 8-bit
    rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
    rgb_8bit = (rgb * 255).astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(rgb_8bit).save(rgb_out_png)