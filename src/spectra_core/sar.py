import numpy as np
from scipy.ndimage import uniform_filter

def to_db(arr, eps=1e-6):
    """Convert amplitude to dB."""
    return 10 * np.log10(np.maximum(arr, eps))

def lee_filter3x3(arr):
    """Simple 3x3 box filter as placeholder for Lee filter."""
    return uniform_filter(arr, size=3)

def normalize01(arr):
    """Normalize array to 0-1 range."""
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)