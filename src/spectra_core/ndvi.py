import numpy as np

def compute_ndvi(nir, red):
    """Compute NDVI from NIR and RED bands."""
    return (nir - red) / (nir + red + 1e-8)

def normalize01(arr):
    """Normalize array to 0-1 range."""
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)