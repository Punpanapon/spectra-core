import rasterio
import rioxarray as rxr
import numpy as np
from rasterio.windows import Window

def load_tif_as_array(path):
    """Load GeoTIFF as numpy array."""
    with rasterio.open(path) as src:
        return src.read(1)

def save_array_as_tif(arr, like_path, out_path):
    """Save array as GeoTIFF using reference file metadata."""
    with rasterio.open(like_path) as src:
        profile = src.profile
        profile.update(dtype=arr.dtype, count=1)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(arr, 1)

def reproject_match(src_path, ref_path, out_path):
    """Reproject source to match reference CRS and resolution."""
    src_da = rxr.open_rasterio(src_path, chunks=True)
    ref_da = rxr.open_rasterio(ref_path, chunks=True)
    reprojected = src_da.rio.reproject_match(ref_da)
    reprojected.rio.to_raster(out_path)

def resample_to_match(arr, ref_ds):
    """Resample array to match reference dataset dimensions."""
    from skimage.transform import resize
    return resize(arr, (ref_ds.height, ref_ds.width), preserve_range=True)

def read_raster_windowed(path, block_size=1024):
    """Yield windowed chunks of raster data."""
    with rasterio.open(path) as src:
        for ji, window in src.block_windows(1):
            # Limit window size
            if window.width > block_size or window.height > block_size:
                # Create smaller windows
                for row in range(window.row_off, window.row_off + window.height, block_size):
                    for col in range(window.col_off, window.col_off + window.width, block_size):
                        h = min(block_size, window.row_off + window.height - row)
                        w = min(block_size, window.col_off + window.width - col)
                        sub_window = Window(col, row, w, h)
                        data = src.read(1, window=sub_window)
                        transform = rasterio.windows.transform(sub_window, src.transform)
                        bounds = rasterio.windows.bounds(sub_window, src.transform)
                        yield data, transform, bounds
            else:
                data = src.read(1, window=window)
                transform = rasterio.windows.transform(window, src.transform)
                bounds = rasterio.windows.bounds(window, src.transform)
                yield data, transform, bounds

def is_large_raster(path, threshold_pixels=12000*12000):
    """Check if raster exceeds size threshold."""
    with rasterio.open(path) as src:
        return src.width * src.height > threshold_pixels