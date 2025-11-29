import os
import rasterio
import rioxarray as rxr
import xarray as xr
import s3fs

def open_raster(path):
    """Open raster from local path or S3 URL, return xarray.DataArray with CRS/transform."""
    if path.startswith('s3://'):
        fs = s3fs.S3FileSystem()
        with fs.open(path, 'rb') as f:
            return rxr.open_rasterio(f, chunks=True).squeeze('band', drop=True)
    else:
        return rxr.open_rasterio(path, chunks=True).squeeze('band', drop=True)

def save_tif(path, array, profile):
    """Save xarray.DataArray as GeoTIFF to local path or S3."""
    if path.startswith('s3://'):
        fs = s3fs.S3FileSystem()
        with fs.open(path, 'wb') as f:
            array.rio.to_raster(f, **profile)
    else:
        array.rio.to_raster(path, **profile)

def parse_path(path_str):
    """Parse path string into type and components."""
    if path_str.startswith('s3://'):
        return 's3', path_str
    elif os.path.exists(path_str):
        return 'local', path_str
    else:
        return 'unknown', path_str