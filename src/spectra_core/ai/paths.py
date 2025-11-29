from __future__ import annotations
from typing import Optional, Tuple, List, Union
import numpy as np
import xarray as xr
import rioxarray as rxr

# open a single-band or multi-band raster as xr.DataArray with .rio accessor
def open_da(path: str) -> xr.DataArray:
    da = rxr.open_rasterio(path, masked=True)  # returns xr.DataArray
    # squeeze a length-1 'band' dim to 2D (y,x)
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.isel(band=0).drop_vars("band")
    return da

def reproject_match(src_da: xr.DataArray, ref_da: xr.DataArray, resampling: str = "nearest") -> xr.DataArray:
    return src_da.rio.reproject_match(ref_da, resampling=resampling)

def ensure_common_grid(red_da: xr.DataArray, *others: xr.DataArray) -> Tuple[xr.DataArray, List[xr.DataArray]]:
    aligned: List[xr.DataArray] = []
    for o in others:
        if (o.rio.crs != red_da.rio.crs) or (o.rio.transform() != red_da.rio.transform()):
            o = o.rio.reproject_match(red_da)
        aligned.append(o)
    return red_da, aligned

def _to_da(arr: Union[xr.DataArray, np.ndarray], like: Optional[xr.DataArray]) -> xr.DataArray:
    if isinstance(arr, xr.DataArray):
        return arr
    if like is None:
        raise ValueError("When saving a numpy array, provide 'like' DataArray for georeferencing.")
    # make DA with like's coords/attrs
    da = xr.DataArray(arr, coords=like.coords, dims=like.dims, attrs=like.attrs)
    da = da.rio.write_crs(like.rio.crs, inplace=False)
    return da

def save_tif(path: str, da_or_np: Union[xr.DataArray, np.ndarray], like_da: Optional[xr.DataArray] = None) -> None:
    da = _to_da(da_or_np, like_da)
    if "nodata" not in da.attrs or da.attrs.get("_FillValue") is None:
        da = da.rio.write_nodata(0, inplace=False)
    da.rio.to_raster(path, compress="DEFLATE", tiled=True, predictor=2)