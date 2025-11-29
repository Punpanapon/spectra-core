import os
import click
from pystac_client import Client
import stackstac
import rioxarray
import rasterio
from shapely.geometry import box
from pyproj import Transformer

def fetch_sentinel_data(bbox, t0, t1, out_dir, max_items=3, cloud=30):
    """Fetch Sentinel-2 and Sentinel-1 data via STAC."""
    os.makedirs(out_dir, exist_ok=True)
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    minx, miny, maxx, maxy = bbox
    print(f"Searching STAC: bbox={bbox}, time={t0}/{t1}, cloud<={cloud}%")
    
    # Search Sentinel-2 L2A
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=[minx, miny, maxx, maxy],
        datetime=f"{t0}/{t1}",
        query={"eo:cloud_cover": {"lt": cloud}},
        max_items=max_items
    )
    items = list(search.get_items())
    
    # Retry with COGs collection if no items
    if not items:
        search = client.search(
            collections=["sentinel-2-l2a-cogs"],
            bbox=[minx, miny, maxx, maxy],
            datetime=f"{t0}/{t1}",
            query={"eo:cloud_cover": {"lt": cloud}},
            max_items=max_items
        )
        items = list(search.get_items())
    
    if not items:
        raise RuntimeError("No Sentinel-2 found in window/bbox. Try widening dates or increasing --cloud.")
    
    # Choose least-cloudy item
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    best_item = items[0]
    cloud_cover = best_item.properties.get("eo:cloud_cover", 0)
    print(f"Using S2 item: {best_item.id} (cloud: {cloud_cover:.1f}%)")
    
    # Determine EPSG
    epsg = best_item.properties.get("proj:epsg")
    if not epsg:
        with rasterio.open(best_item.assets["B04"].href) as src:
            epsg = src.crs.to_epsg()
    
    # Transform bbox to target EPSG
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    bbox_geom = box(minx, miny, maxx, maxy)
    coords = list(bbox_geom.exterior.coords)
    transformed_coords = [transformer.transform(x, y) for x, y in coords]
    xs, ys = zip(*transformed_coords)
    bounds_epsg = (min(xs), min(ys), max(xs), max(ys))
    
    try:
        # Build S2 stack with explicit EPSG + bounds
        s2_da = stackstac.stack(
            [best_item],
            assets=["B04", "B08"],
            epsg=int(epsg),
            bounds=bounds_epsg,
            resolution=10,
            chunksize=2048
        ).isel(time=0)
        s2_ds = s2_da.to_dataset("band")
        b04 = s2_ds["B04"].astype("float32")
        b08 = s2_ds["B08"].astype("float32")
    except AssertionError as e:
        if "out_bounds" in str(e):
            print("Hint: Try widening the bbox slightly if stackstac bounds error occurs.")
        raise
    
    # Write S2 files
    b04.rio.to_raster(os.path.join(out_dir, "S2_B04.tif"), compress="DEFLATE", tiled=True, predictor=2)
    b08.rio.to_raster(os.path.join(out_dir, "S2_B08.tif"), compress="DEFLATE", tiled=True, predictor=2)
    print(f"Saved: {os.path.join(out_dir, 'S2_B04.tif')} ({b04.shape})")
    print(f"Saved: {os.path.join(out_dir, 'S2_B08.tif')} ({b08.shape})")
    
    # Only after S2 succeeds, try S1
    s1_search = client.search(
        collections=["sentinel-1-grd"],
        bbox=[minx, miny, maxx, maxy],
        datetime=f"{t0}/{t1}",
        max_items=max_items
    )
    s1_items = list(s1_search.get_items())
    
    if s1_items:
        s1_item = s1_items[0]
        print(f"Using S1 item: {s1_item.id}")
        s1_da = stackstac.stack(
            [s1_item],
            assets=["vv"],
            epsg=int(epsg),
            bounds=bounds_epsg,
            resolution=10,
            chunksize=2048
        ).isel(time=0)
        s1 = s1_da.to_dataset("band")["vv"].astype("float32")
        s1m = s1.rio.reproject_match(b04)
        s1m.rio.to_raster(os.path.join(out_dir, "S1_VV.tif"), compress="DEFLATE", tiled=True, predictor=2)
        print(f"Saved: {os.path.join(out_dir, 'S1_VV.tif')} ({s1m.shape})")
    else:
        print("No Sentinel-1 in window or skipping because S2 failed.")

@click.command()
@click.option('--bbox', required=True, help='Bounding box: minx,miny,maxx,maxy')
@click.option('--t0', required=True, help='Start date: YYYY-MM-DD')
@click.option('--t1', required=True, help='End date: YYYY-MM-DD')
@click.option('--out', default='data', help='Output directory')
@click.option('--max-items', default=3, help='Max items per collection')
@click.option('--cloud', default=30, help='Max cloud cover percentage')
def main(bbox, t0, t1, out, max_items, cloud):
    """Fetch Sentinel data via STAC."""
    bbox_coords = [float(x) for x in bbox.split(',')]
    fetch_sentinel_data(bbox_coords, t0, t1, out, max_items, cloud)

if __name__ == '__main__':
    main()