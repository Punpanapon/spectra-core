import os
import click
from datetime import datetime
from pystac_client import Client
import stackstac
import rioxarray as rxr

def fetch_s2_s1_by_bbox_time(bbox, t0, t1, out_dir, max_items=2):
    """Fetch Sentinel-2 and Sentinel-1 data via STAC."""
    
    os.makedirs(out_dir, exist_ok=True)
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    print(f"Searching for data: bbox={bbox}, time={t0}/{t1}")
    
    # Search Sentinel-2 L2A
    s2_search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{t0}/{t1}",
        limit=max_items
    )
    
    s2_items = list(s2_search.items())
    if s2_items:
        print(f"Found {len(s2_items)} Sentinel-2 items")
        
        # Get B04 and B08
        for band, asset_key in [("B04", "red"), ("B08", "nir")]:
            try:
                stack = stackstac.stack(
                    s2_items,
                    assets=[asset_key],
                    bbox=bbox,
                    resolution=10
                )
                
                # Take first time slice and squeeze
                data = stack.isel(time=0).squeeze()
                
                # Save as GeoTIFF
                out_path = os.path.join(out_dir, f"S2_{band}.tif")
                data.rio.to_raster(out_path)
                print(f"Saved: {out_path}")
                
            except Exception as e:
                print(f"Error processing S2 {band}: {e}")
    
    # Search Sentinel-1 GRD
    try:
        s1_search = client.search(
            collections=["sentinel-1-grd"],
            bbox=bbox,
            datetime=f"{t0}/{t1}",
            limit=max_items
        )
        
        s1_items = list(s1_search.items())
        if s1_items:
            print(f"Found {len(s1_items)} Sentinel-1 items")
            
            # Get VV polarization
            stack = stackstac.stack(
                s1_items,
                assets=["vv"],
                bbox=bbox,
                resolution=10
            )
            
            # Take first time slice and squeeze
            data = stack.isel(time=0).squeeze()
            
            # Save as GeoTIFF
            out_path = os.path.join(out_dir, "S1.tif")
            data.rio.to_raster(out_path)
            print(f"Saved: {out_path}")
            
    except Exception as e:
        print(f"Error processing S1: {e}")

@click.command()
@click.option('--bbox', required=True, help='Bounding box: minx,miny,maxx,maxy')
@click.option('--t0', required=True, help='Start date: YYYY-MM-DD')
@click.option('--t1', required=True, help='End date: YYYY-MM-DD')
@click.option('--out', default='data', help='Output directory')
@click.option('--max-items', default=2, help='Max items per collection')
def main(bbox, t0, t1, out, max_items):
    """Fetch Sentinel data via STAC."""
    bbox_coords = [float(x) for x in bbox.split(',')]
    fetch_s2_s1_by_bbox_time(bbox_coords, t0, t1, out, max_items)

if __name__ == '__main__':
    main()