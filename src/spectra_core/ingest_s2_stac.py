import rasterio
from rasterio.windows import from_bounds
from pystac_client import Client
from datetime import datetime, timedelta

def search_s2(bbox, date, max_clouds=20):
    """Search for Sentinel-2 L2A item with lowest cloud cover."""
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    # Parse date and create search window
    search_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = (search_date - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (search_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Search for items
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_clouds}}
    )
    
    items = list(search.items())
    if not items:
        raise ValueError(f"No Sentinel-2 items found for bbox {bbox} on {date} with <{max_clouds}% clouds")
    
    # Find item with lowest cloud cover
    best_item = min(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    
    return {
        "item_id": best_item.id,
        "cloud_cover": best_item.properties.get("eo:cloud_cover", 0),
        "url_b04": best_item.assets["red"].href,
        "url_b08": best_item.assets["nir"].href
    }

def save_band_to_tif(asset_url, bbox, out_path):
    """Download and crop COG to bbox, save as GeoTIFF."""
    with rasterio.open(asset_url) as src:
        # Create window from bbox
        window = from_bounds(*bbox, src.transform)
        
        # Read data within window
        data = src.read(1, window=window)
        
        # Calculate transform for the window
        transform = rasterio.windows.transform(window, src.transform)
        
        # Write output
        profile = src.profile.copy()
        profile.update({
            'height': data.shape[0],
            'width': data.shape[1],
            'transform': transform,
            'count': 1
        })
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data, 1)