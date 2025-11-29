#!/usr/bin/env python3
import argparse
import os
from ..ingest_s2_stac import search_s2, save_band_to_tif

def main():
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 B04/B08 bands via STAC")
    parser.add_argument("--bbox", required=True, help="Bounding box: minx,miny,maxx,maxy (WGS84)")
    parser.add_argument("--date", required=True, help="Date: YYYY-MM-DD")
    parser.add_argument("--max-clouds", type=int, default=20, help="Maximum cloud cover percentage")
    parser.add_argument("--out", default="data/", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse bbox
    bbox = [float(x) for x in args.bbox.split(",")]
    if len(bbox) != 4:
        raise ValueError("Bbox must have 4 values: minx,miny,maxx,maxy")
    
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Searching for Sentinel-2 data: bbox={bbox}, date={args.date}, max_clouds={args.max_clouds}%")
    
    # Search for best item
    result = search_s2(bbox, args.date, args.max_clouds)
    print(f"Found item: {result['item_id']} (cloud cover: {result['cloud_cover']:.1f}%)")
    
    # Download bands
    b04_path = os.path.join(args.out, "S2_B04.tif")
    b08_path = os.path.join(args.out, "S2_B08.tif")
    
    print("Downloading B04 (RED)...")
    save_band_to_tif(result["url_b04"], bbox, b04_path)
    
    print("Downloading B08 (NIR)...")
    save_band_to_tif(result["url_b08"], bbox, b08_path)
    
    print(f"✅ Saved files:")
    print(f"   RED: {b04_path}")
    print(f"   NIR: {b08_path}")

if __name__ == "__main__":
    main()