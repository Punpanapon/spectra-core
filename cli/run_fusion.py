#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra_core.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run SPECTRA fusion pipeline")
    parser.add_argument("--red", required=True, help="Path to RED band GeoTIFF")
    parser.add_argument("--nir", required=True, help="Path to NIR band GeoTIFF")
    parser.add_argument("--sar_c", help="Path to C-band SAR GeoTIFF")
    parser.add_argument("--sar_l", help="Path to L-band SAR GeoTIFF")
    parser.add_argument("--out", default="outputs/", help="Output directory")
    
    args = parser.parse_args()
    
    efc_path, metrics, summary = run_pipeline(
        args.red, args.nir, args.sar_c, args.sar_l, args.out
    )
    
    print(f"EFC saved to: {efc_path}")
    print(f"Summary: {summary}")
    print(f"Metrics: {os.path.join(args.out, 'metrics.json')}")

if __name__ == "__main__":
    main()