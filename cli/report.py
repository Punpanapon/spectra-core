#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra_core.report import generate_report

def main():
    parser = argparse.ArgumentParser(description="Generate SPECTRA report")
    parser.add_argument("--in", dest="input_dir", default="outputs/", help="Input directory")
    parser.add_argument("--open", action="store_true", help="Open report after generation")
    
    args = parser.parse_args()
    
    report_path = generate_report(args.input_dir)
    print(f"Report generated: {report_path}")
    
    if args.open:
        try:
            subprocess.run(["wslview", report_path], check=True)
        except:
            try:
                subprocess.run(["xdg-open", report_path], check=True)
            except:
                print(f"Could not open report automatically. Open {report_path} manually.")

if __name__ == "__main__":
    main()