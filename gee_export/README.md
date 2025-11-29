# SPECTRA Earth Engine exporter

Exports Sentinel-2, Sentinel-1, optional Dynamic World, and Hansen loss labels from Google Earth Engine into aligned GeoTIFFs under `spectra-core/data/`.

## Prerequisites
- Install `earthengine-api`, `requests`, and `tqdm` (e.g., `pip install earthengine-api requests tqdm`).
- Authenticate once: `earthengine authenticate`

## Usage
```
python gee_export/export_spectra.py --bbox "101.2,14.3,101.7,14.8" --t0 2024-01-01 --t1 2024-03-31 --mode auto --prefix "th_demo" --loss_years "2024"
python gee_export/export_spectra.py --bbox "101.2,14.3,101.7,14.8" --t0 2024-01-01 --t1 2024-03-31 --mode drive --gdrive_folder SPECTRA_GEE --prefix "th_drive"
```

Key flags:
- `--mode auto|download|drive`: auto switches to Drive when the AOI is larger than ~8000x8000 pixels.
- `--include_dynamicworld true` to add Dynamic World mode/probability.
- `--loss_years "2024,2025"` to emit a binary Hansen lossyear label.
- Override grid with `--crs`/`--scale` (defaults follow the Sentinel-2 grid, 10 m).
- If your Earth Engine account requires a Cloud Project, pass `--project YOUR_PROJECT` or set `EE_PROJECT` / `GOOGLE_CLOUD_PROJECT`.

Outputs land in `data/` (download mode) or the chosen Drive folder. All rasters share the same CRS, pixel size, and origin.

## Retrieving Drive exports
- Open Google Drive and find the folder you passed via `--gdrive_folder` (default `SPECTRA_GEE`).
- Each task writes a GeoTIFF; download them locally (unzip if Drive bundles them).
- For tiny AOIs you can re-run in `--mode download` to stream files directly into `spectra-core/data/`.

## Sentinel-2 notes
The exporter prefers `COPERNICUS/S2_SR_HARMONIZED` with cloud probability masking at 60%. If SR is unavailable (e.g., ingestion outage), it falls back to `COPERNICUS/S2_HARMONIZED` and logs the switch. All outputs are clipped to your bbox and unmasked to nodata (-9999 for float layers, 0 for labels) for consistent chipping.
