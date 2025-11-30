# Optical + SAR Fusion (Deforestation)

This branch adds a late-fusion head that blends:
- Optical UNet (Sentinel-2 B4/B3/B2/B8) from `UNet-defmapping`
- SAR UNet (Sentinel-1) from `unet-sentinel`
- A logistic regression fusion head trained on per-pixel probabilities (+optional NDVI, VV/VH ratio)

## Where things live
- Backend pipelines: `src/spectra_core` (existing EFC and change detection live here).
- Streamlit entry: `app/streamlit_app.py` (now exposes an “AI Deforestation (Optical + SAR Fusion)” overlay).
- New modules: `src/spectra_core/models/optical_sar_wrappers.py`, `src/spectra_core/data/optical_sar_alignment.py`, `src/spectra_core/models/optical_sar_fusion.py`, `src/spectra_core/pipelines/optical_sar_fusion_pipeline.py`.

## Training the fusion head
1) Prepare a manifest CSV with columns:
   - `probs_opt`: path to optical UNet probability raster (.tif or .npy)
   - `probs_sar`: path to SAR UNet probability raster
   - `label`: binary ground-truth raster
   - Optional: `ndvi`, `vv_vh_ratio`, `tile_id`
2) Run:
   ```bash
   python scripts/train_optical_sar_fusion.py \
     --manifest data/fusion_manifest.csv \
     --model-out models/optical_sar_fusion.joblib \
     --max-samples 200000 --class-balance
   ```
   The script samples pixels (balanced if requested), trains `LogisticRegression`, prints precision/recall/F1/IoU, and saves the joblib (stores the model + feature names).

## Running inference
- Provide weight/stat paths via env:
  - `SPECTRA_OPTICAL_UNET_WEIGHTS`, `SPECTRA_OPTICAL_BANDS_THIRD`, `SPECTRA_OPTICAL_BANDS_NIN`
  - `SPECTRA_SAR_UNET_WEIGHTS`
- Fusion head path resolution (for Streamlit overlay and helpers):
  1) `st.secrets["ai"]["fusion_model_path"]`
  2) `SPECTRA_FUSION_MODEL_PATH`
  If nothing is configured, the app uses a built-in simple fusion (average of optical/SAR probabilities) so the overlay still works.
- Call the pipeline:
  ```python
  from spectra_core.pipelines.optical_sar_fusion_pipeline import run_optical_sar_fusion_for_aoi

  # tile_fetcher should yield aligned tiles; see docstring for accepted keys.
  out_dir = run_optical_sar_fusion_for_aoi(
      aoi=my_geojson,
      t0="2024-01-01",
      t1="2024-02-01",
      fusion_model_path="models/optical_sar_fusion.joblib",
      tile_fetcher=my_tile_fetcher,
      device="cuda",
  )
  print("Fused rasters:", out_dir)
  ```
- Each tile writes a GeoTIFF probability map aligned to the input grid.

## Validation metrics to monitor
- Pixel-wise Precision/Recall/F1 and IoU (Jaccard) on a held-out split.
- Compare fusion vs. single-modality baselines (optical-only, SAR-only).
- Inspect calibration curves if probabilities are used as heatmaps.
