# SPECTRA Streamlit Web App

Interactive web interface for generating Enhanced Forest Composite (EFC) visualizations.

## Features

- **Large File Support**: Upload limits raised to 2GB
- **Dual Input Modes**: Upload files or use server files
- **Windowed Processing**: Automatic handling of large rasters (>12k x 12k pixels)
- **Memory Management**: Auto-cleanup of temporary uploads
- **Progress Tracking**: Real-time processing status
- **File Size Warnings**: Alerts for large files with memory estimates

## How to Use

### Input Mode Selection
Choose between:
- **Upload files**: Upload GeoTIFFs directly (up to 2GB each)
- **Use server files**: Reference files already on the server (faster, no size limits)

### 1. Required Files
- **RED Band (B04)**: Sentinel-2 red band GeoTIFF
- **NIR Band (B08)**: Sentinel-2 near-infrared band GeoTIFF

### 2. Optional SAR Files
- **C-band SAR**: Sentinel-1 or other C-band SAR GeoTIFF
- **L-band SAR**: ALOS PALSAR or other L-band SAR GeoTIFF

### 3. Run Fusion
Click "Run Fusion" to process the data and generate:
- Enhanced Forest Composite PNG image
- Processing summary and metrics
- Downloadable HTML report

## Expected Outputs

- **EFC Image**: RGB composite where R=1-NDVI, G=NDVI, B=normalized SAR dB
- **Summary**: One-line processing summary with NDVI range and SAR band count
- **Metrics Table**: NDVI statistics and SAR band presence
- **Downloads**: EFC PNG and HTML report files

## Input Modes

### Upload Files Mode
- Upload GeoTIFF files directly (up to 2GB each)
- Shows file sizes and memory estimates
- Automatic cleanup after processing
- Warns about large files that may impact performance

### Server Files Mode
For large files or better performance:
1. Select "Use server files" mode
2. Enter file paths relative to the app directory (e.g., `data/S2_B04.tif`)
3. Files must exist on the server filesystem
4. No upload size limits apply
5. Faster processing (no upload time)

## Windowed Processing

For rasters larger than 12k x 12k pixels:
- Automatic windowed processing is enabled
- Creates downscaled quicklook (max 2048x2048) for visualization
- Computes accurate statistics across full resolution
- Reduces memory usage significantly

## Troubleshooting

### File Upload Issues
- Files >1GB: Consider using server files mode
- Upload limit: 2GB per file
- Ensure files are in GeoTIFF format (.tif or .tiff)
- Files should be georeferenced with valid CRS
- Check that RED and NIR bands are from the same scene/area

### Server Files Issues
- Verify file paths are correct and files exist
- Use forward slashes in paths (e.g., `data/file.tif`)
- Check file permissions are readable

### Processing Errors
- Verify GeoTIFF files are not corrupted
- Ensure sufficient RAM for large files
- Check that files contain valid raster data

### Performance Tips
- Use server files mode for files >1GB
- Server files avoid upload time and memory usage
- Large rasters automatically use windowed processing
- Memory estimate shown for server files
- Temporary uploads are auto-cleaned after processing
- Very large files (>8GB estimated RAM) trigger windowed mode