import json
import os
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from PIL import Image
from .geo import load_tif_as_array, resample_to_match
from .ndvi import compute_ndvi, normalize01
from .sar import to_db

def align_and_stack(before_paths, after_paths):
    """Align and stack before/after imagery."""
    # Use first RED as reference
    with rasterio.open(before_paths['red']) as ref_ds:
        ref_profile = ref_ds.profile
        
    # Load and align all bands
    result = {}
    
    # Before imagery
    result['red1'] = load_tif_as_array(before_paths['red'])
    result['nir1'] = load_tif_as_array(before_paths['nir'])
    with rasterio.open(before_paths['red']) as ref_ds:
        result['nir1'] = resample_to_match(result['nir1'], ref_ds)
    
    if 'sar_c' in before_paths and before_paths['sar_c']:
        sar_c1 = load_tif_as_array(before_paths['sar_c'])
        with rasterio.open(before_paths['red']) as ref_ds:
            result['sar_c1'] = resample_to_match(sar_c1, ref_ds)
    
    if 'sar_l' in before_paths and before_paths['sar_l']:
        sar_l1 = load_tif_as_array(before_paths['sar_l'])
        with rasterio.open(before_paths['red']) as ref_ds:
            result['sar_l1'] = resample_to_match(sar_l1, ref_ds)
    
    # After imagery
    red2 = load_tif_as_array(after_paths['red'])
    nir2 = load_tif_as_array(after_paths['nir'])
    with rasterio.open(before_paths['red']) as ref_ds:
        result['red2'] = resample_to_match(red2, ref_ds)
        result['nir2'] = resample_to_match(nir2, ref_ds)
    
    if 'sar_c' in after_paths and after_paths['sar_c']:
        sar_c2 = load_tif_as_array(after_paths['sar_c'])
        with rasterio.open(before_paths['red']) as ref_ds:
            result['sar_c2'] = resample_to_match(sar_c2, ref_ds)
    
    if 'sar_l' in after_paths and after_paths['sar_l']:
        sar_l2 = load_tif_as_array(after_paths['sar_l'])
        with rasterio.open(before_paths['red']) as ref_ds:
            result['sar_l2'] = resample_to_match(sar_l2, ref_ds)
    
    result['transform'] = ref_profile['transform']
    result['crs'] = ref_profile['crs']
    
    return result

def compute_change(stacked, ndvi_min=0.4, ndvi_drop=-0.15, min_patch=100, use_sar=True):
    """Compute change detection metrics and polygons."""
    # Compute NDVI
    ndvi1 = compute_ndvi(stacked['nir1'], stacked['red1'])
    ndvi2 = compute_ndvi(stacked['nir2'], stacked['red2'])
    dndvi = ndvi2 - ndvi1
    
    # Compute SAR changes if available
    dsar = None
    if use_sar and 'sar_c1' in stacked and 'sar_c2' in stacked:
        sar1_db = to_db(stacked['sar_c1'])
        sar2_db = to_db(stacked['sar_c2'])
        dsar = sar2_db - sar1_db
        
        # If L-band available, average with C-band
        if 'sar_l1' in stacked and 'sar_l2' in stacked:
            sar_l1_db = to_db(stacked['sar_l1'])
            sar_l2_db = to_db(stacked['sar_l2'])
            dsar_l = sar_l2_db - sar_l1_db
            dsar = (dsar + dsar_l) / 2
    
    # Create change mask
    mask = (ndvi1 >= ndvi_min) & (dndvi <= ndvi_drop)
    
    # Remove small patches
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) < min_patch:
            mask[labeled == i] = False
    
    # Compute metrics
    changed_pixels = np.sum(mask)
    total_pixels = mask.size
    pct_area = (changed_pixels / total_pixels) * 100
    mean_dndvi = np.mean(dndvi[mask]) if changed_pixels > 0 else 0
    mean_dsar = np.mean(dsar[mask]) if dsar is not None and changed_pixels > 0 else None
    
    metrics = {
        'changed_pixels': int(changed_pixels),
        'pct_area': float(pct_area),
        'mean_dndvi': float(mean_dndvi),
        'mean_dsar': float(mean_dsar) if mean_dsar is not None else None
    }
    
    # Compute confidence
    a, b, c = 6, 3, 0.5
    conf_score = a * abs(mean_dndvi)
    if mean_dsar is not None:
        conf_score += b * abs(mean_dsar)
    conf_score += c * np.log1p(changed_pixels)
    confidence = 1 / (1 + np.exp(-conf_score))  # sigmoid
    
    # Vectorize to polygons
    polygons_gj = {'type': 'FeatureCollection', 'features': []}
    if changed_pixels > 0:
        # Convert mask to polygons
        mask_uint8 = mask.astype(np.uint8)
        for geom, value in shapes(mask_uint8, transform=stacked['transform']):
            if value == 1:  # Changed areas
                poly = shape(geom)
                area_ha = poly.area / 10000  # m² to ha
                feature = {
                    'type': 'Feature',
                    'geometry': geom,
                    'properties': {'area_ha': round(area_ha, 2)}
                }
                polygons_gj['features'].append(feature)
    
    return {
        'ndvi1': ndvi1,
        'ndvi2': ndvi2,
        'dndvi': dndvi,
        'dsar': dsar,
        'mask': mask,
        'metrics': metrics,
        'polygons_gj': polygons_gj,
        'confidence': float(confidence)
    }

def write_artifacts(result, stacked, out_dir, quicklooks=True):
    """Write change detection artifacts."""
    os.makedirs(out_dir, exist_ok=True)
    
    if quicklooks:
        # Before quicklook (false color: NIR-R-G)
        before_rgb = np.stack([
            normalize01(stacked['nir1']),
            normalize01(stacked['red1']),
            normalize01(stacked['red1'] * 0.5)  # Fake green
        ], axis=-1)
        before_img = (before_rgb * 255).astype(np.uint8)
        Image.fromarray(before_img).save(os.path.join(out_dir, 'before.png'))
        
        # After quicklook
        after_rgb = np.stack([
            normalize01(stacked['nir2']),
            normalize01(stacked['red2']),
            normalize01(stacked['red2'] * 0.5)
        ], axis=-1)
        after_img = (after_rgb * 255).astype(np.uint8)
        Image.fromarray(after_img).save(os.path.join(out_dir, 'after.png'))
        
        # Delta NDVI
        dndvi_norm = normalize01(result['dndvi'])
        dndvi_img = (dndvi_norm * 255).astype(np.uint8)
        Image.fromarray(dndvi_img, mode='L').save(os.path.join(out_dir, 'delta.png'))
        
        # Change mask
        mask_img = (result['mask'] * 255).astype(np.uint8)
        Image.fromarray(mask_img, mode='L').save(os.path.join(out_dir, 'mask.png'))
    
    # Save polygons
    with open(os.path.join(out_dir, 'polygons.geojson'), 'w') as f:
        json.dump(result['polygons_gj'], f, indent=2)
    
    # Save metrics
    metrics_full = result['metrics'].copy()
    metrics_full['confidence'] = result['confidence']
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_full, f, indent=2)
    
    # Save summary
    n_polygons = len(result['polygons_gj']['features'])
    total_area = sum(f['properties']['area_ha'] for f in result['polygons_gj']['features'])
    conf_band = 'High' if result['confidence'] > 0.7 else 'Medium' if result['confidence'] > 0.4 else 'Low'
    
    summary = f"Change detected: {n_polygons} polygons, {total_area:.1f} ha total, confidence: {conf_band}"
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    
    return {
        'before_png': os.path.join(out_dir, 'before.png'),
        'after_png': os.path.join(out_dir, 'after.png'),
        'delta_png': os.path.join(out_dir, 'delta.png'),
        'mask_png': os.path.join(out_dir, 'mask.png'),
        'polygons_path': os.path.join(out_dir, 'polygons.geojson'),
        'summary': summary
    }