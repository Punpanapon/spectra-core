"""Earth Engine exporter for SPECTRA training inputs.

This script fetches Sentinel-2 (B4, B8), Sentinel-1 (VV/VH dB),
optional Dynamic World, and optional Hansen loss labels, aligning
everything to a single grid and writing GeoTIFFs ready for chipping.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ee
import requests
from tqdm import tqdm

CLOUD_PROB_THRESHOLD = 60
AUTO_PIXEL_THRESHOLD = 8000 * 8000  # switch to Drive above this
DW_PROB_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]
MAX_PIXELS = int(1e13)
DOWNLOAD_CHUNK = 1024 * 1024


@dataclass
class ExportResult:
    name: str
    path: str
    shape: Tuple[int, int]
    dtype: str
    crs: str
    mode: str
    task: Optional[ee.batch.Task] = None


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def parse_bbox(bbox_str: str) -> List[float]:
    parts = [float(p) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "bbox must be four comma-separated numbers: minLon,minLat,maxLon,maxLat"
        )
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise argparse.ArgumentTypeError("bbox coordinates are invalid (min >= max).")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aligned SPECTRA training inputs from Google Earth Engine."
    )
    parser.add_argument(
        "--bbox",
        required=True,
        help='Bounding box "minLon,minLat,maxLon,maxLat" in EPSG:4326',
        type=parse_bbox,
    )
    parser.add_argument("--t0", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--t1", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--gdrive_folder",
        default="SPECTRA_GEE",
        help="Google Drive folder for Drive exports",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Output pixel size in meters (default: match S2 projection or 10 m).",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="Target CRS (e.g., EPSG:32647). Defaults to S2 native projection.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "download", "drive"],
        default="auto",
        help="Export mode: auto=download if small else Drive.",
    )
    parser.add_argument(
        "--include_dynamicworld",
        type=parse_bool,
        default=False,
        help="Include Dynamic World mode/probability exports.",
    )
    parser.add_argument(
        "--loss_years",
        default=None,
        help='Comma-separated loss years (e.g., "2024,2025") to export Hansen label.',
    )
    parser.add_argument("--prefix", default="aoi", help="Filename prefix.")
    parser.add_argument(
        "--project",
        default=None,
        help="GEE Cloud Project ID to bill against. Falls back to EE_PROJECT or GOOGLE_CLOUD_PROJECT env vars.",
    )
    return parser.parse_args()


def resolve_project(cli_project: Optional[str]) -> Optional[str]:
    return cli_project or os.environ.get("EE_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )


def init_earth_engine(project: Optional[str]) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001
        hint = (
            "Failed to initialize Earth Engine. "
            "Ensure `earthengine authenticate` succeeded and set a billing project "
            "via --project or env EE_PROJECT/GOOGLE_CLOUD_PROJECT."
        )
        raise SystemExit(f"{hint}\nDetails: {exc}") from exc


def geometry_from_bbox(bbox: Sequence[float]) -> ee.Geometry:
    # Use Rectangle for broader compatibility across ee API versions.
    return ee.Geometry.Rectangle(bbox, geodesic=False)


def select_one(img: ee.Image, candidates: Sequence[str]) -> Optional[ee.Image]:
    band_names = ee.Image(img).bandNames().getInfo()
    for name in candidates:
        if name in band_names:
            return ee.Image(img).select(name)
    return None


def require_band(img: ee.Image, candidates: Sequence[str], label: str) -> ee.Image:
    selected = select_one(img, candidates)
    if selected is None:
        available = ee.Image(img).bandNames().getInfo()
        raise RuntimeError(
            f"Band search for '{label}' failed. Tried {candidates}, "
            f"available bands: {available}"
        )
    return selected


def mask_s2_clouds(
    s2: ee.ImageCollection, cloud_prob: ee.ImageCollection
) -> ee.ImageCollection:
    joined = ee.Join.saveFirst("cloud_prob").apply(
        s2,
        cloud_prob,
        ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def _mask(img: ee.Image) -> ee.Image:
        cp = ee.Image(img.get("cloud_prob"))
        masked = ee.Algorithms.If(
            cp,
            img.updateMask(cp.lt(CLOUD_PROB_THRESHOLD)),
            img,
        )
        masked_img = ee.Image(masked)
        return masked_img.copyProperties(img, img.propertyNames())

    return ee.ImageCollection(joined).map(_mask)


def build_s2_composite(
    geom: ee.Geometry, t0: str, t1: str
) -> Tuple[ee.Image, str, int, int]:
    cloud_prob = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(geom)
        .filterDate(t0, t1)
    )
    for coll_id in ["COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/S2_HARMONIZED"]:
        s2 = (
            ee.ImageCollection(coll_id)
            .filterBounds(geom)
            .filterDate(t0, t1)
        )
        count = s2.size().getInfo()
        if count == 0:
            continue
        masked = mask_s2_clouds(s2, cloud_prob)
        with_prob = masked.filter(ee.Filter.notNull(["cloud_prob"]))
        cp_count = with_prob.size().getInfo()
        print(
            f"[S2] Collection {coll_id}: {count} images "
            f"({cp_count} with cloud-probability); cloud threshold={CLOUD_PROB_THRESHOLD}"
        )
        sorted_coll = masked.sort("CLOUDY_PIXEL_PERCENTAGE")
        composite = sorted_coll.limit(min(count, 40)).median()
        return composite, coll_id, count, cp_count
    raise SystemExit("No Sentinel-2 imagery found for the requested bbox/dates.")


def build_s1_composites(
    geom: ee.Geometry, t0: str, t1: str
) -> Tuple[ee.Image, ee.Image, int]:
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(t0, t1)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    count = s1.size().getInfo()
    if count == 0:
        raise SystemExit("No Sentinel-1 IW dual-pol scenes found for the bbox/dates.")
    vv = s1.select("VV").median()
    vh = s1.select("VH").median()
    vv_db = vv.log10().multiply(10).rename("VV_dB")
    vh_db = vh.log10().multiply(10).rename("VH_dB")
    print(f"[S1] COPERNICUS/S1_GRD: {count} images used for VV/VH median composites.")
    return vv_db, vh_db, count


def build_dynamic_world(
    geom: ee.Geometry, t0: str, t1: str
) -> Tuple[Optional[ee.Image], Optional[ee.Image], int]:
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(geom)
        .filterDate(t0, t1)
    )
    count = dw.size().getInfo()
    if count == 0:
        print("[DW] No Dynamic World scenes; skipping.")
        return None, None, 0
    mode = dw.select("label").mode().rename("DW_mode")
    prob = dw.select(DW_PROB_BANDS).mean().rename(DW_PROB_BANDS)
    print(f"[DW] GOOGLE/DYNAMICWORLD/V1: {count} images used for mode/mean probs.")
    return mode, prob, count


def build_hansen_loss(
    geom: ee.Geometry, loss_years: Sequence[int], s2_proj: ee.Projection, scale: float
) -> Optional[ee.Image]:
    if not loss_years:
        return None
    gfc = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    lossyear = gfc.select("lossyear")
    years = [int(y) for y in loss_years]
    mask = None
    for year in years:
        offset = year - 2000 if year > 300 else year
        ymask = lossyear.eq(offset)
        mask = ymask if mask is None else mask.Or(ymask)
    if mask is None:
        return None
    label_img = mask.rename("loss").toUint8()
    label_aligned = label_img.reproject(s2_proj.atScale(scale)).clip(geom).unmask(0)
    print(f"[Label] Hansen lossyear for years: {', '.join(map(str, loss_years))}")
    print("[Label] Projection:", label_aligned.projection().getInfo())
    return label_aligned


def resolve_projection(
    image: ee.Image, crs: Optional[str], scale: Optional[float]
) -> ee.Projection:
    base = image.select(0).projection()
    if crs:
        target = ee.Projection(crs)
    else:
        target = ee.Projection(base.crs())
    if scale:
        target = target.atScale(scale)
    else:
        target = target.atScale(base.nominalScale())
    return target


def compute_dimensions(
    region: ee.Geometry, projection: ee.Projection
) -> Tuple[int, int]:
    coords = ee.Image.pixelCoordinates(projection)
    stats = coords.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=region,
        scale=projection.nominalScale(),
        maxPixels=MAX_PIXELS,
        tileScale=4,
    )
    info = stats.getInfo()
    if not info:
        raise SystemExit("Failed to compute output dimensions.")
    width = int(info["x_max"] - info["x_min"] + 1)
    height = int(info["y_max"] - info["y_min"] + 1)
    return height, width


def prepare_image(
    image: ee.Image,
    region: ee.Geometry,
    projection: ee.Projection,
    nodata: float,
    resampling: str,
    to_type: str,
) -> ee.Image:
    casted = (
        image.toFloat()
        if to_type == "float"
        else image.toUint8()
        if to_type == "byte"
        else image
    )
    prepared = (
        casted.resample(resampling)
        .reproject(projection)
        .clip(region)
        .unmask(nodata)
    )
    return prepared


def download_image(
    image: ee.Image,
    out_path: Path,
    region: ee.Geometry,
    crs: str,
    scale: float,
) -> None:
    params: Dict[str, object] = {
        "scale": scale,
        "crs": crs,
        "region": region,
        "fileFormat": "GeoTIFF",
        "filePerBand": False,
        "name": out_path.stem,
    }
    url = image.getDownloadURL(params)
    print(f"[Download] Fetching {out_path.name} ...")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fp, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=out_path.name,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if chunk:
                    fp.write(chunk)
                    pbar.update(len(chunk))


def export_to_drive(
    image: ee.Image,
    description: str,
    drive_folder: str,
    crs: str,
    scale: float,
    region: ee.Geometry,
) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=drive_folder,
        fileNamePrefix=description,
        region=region,
        crs=crs,
        scale=scale,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
        maxPixels=MAX_PIXELS,
    )
    task.start()
    return task


def monitor_drive_tasks(tasks: List[ExportResult]) -> None:
    active = [task for task in tasks if task.task is not None]
    if not active:
        return
    print("\n[Drive] Monitoring export tasks ...")
    pending = active.copy()
    while pending:
        time.sleep(15)
        still_running = []
        for item in pending:
            status = item.task.status()
            state = status.get("state", "UNKNOWN")
            if state in {"COMPLETED", "FAILED", "CANCELLED"}:
                msg = status.get("error_message", "")
                suffix = f" ({msg})" if msg else ""
                print(f" - {item.name}: {state}{suffix}")
            else:
                still_running.append(item)
        pending = still_running


def summarize_outputs(outputs: List[ExportResult]) -> None:
    if not outputs:
        return
    print("\nOutput summary:")
    header = f"{'Layer':18} {'Path':40} {'Shape':18} {'Dtype':8} {'CRS'}"
    print(header)
    print("-" * len(header))
    for item in outputs:
        shape_str = f"{item.shape[0]}x{item.shape[1]}"
        path_str = item.path
        print(
            f"{item.name:18} {path_str:40} {shape_str:18} {item.dtype:8} {item.crs}"
        )


def main() -> None:
    args = parse_args()
    project = resolve_project(args.project)
    init_earth_engine(project)

    geom = geometry_from_bbox(args.bbox)
    s2_composite, s2_source, _, cp_count = build_s2_composite(geom, args.t0, args.t1)
    print(f"[S2] Using composite from {s2_source}.")
    print("S2 bands:", ee.Image(s2_composite).bandNames().getInfo())
    target_proj = resolve_projection(s2_composite, args.crs, args.scale or 10)
    crs = target_proj.crs().getInfo()
    scale = float(target_proj.nominalScale().getInfo())
    s2_proj = target_proj
    height, width = compute_dimensions(geom, target_proj)
    pixel_est = height * width
    print(
        f"[Grid] CRS={crs}, scale={scale}m, shape={height}x{width}, "
        f"cloud-prob matches={cp_count}"
    )

    export_mode = args.mode
    if export_mode == "auto":
        if pixel_est > AUTO_PIXEL_THRESHOLD:
            export_mode = "drive"
            print(
                f"[Mode] Auto-switching to Drive (pixels ~{pixel_est:,} "
                f"> {AUTO_PIXEL_THRESHOLD:,})."
            )
        else:
            export_mode = "download"
            print(
                f"[Mode] Auto using direct download (pixels ~{pixel_est:,} "
                f"<= {AUTO_PIXEL_THRESHOLD:,})."
            )
    else:
        print(f"[Mode] Using explicit export mode: {export_mode}")

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    vv_db, vh_db, _ = build_s1_composites(geom, args.t0, args.t1)
    dw_mode, dw_prob, _ = build_dynamic_world(geom, args.t0, args.t1)
    loss_years = (
        [int(y.strip()) for y in args.loss_years.split(",")] if args.loss_years else []
    )
    loss_label = (
        build_hansen_loss(geom, loss_years, s2_proj, scale) if loss_years else None
    )
    red = require_band(s2_composite, ["B4", "B04"], "red").rename("B4")
    nir = require_band(s2_composite, ["B8", "B08"], "nir").rename("B8")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    outputs: List[ExportResult] = []
    drive_tasks: List[ExportResult] = []

    export_targets = [
        (
            "S2_B04",
            prepare_image(
                red,
                geom,
                target_proj,
                nodata=-9999,
                resampling="bilinear",
                to_type="float",
            ),
            "float32",
        ),
        (
            "S2_B08",
            prepare_image(
                nir,
                geom,
                target_proj,
                nodata=-9999,
                resampling="bilinear",
                to_type="float",
            ),
            "float32",
        ),
        (
            "S1_VV_dB",
            prepare_image(
                vv_db,
                geom,
                target_proj,
                nodata=-9999,
                resampling="bilinear",
                to_type="float",
            ),
            "float32",
        ),
        (
            "S1_VH_dB",
            prepare_image(
                vh_db,
                geom,
                target_proj,
                nodata=-9999,
                resampling="bilinear",
                to_type="float",
            ),
            "float32",
        ),
    ]

    if dw_mode is not None and args.include_dynamicworld:
        export_targets.append(
            (
                "DW_mode",
                prepare_image(
                    dw_mode,
                    geom,
                    target_proj,
                    nodata=0,
                    resampling="nearest",
                    to_type="byte",
                ),
                "uint8",
            )
        )
        if pixel_est <= 16_000_000 and dw_prob is not None:
            export_targets.append(
                (
                    "DW_prob",
                    prepare_image(
                        dw_prob,
                        geom,
                        target_proj,
                        nodata=-9999,
                        resampling="bilinear",
                        to_type="float",
                    ),
                    "float32",
                )
            )
        elif dw_prob is not None:
            print(
                "[DW] Probability stack is large; skipping per-class probabilities."
            )

    if loss_label is not None:
        export_targets.append(
            (
                "label_loss",
                loss_label,
                "uint8",
            )
        )

    for name, img, dtype in export_targets:
        filename = f"{args.prefix}_{name}.tif"
        out_path = out_dir / filename
        description = f"{args.prefix}_{name}"
        if export_mode == "download":
            try:
                download_image(img, out_path, geom, crs, scale)
                outputs.append(
                    ExportResult(
                        name=name,
                        path=str(out_path),
                        shape=(height, width),
                        dtype=dtype,
                        crs=crs,
                        mode="download",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[Warning] Download failed for {name}: {exc}. "
                    "Switching to Drive export."
                )
                task = export_to_drive(
                    img,
                    description=description,
                    drive_folder=args.gdrive_folder,
                    crs=crs,
                    scale=scale,
                    region=geom,
                )
                result = ExportResult(
                    name=name,
                    path=f"Drive:{args.gdrive_folder}/{description}",
                    shape=(height, width),
                    dtype=dtype,
                    crs=crs,
                    mode="drive",
                    task=task,
                )
                outputs.append(result)
                drive_tasks.append(result)
        else:
            task = export_to_drive(
                img,
                description=description,
                drive_folder=args.gdrive_folder,
                crs=crs,
                scale=scale,
                region=geom,
            )
            result = ExportResult(
                name=name,
                path=f"Drive:{args.gdrive_folder}/{description}",
                shape=(height, width),
                dtype=dtype,
                crs=crs,
                mode="drive",
                task=task,
            )
            outputs.append(result)
            drive_tasks.append(result)

    if drive_tasks and export_mode == "drive":
        print(
            f"[Drive] Started {len(drive_tasks)} exports to folder "
            f"{args.gdrive_folder}. Task IDs:"
        )
        for item in drive_tasks:
            assert item.task is not None
            print(f" - {item.name}: {item.task.id}")

    monitor_drive_tasks(drive_tasks)
    summarize_outputs(outputs)
    print(f"\nDone. Data directory: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except ee.EEException as exc:
        sys.exit(f"Earth Engine error: {exc}")
