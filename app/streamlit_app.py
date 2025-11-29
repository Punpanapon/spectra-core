# ENTRYPOINT FOR STREAMLIT COMMUNITY CLOUD
# Chosen automatically by Codex based on project structure.

import os
import sys
import shutil
import tempfile
import csv
from datetime import datetime, date
import json
import requests
import numpy as np
import rasterio
from rasterio.transform import from_origin
import streamlit as st
import ee
import random
import torch
from PIL import Image, ImageDraw, ImageFont

# Ensure the project root (spectra-core) is on sys.path so we can import spectra_ai
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../spectra-core/app
PROJECT_ROOT = os.path.dirname(THIS_DIR)                       # .../spectra-core
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DEFAULT_UNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "efc_unet.pt")

# Secrets/environment handling for Streamlit Cloud and local use
secrets_llm = st.secrets.get("llm", {}) if hasattr(st, "secrets") else {}
secrets_news = st.secrets.get("newsdata", {}) if hasattr(st, "secrets") else {}
secrets_ee = st.secrets.get("earthengine", {}) if hasattr(st, "secrets") else {}
secrets_ai = st.secrets.get("ai", {}) if hasattr(st, "secrets") else {}

# Prefer secrets.toml values, fall back to env for local runs
NEWSDATA_API_KEY = secrets_news.get("api_key") or os.getenv("NEWSDATA_API_KEY")
if secrets_llm.get("api_key") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = secrets_llm["api_key"]
if secrets_llm.get("model") and not os.getenv("GEMINI_MODEL"):
    os.environ["GEMINI_MODEL"] = secrets_llm["model"]
if secrets_llm.get("mode") and not os.getenv("LLM_MODE"):
    os.environ["LLM_MODE"] = secrets_llm["mode"]
if "use_llm" in secrets_llm and not os.getenv("SPECTRA_USE_LLM"):
    os.environ["SPECTRA_USE_LLM"] = "1" if secrets_llm.get("use_llm") else "0"

# Optionally seed Earth Engine service account creds for downstream tools
if secrets_ee.get("service_account") and not os.getenv("EE_SERVICE_ACCOUNT"):
    os.environ["EE_SERVICE_ACCOUNT"] = secrets_ee["service_account"]
if secrets_ee.get("private_key") and not os.getenv("EE_PRIVATE_KEY"):
    os.environ["EE_PRIVATE_KEY"] = secrets_ee["private_key"]
def init_earthengine():
    """
    Initialize Earth Engine using credentials from st.secrets['earthengine'].

    Returns
    -------
    (ok: bool, message: str)
    """
    ee_cfg = st.secrets.get("earthengine", {}) if hasattr(st, "secrets") else {}
    service_account = ee_cfg.get("service_account")
    key_json = ee_cfg.get("key_json") or ee_cfg.get("private_key")
    project_id = ee_cfg.get("project_id")

    if not service_account or not key_json:
        return False, "Earth Engine credentials missing in st.secrets['earthengine']"

    try:
        credentials = ee.ServiceAccountCredentials(
            service_account,
            key_data=key_json,
        )

        if project_id:
            ee.Initialize(credentials=credentials, project=project_id)
        else:
            ee.Initialize(credentials=credentials)

        # tiny test computation
        ee.Number(1).add(1).getInfo()
        return True, "Earth Engine initialized"

    except Exception as e:  # noqa: BLE001
        return False, f"Earth Engine error: {e}"


ee_ok, ee_msg = init_earthengine()


def fetch_environment_news(topic: str, max_articles: int = 6):
    """
    Fetch environmental news using NewsData.io's 'latest' endpoint.

    We constrain results by:
    - category=environment
    - a topic-specific keyword in q
    """
    if not NEWSDATA_API_KEY:
        return [
            {
                "title": "Configure NEWSDATA_API_KEY to pull live environmental news",
                "source": "SPECTRA Core",
                "url": "https://newsdata.io/",
                "published_at": None,
                "summary": (
                    "Set the NEWSDATA_API_KEY environment variable to enable live "
                    "environmental, deforestation, and climate news in this tab."
                ),
            }
        ]

    topic_to_query = {
        "All environment": "environment",
        "Deforestation & land use": "deforestation",
        "Weather & climate": "climate change",
    }
    keyword = topic_to_query.get(topic, "environment")

    endpoint = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": keyword,
        "category": "environment",
        "language": "en",
        "size": min(max_articles, 10),
        "prioritydomain": "top",
        "removeduplicate": 1,
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [
            {
                "title": "Unable to fetch live news",
                "source": "SPECTRA Core",
                "url": "https://newsdata.io/",
                "published_at": None,
                "summary": "Check your internet connection or NEWSDATA_API_KEY, then retry.",
            }
        ]

    if data.get("status") != "success":
        return [
            {
                "title": "News API returned an error",
                "source": "SPECTRA Core",
                "url": "https://newsdata.io/",
                "published_at": None,
                "summary": f"Raw status from NewsData.io: {data.get('status')}",
            }
        ]

    articles = []
    for item in data.get("results", []):
        articles.append(
            {
                "title": item.get("title"),
                "source": item.get("source_id") or "Unknown",
                "url": item.get("link"),
                "published_at": item.get("pubDate"),
                "summary": item.get("description"),
            }
        )

    return articles or [
        {
            "title": "No recent articles found for this topic",
            "source": "SPECTRA Core",
            "url": "https://newsdata.io/",
            "published_at": None,
            "summary": f"No results for keyword '{keyword}' in environment category.",
        }
    ]

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra_core.pipeline import run_pipeline
from spectra_core.report import generate_report
from spectra_core.agent.insights import summarize_ndvi, insight_bullets, narrative, qa_answer, load_fusion_summary, llm_caps_from_env
from spectra_core.agent.llm_providers import LocalLlamaCpp, GeminiProvider, HuggingFaceProvider, OpenAICompatible, OllamaProvider
from spectra_core.agent.usage_limits import reset_session, get_usage
from spectra_core.ai.paths import open_da
from spectra_core.util.config import get_env_or_secret, has_env_or_secret
from spectra_loader import show_spectra_loader
try:
    from spectra_ai.infer_unet import run_unet_on_efc_tile
    AI_AVAILABLE = True
    AI_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    AI_AVAILABLE = False
    AI_IMPORT_ERROR = e
def load_gee_s2_patch(lat, lon, start_date, end_date, cloud_max, patch_size_m):
    """Fetch a Sentinel-2 patch (B4, B8) from Earth Engine."""
    if not ee_ok:
        raise RuntimeError("Earth Engine is not initialized.")

    point = ee.Geometry.Point([lon, lat])
    half = patch_size_m / 2.0
    region = point.buffer(half).bounds()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(point)
        .filterDate(start_date.isoformat(), end_date.isoformat())
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        .select(["B4", "B8"])
    ).sort("CLOUDY_PIXEL_PERCENTAGE")

    image = collection.first()
    if image is None:
        raise RuntimeError(
            "No Sentinel-2 images found for this AOI/date and cloud filter."
        )

    rect_dict = image.sampleRectangle(region=region, defaultValue=0).getInfo()
    print("GEE sampleRectangle keys:", rect_dict.keys())

    props = rect_dict.get("properties", rect_dict)
    if "B4" not in props or "B8" not in props:
        raise RuntimeError(
            f"Unexpected sampleRectangle structure; missing B4/B8 keys. Available keys: {list(props.keys())}"
        )

    red_arr = np.array(props["B4"])
    nir_arr = np.array(props["B8"])

    if red_arr.ndim == 3 and red_arr.shape[-1] == 1:
        red_arr = red_arr[..., 0]
    if nir_arr.ndim == 3 and nir_arr.shape[-1] == 1:
        nir_arr = nir_arr[..., 0]

    if red_arr.ndim != 2 or nir_arr.ndim != 2:
        raise RuntimeError(
            f"Fetched Sentinel-2 arrays are not 2-D rasters. Shapes: B4={red_arr.shape}, B8={nir_arr.shape}"
        )

    if red_arr.shape != nir_arr.shape:
        h_min = min(red_arr.shape[0], nir_arr.shape[0])
        w_min = min(red_arr.shape[1], nir_arr.shape[1])
        red_arr = red_arr[:h_min, :w_min]
        nir_arr = nir_arr[:h_min, :w_min]

    return red_arr, nir_arr


def load_gee_s1_patch(lat, lon, start_date, end_date, patch_size_m):
    """Fetch a Sentinel-1 VV patch (in dB) from Earth Engine."""
    if not ee_ok:
        raise RuntimeError("Earth Engine is not initialized.")

    point = ee.Geometry.Point([lon, lat])
    half = patch_size_m / 2.0
    region = point.buffer(half).bounds()

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(point)
        .filterDate(start_date.isoformat(), end_date.isoformat())
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("productType", "GRD"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    ).sort("relativeOrbitNumber")

    image = collection.first()
    if image is None:
        raise RuntimeError("No Sentinel-1 GRD images found for this AOI/date.")

    vv_db = image.select("VV").log10().multiply(10.0)
    rect_dict = vv_db.sampleRectangle(
        region=region,
        defaultValue=-30,
    ).getInfo()
    print("GEE S1 sampleRectangle keys:", rect_dict.keys())
    props = rect_dict.get("properties", rect_dict)

    s1_data = None
    for key in ("VV", "constant", "array"):
        if key in props:
            s1_data = props[key]
            break
    if s1_data is None:
        raise RuntimeError(
            f"Unexpected Sentinel-1 sampleRectangle structure: {list(props.keys())}"
        )

    sar_arr = np.array(s1_data)
    if sar_arr.ndim == 3 and sar_arr.shape[-1] == 1:
        sar_arr = sar_arr[..., 0]
    if sar_arr.ndim != 2:
        raise RuntimeError(f"Sentinel-1 array is not 2-D. Shape: {sar_arr.shape}")

    return sar_arr


def load_gee_l_band_patch(lat, lon, start_date, end_date, patch_size_m):
    """
    Placeholder for future L-band SAR (e.g. ALOS-2) support.

    When a specific EE dataset is chosen, implement it similarly to load_gee_s1_patch
    and return a 2-D NumPy array aligned to S2/S1.
    """
    raise RuntimeError(
        "L-band SAR (e.g., ALOS-2) is not configured yet. Choose an Earth Engine dataset and implement this function."
    )


def gee_input_panel():
    """
    Render UI to configure an Earth Engine Sentinel-2 input patch and fetch it.

    Stores fetched arrays in st.session_state on success.
    """
    if not ee_ok:
        st.error("Earth Engine is not initialized. Check secrets and restart the app.")
        return

    st.subheader("Use GEE (Sentinel-2)")
    if "gee_ready" not in st.session_state:
        st.session_state["gee_ready"] = False
        st.session_state["gee_red"] = None
        st.session_state["gee_nir"] = None
    if "gee_s1_ready" not in st.session_state:
        st.session_state["gee_s1_ready"] = False
        st.session_state["gee_s1"] = None
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=14.3, format="%.6f")
        start_date = st.date_input("Start date", value=date(2024, 1, 1))
        cloud_max = st.slider("Max cloud cover (%)", 0, 100, 30)
    with col2:
        lon = st.number_input("Longitude", value=101.2, format="%.6f")
        end_date = st.date_input("End date", value=date(2024, 1, 31))
        patch_size = st.selectbox(
            "Patch size (meters, square)",
            options=[512, 1024, 2048],
            index=1,
        )
    include_s1 = st.checkbox("Include Sentinel-1 C-band SAR", value=False)

    fetch = st.button("Fetch Sentinel-2 from GEE")
    if fetch:
        with st.spinner("Fetching Sentinel-2 patch from Earth Engine..."):
            try:
                red_arr, nir_arr = load_gee_s2_patch(
                    lat=lat,
                    lon=lon,
                    start_date=start_date,
                    end_date=end_date,
                    cloud_max=cloud_max,
                    patch_size_m=patch_size,
                )
                sar_arr = None
                if include_s1:
                    sar_arr = load_gee_s1_patch(
                        lat=lat,
                        lon=lon,
                        start_date=start_date,
                        end_date=end_date,
                        patch_size_m=patch_size,
                    )
                if sar_arr is not None:
                    h_min = min(red_arr.shape[0], sar_arr.shape[0])
                    w_min = min(red_arr.shape[1], sar_arr.shape[1])
                    red_arr = red_arr[:h_min, :w_min]
                    nir_arr = nir_arr[:h_min, :w_min]
                    sar_arr = sar_arr[:h_min, :w_min]
                st.session_state["gee_red"] = red_arr
                st.session_state["gee_nir"] = nir_arr
                st.session_state["gee_ready"] = True
                st.session_state["gee_params"] = {
                    "lat": lat,
                    "lon": lon,
                    "patch_size_m": patch_size,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "cloud_max": cloud_max,
                }
                if sar_arr is not None:
                    st.session_state["gee_s1"] = sar_arr
                    st.session_state["gee_s1_ready"] = True
                else:
                    st.session_state["gee_s1_ready"] = False
                msg = f"GEE S2 patch loaded. Shape: {red_arr.shape}"
                if sar_arr is not None:
                    msg += " (Sentinel-1 SAR included)"
                st.success(msg)
            except Exception as e:  # noqa: BLE001
                st.session_state["gee_ready"] = False
                st.session_state["gee_s1_ready"] = False
                st.error(f"Error fetching from Earth Engine: {e}")

    if st.session_state.get("gee_ready"):
        red = st.session_state.get("gee_red")
        details = f"Using GEE patch with shape: {getattr(red, 'shape', '?')}"
        if st.session_state.get("gee_s1_ready"):
            details += ". Sentinel-1 SAR loaded."
        st.info(details)
    else:
        st.info("Click 'Fetch Sentinel-2 from GEE' to load imagery, then click 'Run Fusion'.")


def run_fusion_from_gee_arrays(
    red_arr: np.ndarray, nir_arr: np.ndarray, output_dir: str, c_sar_arr: np.ndarray | None = None
):
    """
    Convenience bridge: write GEE arrays to temporary GeoTIFFs and call the existing fusion function.
    """
    if red_arr.shape != nir_arr.shape:
        raise RuntimeError("RED and NIR arrays must have matching shapes.")

    temp_dir = tempfile.mkdtemp(prefix="gee_inputs_")
    red_arr = np.asarray(red_arr, dtype=np.float32)
    nir_arr = np.asarray(nir_arr, dtype=np.float32)
    transform = from_origin(0, 0, 10, 10)
    profile = {
        "driver": "GTiff",
        "height": red_arr.shape[0],
        "width": red_arr.shape[1],
        "count": 1,
        "dtype": red_arr.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
    }

    red_path = os.path.join(temp_dir, "red.tif")
    nir_path = os.path.join(temp_dir, "nir.tif")

    with rasterio.open(red_path, "w", **profile) as dst:
        dst.write(red_arr, 1)
    with rasterio.open(nir_path, "w", **profile) as dst:
        dst.write(nir_arr, 1)

    sar_c_path = None
    if c_sar_arr is not None:
        sar_c_arr = np.asarray(c_sar_arr, dtype=np.float32)
        with rasterio.open(sar_c_path, "w", **profile) as dst:
            dst.write(sar_c_arr, 1)

    efc_path, metrics, summary = run_pipeline(
        red_path, nir_path, sar_c_path=sar_c_path, sar_l_path=None, output_dir=output_dir
    )
    return efc_path, metrics, summary, red_path, nir_path, temp_dir


def get_project_root() -> str:
    # Reuse the existing PROJECT_ROOT if defined, otherwise derive from this file.
    try:
        return PROJECT_ROOT
    except NameError:
        return os.path.dirname(os.path.abspath(__file__))


METADATA_HEADERS = ["tile_id", "split", "lat", "lon", "patch_size_m", "start_date", "end_date"]


def _tile_metadata_csv_path() -> str:
    project_root = get_project_root()
    return os.path.join(project_root, "data", "efc_tiles", "tile_metadata.csv")


def _ensure_metadata_csv():
    """
    Create the metadata CSV with header if it does not exist yet.
    """
    csv_path = _tile_metadata_csv_path()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(METADATA_HEADERS)


def _format_meta_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return "" if value is None else value


def append_tile_metadata(tile_path: str, split: str, metadata: dict | None):
    """
    Append a single row to data/efc_tiles/tile_metadata.csv for the saved tile.
    """
    _ensure_metadata_csv()
    csv_path = _tile_metadata_csv_path()
    tile_id = os.path.splitext(os.path.basename(tile_path))[0]
    meta = metadata or {}
    row = [
        tile_id,
        split,
        _format_meta_value(meta.get("lat")),
        _format_meta_value(meta.get("lon")),
        _format_meta_value(meta.get("patch_size_m")),
        _format_meta_value(meta.get("start_date")),
        _format_meta_value(meta.get("end_date")),
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"[tile_metadata] Appended metadata for {tile_id} -> {csv_path}")


def save_single_efc_tile(efc_rgb: np.ndarray, split: str, metadata: dict | None = None) -> str:
    """
    Save the full EFC tile to:
      <PROJECT_ROOT>/data/efc_tiles/<split>/images/tile_<timestamp>.png
    Returns the absolute path.
    """
    assert split in ("train", "val")
    project_root = get_project_root()
    images_dir = os.path.join(project_root, "data", "efc_tiles", split, "images")
    os.makedirs(images_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"tile_{ts}.png"
    out_path = os.path.join(images_dir, filename)

    Image.fromarray(efc_rgb).save(out_path)
    append_tile_metadata(out_path, split, metadata)
    print("[save_single_efc_tile] Saved:", out_path)
    return out_path


def auto_save_cropped_tiles(
    efc_rgb: np.ndarray,
    split: str,
    num_tiles: int,
    tile_size: int,
    metadata: dict | None = None,
) -> list[str]:
    """
    Randomly crop `num_tiles` tiles of size tile_size x tile_size from efc_rgb
    and save them under:
      <PROJECT_ROOT>/data/efc_tiles/<split>/images/
    Returns list of saved paths.
    """
    assert split in ("train", "val")
    if num_tiles <= 0:
        return []

    project_root = get_project_root()
    images_dir = os.path.join(project_root, "data", "efc_tiles", split, "images")
    os.makedirs(images_dir, exist_ok=True)

    h, w, _ = efc_rgb.shape
    tile_size = int(tile_size)
    tile_size = max(1, min(tile_size, h, w))

    saved = []
    for i in range(int(num_tiles)):
        if h == tile_size and w == tile_size:
            y0, x0 = 0, 0
        else:
            y0 = random.randint(0, h - tile_size)
            x0 = random.randint(0, w - tile_size)
        crop = efc_rgb[y0 : y0 + tile_size, x0 : x0 + tile_size, :]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"tile_{split}_{tile_size}px_{ts}_{i:03d}.png"
        out_path = os.path.join(images_dir, filename)
        Image.fromarray(crop).save(out_path)
        saved.append(out_path)
        append_tile_metadata(out_path, split, metadata)

    print(f"[auto_save_cropped_tiles] Saved {len(saved)} tiles to", images_dir)
    return saved


def render_dataset_tools():
    """
    Render dataset tools UI for saving EFC tiles.

    This function MUST NOT be inside any button `if`-block.
    It should be called unconditionally from the EFC Fusion tab.
    """
    efc_rgb = st.session_state.get("efc_rgb", None)
    with st.expander("Dataset tools (UNet training)", expanded=False):
        if efc_rgb is None:
            st.warning("No EFC tile in memory yet. Run fusion first.")
            return

        st.info("Using the latest Enhanced Forest Composite stored in session_state.")
        current_metadata = st.session_state.get("efc_tile_metadata")

        # Manual save of the full EFC tile
        split_manual = st.selectbox(
            "Split for single save",
            options=["train", "val"],
            index=0,
            key="ds_split_manual",
        )
        if st.button("Save THIS full EFC tile", key="ds_save_full"):
            try:
                path = save_single_efc_tile(efc_rgb, split_manual, metadata=current_metadata)
                st.success(f"Saved 1 tile to: {path}")
            except Exception as e:
                st.error(f"Error saving full tile: {e}")
                st.exception(e)

        st.markdown("---")
        st.subheader("Auto-generate cropped tiles")

        tile_size = st.number_input(
            "Tile size (pixels)",
            min_value=64,
            max_value=4096,
            value=512,
            step=64,
            key="ds_tile_size",
        )
        num_train = st.number_input(
            "Number of TRAIN tiles",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            key="ds_num_train",
        )
        num_val = st.number_input(
            "Number of VAL tiles",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            key="ds_num_val",
        )

        if st.button("Run auto-save", key="ds_auto_save"):
            try:
                saved_train = auto_save_cropped_tiles(efc_rgb, "train", num_train, tile_size, metadata=current_metadata)
                saved_val = auto_save_cropped_tiles(efc_rgb, "val", num_val, tile_size, metadata=current_metadata)
                total = len(saved_train) + len(saved_val)
                if total == 0:
                    st.info("No tiles requested (both numbers are 0).")
                else:
                    st.success(
                        f"Saved {total} tiles "
                        f"({len(saved_train)} train, {len(saved_val)} val). "
                        "Check data/efc_tiles/train/images and data/efc_tiles/val/images."
                    )
                    if saved_train:
                        st.write("Example TRAIN tile:", saved_train[0])
                    elif saved_val:
                        st.write("Example VAL tile:", saved_val[0])
            except Exception as e:
                st.error(f"Error during auto-save: {e}")
                st.exception(e)

st.set_page_config(page_title="SPECTRA Fusion", page_icon="🛰️", layout="wide")

st.markdown(
    """
    <div style="
        padding: 0.75rem 1.0rem 0.5rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #020617 0%, #0F172A 60%, #022C22 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.0rem;
    ">
      <div>
        <div style="font-size: 1.1rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #A5F3FC;">
          SPECTRA Core
        </div>
        <div style="font-size: 0.9rem; color: #CBD5F5;">
          South-East Asia Platform for Environmental Change Tracking & Remote Analysis
        </div>
      </div>
      <div style="font-size: 1.5rem;">
        🌍🛰️🌲
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🛰️ SPECTRA Enhanced Forest Composite")
st.markdown("Upload Sentinel-2 optical bands and optional SAR data to generate Enhanced Forest Composite visualizations.")
ai_mode = st.sidebar.selectbox(
    "AI overlay",
    ["EFC only", "EFC + AI detections"],
    index=0,
)
show_raw_ai_mask = st.sidebar.checkbox("Show raw AI mask (0/1/2 classes)", value=False)

# Create tabs
tabs = st.tabs(["EFC Fusion", "Change Detection", "Agentic Insights", "News"])
efc_tab, change_tab, agentic_tab, news_tab = tabs

# Sidebar for data input
st.sidebar.header("Data Input")
with st.sidebar:
    st.info(
        "Theme tip: use the Streamlit menu in the top-right "
        "(three dots → Settings → Theme) to switch Light/Dark mode."
    )
    st.markdown("### System status")
    status_icon = "✅" if ee_ok else "⚠️"
    st.write(f"{status_icon} Earth Engine: {ee_msg}")

# Mode selector
input_mode = st.sidebar.radio(
    "Input Mode", ["Upload files", "Use server files", "Use GEE (Sentinel-2)"], index=1
)

with efc_tab:
    red_file = None
    nir_file = None
    sar_c_file = None
    sar_l_file = None
    red_path = None
    nir_path = None
    sar_c_path = None
    sar_l_path = None

    if input_mode == "Upload files":
        red_file = st.sidebar.file_uploader("RED Band (B04) - Required", type=['tif', 'tiff'], key="red")
        nir_file = st.sidebar.file_uploader("NIR Band (B08) - Required", type=['tif', 'tiff'], key="nir")
        sar_c_file = st.sidebar.file_uploader("C-band SAR - Optional", type=['tif', 'tiff'], key="sar_c")
        sar_l_file = st.sidebar.file_uploader("L-band SAR - Optional", type=['tif', 'tiff'], key="sar_l")
        
        # Show file size warnings
        if red_file and red_file.size > 1e9:
            st.sidebar.warning("⚠️ Large file detected. Consider using server files for better performance.")
        if nir_file and nir_file.size > 1e9:
            st.sidebar.warning("⚠️ Large file detected. Consider using server files for better performance.")
    elif input_mode == "Use server files":
        red_path = st.sidebar.text_input("RED Band Path", value="data/S2_B04.tif")
        nir_path = st.sidebar.text_input("NIR Band Path", value="data/S2_B08.tif")
        sar_c_path = st.sidebar.text_input("C-band SAR Path (optional)", value="")
        sar_l_path = st.sidebar.text_input("L-band SAR Path (optional)", value="")
        
        # Validate file existence
        if red_path and not os.path.exists(red_path):
            st.sidebar.error(f"❌ RED file not found: {red_path}")
        if nir_path and not os.path.exists(nir_path):
            st.sidebar.error(f"❌ NIR file not found: {nir_path}")
        if sar_c_path and not os.path.exists(sar_c_path):
            st.sidebar.error(f"❌ C-band SAR file not found: {sar_c_path}")
        if sar_l_path and not os.path.exists(sar_l_path):
            st.sidebar.error(f"❌ L-band SAR file not found: {sar_l_path}")
        
        # Show file sizes and memory estimates
        total_size_mb = 0
        if red_path and os.path.exists(red_path):
            size_mb = os.path.getsize(red_path) / 1e6
            total_size_mb += size_mb
            st.sidebar.info(f"RED: {size_mb:.1f} MB")
        if nir_path and os.path.exists(nir_path):
            size_mb = os.path.getsize(nir_path) / 1e6
            total_size_mb += size_mb
            st.sidebar.info(f"NIR: {size_mb:.1f} MB")
        if sar_c_path and os.path.exists(sar_c_path):
            size_mb = os.path.getsize(sar_c_path) / 1e6
            total_size_mb += size_mb
            st.sidebar.info(f"C-band: {size_mb:.1f} MB")
        if sar_l_path and os.path.exists(sar_l_path):
            size_mb = os.path.getsize(sar_l_path) / 1e6
            total_size_mb += size_mb
            st.sidebar.info(f"L-band: {size_mb:.1f} MB")
        
        if total_size_mb > 0:
            mem_est_mb = total_size_mb * 4  # float32 expansion factor
            st.sidebar.info(f"Est. RAM: ~{mem_est_mb:.0f} MB")
            if mem_est_mb > 8000:
                st.sidebar.warning("⚠️ Large memory usage expected. Windowed processing will be used.")
    elif input_mode == "Use GEE (Sentinel-2)":
        gee_input_panel()

    loader_placeholder = st.empty()

    if st.sidebar.button("🚀 Run Fusion", type="primary"):
        # Validate inputs based on mode
        valid_inputs = False
        temp_inputs_dir = None
        if input_mode == "Upload files":
            if not red_file or not nir_file:
                st.error("❌ Please upload both RED and NIR bands to proceed.")
            else:
                valid_inputs = True
        elif input_mode == "Use server files":
            if not red_path or not nir_path or not os.path.exists(red_path) or not os.path.exists(nir_path):
                st.error("❌ Please provide valid paths for both RED and NIR bands.")
                valid_inputs = False
            else:
                valid_inputs = True
        elif input_mode == "Use GEE (Sentinel-2)":
            if not st.session_state.get("gee_ready"):
                st.error("No GEE data loaded yet. Click 'Fetch Sentinel-2 from GEE' first.")
                valid_inputs = False
            else:
                valid_inputs = True
        
        if valid_inputs:
            with loader_placeholder:
                show_spectra_loader("Running SPECTRA fusion on satellite data…")
            with st.status("Processing fusion pipeline...", expanded=True) as status:
                # Create session directories
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                upload_dir = f"uploads/{timestamp}"
                output_dir = f"outputs/{timestamp}"
                os.makedirs(upload_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                tile_metadata = None
                
                try:
                    if input_mode == "Upload files":
                        status.write("Saving uploaded files...")
                        # Save uploaded files
                        red_path = os.path.join(upload_dir, "red.tif")
                        nir_path = os.path.join(upload_dir, "nir.tif")
                        
                        with open(red_path, "wb") as f:
                            f.write(red_file.read())
                        with open(nir_path, "wb") as f:
                            f.write(nir_file.read())
                        
                        if sar_c_file:
                            sar_c_path = os.path.join(upload_dir, "sar_c.tif")
                            with open(sar_c_path, "wb") as f:
                                f.write(sar_c_file.read())
                        else:
                            sar_c_path = None
                        
                        if sar_l_file:
                            sar_l_path = os.path.join(upload_dir, "sar_l.tif")
                            with open(sar_l_path, "wb") as f:
                                f.write(sar_l_file.read())
                        else:
                            sar_l_path = None
                    elif input_mode == "Use server files":
                        status.write("Using server files...")
                        # Use server paths, clean empty strings
                        sar_c_path = sar_c_path if sar_c_path and os.path.exists(sar_c_path) else None
                        sar_l_path = sar_l_path if sar_l_path and os.path.exists(sar_l_path) else None
                    elif input_mode == "Use GEE (Sentinel-2)":
                        status.write("Using GEE arrays from session...")
                        red_arr = st.session_state.get("gee_red")
                        nir_arr = st.session_state.get("gee_nir")
                        sar_arr = st.session_state.get("gee_s1") if st.session_state.get("gee_s1_ready") else None
                        if red_arr is None or nir_arr is None:
                            raise RuntimeError("GEE data missing. Please fetch again.")
                        (
                            efc_path,
                            metrics,
                            summary,
                            red_path,
                            nir_path,
                            temp_inputs_dir,
                        ) = run_fusion_from_gee_arrays(
                            red_arr, nir_arr, output_dir=output_dir, c_sar_arr=sar_arr
                        )
                        sar_c_path = None
                        sar_l_path = None
                        if st.session_state.get("gee_params"):
                            tile_metadata = dict(st.session_state.get("gee_params"))
                    
                    if input_mode != "Use GEE (Sentinel-2)":
                        status.write("Running fusion pipeline...")
                        # Run pipeline
                        efc_path, metrics, summary = run_pipeline(
                            red_path, nir_path, sar_c_path, sar_l_path, output_dir
                        )
                        tile_metadata = None
                    
                    status.write("Computing insights...")
                    # Compute NDVI summary for insights
                    red_da = open_da(red_path)
                    nir_da = open_da(nir_path)
                    if red_da.rio.crs != nir_da.rio.crs or red_da.shape != nir_da.shape:
                        nir_da = nir_da.rio.reproject_match(red_da)
                    
                    summary = summarize_ndvi(red_da, nir_da)
                    st.session_state["fusion_summary"] = summary
                    
                    # Save summary to outputs
                    os.makedirs("outputs", exist_ok=True)
                    with open("outputs/fusion_summary.json", 'w') as f:
                        json.dump(summary, f, indent=2)
                    
                    status.write("Generating report...")
                    # Generate report
                    report_path = generate_report(output_dir)
                    
                    # Auto-cleanup uploaded files
                    if input_mode == "Upload files":
                        status.write("Cleaning up temporary files...")
                        try:
                            shutil.rmtree(upload_dir)
                        except:
                            pass  # Ignore cleanup errors
                    
                    status.update(label="✅ Fusion completed successfully!", state="complete")
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Enhanced Forest Composite")
                        ai_mask = None
                        if ai_mode == "EFC + AI detections" and AI_AVAILABLE:
                            try:
                                model_path = st.secrets.get("ai", {}).get(
                                    "unet_model_path", DEFAULT_UNET_MODEL_PATH
                                )
                                ai_mask = run_unet_on_efc_tile(
                                    open_da(efc_path).data.transpose(1, 2, 0),
                                    model_path=model_path,
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                )

                                def make_overlay_from_mask(mask_arr, alpha: float = 0.4):
                                    h, w = mask_arr.shape
                                    overlay = np.zeros((h, w, 4), dtype=np.uint8)
                                    # 0 = background (transparent), 1 = vegetation (green), 2 = deforested (red).
                                    overlay[mask_arr == 1] = [0, 255, 0, int(alpha * 255)]
                                    overlay[mask_arr == 2] = [255, 0, 0, int(alpha * 255)]
                                    return overlay

                                overlay = make_overlay_from_mask(ai_mask)
                                st.image(
                                    [efc_path, overlay],
                                    caption=["EFC Visualization", "AI detections"],
                                    use_container_width=True,
                                )
                            except Exception as e:  # noqa: BLE001
                                st.warning(f"AI overlay unavailable ({e}); showing EFC only.")
                                st.image(efc_path, caption="EFC Visualization", use_container_width=True)
                        else:
                            if ai_mode == "EFC + AI detections" and not AI_AVAILABLE:
                                st.warning(f"AI overlay is disabled: {AI_IMPORT_ERROR}")
                            st.image(efc_path, caption="EFC Visualization", use_container_width=True)
                        efc_rgb_current = open_da(efc_path).data.transpose(1, 2, 0).astype(np.uint8)
                        st.session_state["efc_rgb"] = efc_rgb_current
                        st.session_state["efc_tile_metadata"] = tile_metadata
                        if show_raw_ai_mask and ai_mask is not None:
                            try:
                                h, w = ai_mask.shape
                                scale = 4
                                rgb = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
                                k = np.ones((scale, scale), dtype=np.uint8)
                                m0 = np.kron((ai_mask == 0).astype(np.uint8), k).astype(bool)
                                m1 = np.kron((ai_mask == 1).astype(np.uint8), k).astype(bool)
                                m2 = np.kron((ai_mask == 2).astype(np.uint8), k).astype(bool)
                                rgb[m0] = (30, 30, 30)
                                rgb[m1] = (0, 200, 0)
                                rgb[m2] = (200, 0, 0)

                                img = Image.fromarray(rgb, mode="RGB")
                                draw = ImageDraw.Draw(img)
                                font = ImageFont.load_default()
                                for y in range(h):
                                    for x in range(w):
                                        cls = int(ai_mask[y, x])
                                        txt = str(cls)
                                        cx = (x + 0.5) * scale
                                        cy = (y + 0.5) * scale
                                        draw.text((cx, cy), txt, font=font, fill=(255, 255, 255), anchor="mm")

                                st.subheader("Raw AI mask with numeric labels (0/1/2)")
                                st.image(
                                    img,
                                    caption="0 = background, 1 = vegetation, 2 = deforested/change",
                                    use_container_width=True,
                                )
                            except Exception as e:  # noqa: BLE001
                                st.warning(f"Failed to render raw AI mask with labels: {e}")

                except Exception as e:  # noqa: BLE001
                    status.update(label="❌ Processing failed", state="error")
                    st.error(f"❌ Error processing files: {str(e)}")
                    st.info("Please ensure files are valid GeoTIFF format.")
                finally:
                    if temp_inputs_dir:
                        shutil.rmtree(temp_inputs_dir, ignore_errors=True)
                    loader_placeholder.empty()

    # Dataset tools (runs regardless of run_fusion)
    render_dataset_tools()

with change_tab:
    from spectra_core.pipeline_change import align_and_stack, compute_change, write_artifacts
    from spectra_core.nl import make_change_brief

    st.header("🔍 Change Detection")
    st.markdown("Compare BEFORE vs AFTER imagery to detect vegetation changes.")

    # Change detection inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 BEFORE")
        if input_mode == "Upload files":
            before_red = st.file_uploader("RED Band (B04)", type=['tif', 'tiff'], key="before_red")
            before_nir = st.file_uploader("NIR Band (B08)", type=['tif', 'tiff'], key="before_nir")
            before_sar_c = st.file_uploader("C-band SAR (optional)", type=['tif', 'tiff'], key="before_sar_c")
        else:
            before_red_path = st.text_input("RED Path", value="data/before_S2_B04.tif", key="before_red_path")
            before_nir_path = st.text_input("NIR Path", value="data/before_S2_B08.tif", key="before_nir_path")
            before_sar_c_path = st.text_input("C-band SAR Path (optional)", value="", key="before_sar_c_path")

    with col2:
        st.subheader("🟢 AFTER")
        if input_mode == "Upload files":
            after_red = st.file_uploader("RED Band (B04)", type=['tif', 'tiff'], key="after_red")
            after_nir = st.file_uploader("NIR Band (B08)", type=['tif', 'tiff'], key="after_nir")
            after_sar_c = st.file_uploader("C-band SAR (optional)", type=['tif', 'tiff'], key="after_sar_c")
        else:
            after_red_path = st.text_input("RED Path", value="data/after_S2_B04.tif", key="after_red_path")
            after_nir_path = st.text_input("NIR Path", value="data/after_S2_B08.tif", key="after_nir_path")
            after_sar_c_path = st.text_input("C-band SAR Path (optional)", value="", key="after_sar_c_path")
    
    # Parameters
    st.subheader("⚙️ Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        ndvi_min = st.number_input("Min NDVI Threshold", value=0.4, min_value=0.0, max_value=1.0, step=0.1)
        ndvi_drop = st.number_input("NDVI Drop Threshold", value=-0.15, min_value=-1.0, max_value=0.0, step=0.05)
    with col2:
        min_patch = st.number_input("Min Patch Size (pixels)", value=100, min_value=1, step=10)
        use_sar = st.checkbox("Use SAR for Change Detection", value=True)
    with col3:
        aoi_name = st.text_input("AOI Name (optional)", value="")
    
    if st.button("🔍 Run Change Detection", type="primary"):
        # Validate inputs
        if input_mode == "Upload files":
            if not before_red or not before_nir or not after_red or not after_nir:
                st.error("❌ Please upload BEFORE and AFTER RED/NIR bands.")
            else:
                valid_change_inputs = True
        else:
            required_paths = [before_red_path, before_nir_path, after_red_path, after_nir_path]
            if not all(p and os.path.exists(p) for p in required_paths):
                st.error("❌ Please provide valid paths for BEFORE and AFTER RED/NIR bands.")
                valid_change_inputs = False
            else:
                valid_change_inputs = True
        
        if valid_change_inputs:
            with st.status("Processing change detection...", expanded=True) as status:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                upload_dir = f"uploads/change_{timestamp}"
                output_dir = f"outputs/change_{timestamp}"
                os.makedirs(upload_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                
                try:
                    if input_mode == "Upload files":
                        status.write("Saving uploaded files...")
                        # Save before files
                        before_red_path = os.path.join(upload_dir, "before_red.tif")
                        before_nir_path = os.path.join(upload_dir, "before_nir.tif")
                        with open(before_red_path, "wb") as f:
                            f.write(before_red.read())
                        with open(before_nir_path, "wb") as f:
                            f.write(before_nir.read())
                        
                        # Save after files
                        after_red_path = os.path.join(upload_dir, "after_red.tif")
                        after_nir_path = os.path.join(upload_dir, "after_nir.tif")
                        with open(after_red_path, "wb") as f:
                            f.write(after_red.read())
                        with open(after_nir_path, "wb") as f:
                            f.write(after_nir.read())
                        
                        # SAR files
                        before_sar_c_path = None
                        after_sar_c_path = None
                        if before_sar_c:
                            before_sar_c_path = os.path.join(upload_dir, "before_sar_c.tif")
                            with open(before_sar_c_path, "wb") as f:
                                f.write(before_sar_c.read())
                        if after_sar_c:
                            after_sar_c_path = os.path.join(upload_dir, "after_sar_c.tif")
                            with open(after_sar_c_path, "wb") as f:
                                f.write(after_sar_c.read())
                    else:
                        # Clean empty paths
                        before_sar_c_path = before_sar_c_path if before_sar_c_path and os.path.exists(before_sar_c_path) else None
                        after_sar_c_path = after_sar_c_path if after_sar_c_path and os.path.exists(after_sar_c_path) else None
                    
                    # Prepare path dictionaries
                    before_paths = {'red': before_red_path, 'nir': before_nir_path}
                    after_paths = {'red': after_red_path, 'nir': after_nir_path}
                    
                    if before_sar_c_path:
                        before_paths['sar_c'] = before_sar_c_path
                    if after_sar_c_path:
                        after_paths['sar_c'] = after_sar_c_path
                    
                    status.write("Aligning imagery...")
                    stacked = align_and_stack(before_paths, after_paths)
                    
                    status.write("Computing changes...")
                    result = compute_change(stacked, ndvi_min, ndvi_drop, min_patch, use_sar)
                    
                    status.write("Writing artifacts...")
                    artifacts = write_artifacts(result, stacked, output_dir)
                    
                    # Generate NL brief
                    brief = make_change_brief(result['metrics'], result['polygons_gj'], aoi_name, result['confidence'])
                    
                    # Generate report
                    report_path = generate_report(output_dir)
                    
                    # Auto-cleanup
                    if input_mode == "Upload files":
                        status.write("Cleaning up temporary files...")
                        try:
                            shutil.rmtree(upload_dir)
                        except:
                            pass
                    
                    status.update(label="✅ Change detection completed!", state="complete")
                    
                    # Display results
                    st.subheader("📊 Change Detection Results")
                    
                    # Brief
                    st.info(brief)
                    
                    # Visualizations
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.image(artifacts['before_png'], caption="Before", use_container_width=True)
                    with col2:
                        st.image(artifacts['after_png'], caption="After", use_container_width=True)
                    with col3:
                        st.image(artifacts['delta_png'], caption="ΔNDVI", use_container_width=True)
                    with col4:
                        st.image(artifacts['mask_png'], caption="Change Mask", use_container_width=True)
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Metrics")
                        metrics_table = {
                            "Changed Pixels": result['metrics']['changed_pixels'],
                            "Area Changed": f"{result['metrics']['pct_area']:.2f}%",
                            "Mean ΔNDVI": f"{result['metrics']['mean_dndvi']:.3f}",
                            "Confidence": f"{result['confidence']:.2f}"
                        }
                        if result['metrics']['mean_dsar'] is not None:
                            metrics_table["Mean ΔSAR"] = f"{result['metrics']['mean_dsar']:.2f} dB"
                        st.table(metrics_table)
                    
                    with col2:
                        st.subheader("Polygons")
                        if result['polygons_gj']['features']:
                            poly_data = []
                            for i, feat in enumerate(result['polygons_gj']['features']):
                                poly_data.append({
                                    "ID": i+1,
                                    "Area (ha)": feat['properties']['area_ha']
                                })
                            st.dataframe(poly_data)
                        else:
                            st.info("No change polygons detected")
                    
                    # Downloads
                    st.subheader("Downloads")
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(artifacts['polygons_path'], "rb") as f:
                            st.download_button(
                                "🗺️ Download Polygons (GeoJSON)",
                                f.read(),
                                file_name="change_polygons.geojson",
                                mime="application/json"
                            )
                    with col2:
                        with open(report_path, "rb") as f:
                            st.download_button(
                                "📄 Download Change Report",
                                f.read(),
                                file_name="change_report.html",
                                mime="text/html"
                            )
                    
                except Exception as e:
                    status.update(label="❌ Change detection failed", state="error")
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please ensure all files are valid GeoTIFF format and cover the same area.")
    
    # Info section
    with st.expander("ℹ️ About Change Detection"):
        st.markdown("""
        **Change Detection Method:**
        - Computes ΔNDVI (AFTER - BEFORE) and optional ΔSAR
        - Identifies areas with significant vegetation loss
        - Creates binary change mask and polygons
        - Provides confidence scoring and natural language summary
        
        **Parameters:**
        - **Min NDVI**: Minimum vegetation threshold for analysis
        - **NDVI Drop**: Threshold for significant vegetation loss
        - **Min Patch**: Minimum contiguous pixels for change detection
        - **Use SAR**: Include SAR backscatter in change analysis
        """)

with agentic_tab:
    st.header("🤖 Agentic Insights")
    st.markdown("AI-powered analysis of vegetation health and fusion results.")
    st.caption(
        f"Gemini configured: {has_env_or_secret('GEMINI_API_KEY')} · Model: {get_env_or_secret('GEMINI_MODEL','(unset)')}"
    )
    
    # LLM Controls
    st.subheader("🔧 LLM Controls")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        llm_enabled = st.toggle("LLM mode", key="llm_enabled", value=os.getenv("SPECTRA_USE_LLM", "0") == "1")
    
    with col2:
        provider_options = [
            "Rule-based (OFF)",
            "Local (llama-cpp)", 
            "Online (Gemini)",
            "Online (HuggingFace)", 
            "Online (OpenAI-compatible)",
            "Ollama (local)"
        ]
        provider_ids = ["off", "local", "gemini", "hf", "openai", "ollama"]
        
        # Default selection based on env
        default_idx = 0
        if os.getenv("SPECTRA_USE_LLM", "0") == "1":
            mode = os.getenv("LLM_MODE", "local")
            if mode in provider_ids:
                default_idx = provider_ids.index(mode)
        
        selected_provider = st.selectbox(
            "Provider", 
            provider_options, 
            index=default_idx,
            key="provider_select"
        )
        provider_id = provider_ids[provider_options.index(selected_provider)]

    with st.expander("Developer options", expanded=False):
        debug_checked = st.checkbox(
            "LLM debug (prints one raw Gemini response to logs)",
            value=os.getenv("SPECTRA_LLM_DEBUG", "0") in ("1", "true", "True"),
        )
        os.environ["SPECTRA_LLM_DEBUG"] = "1" if debug_checked else "0"
    
    # Usage limits with sliders
    if llm_enabled and provider_id != "off":
        st.subheader("🛡️ Safety Limits")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_tokens = st.slider(
                "Max tokens per answer", 
                64, 1024, 
                int(os.getenv("LLM_ONLINE_MAX_TOKENS", "256")), 
                64
            )
        
        with col2:
            max_calls_session = st.slider(
                "Session budget (# answers)", 
                1, 50, 
                int(os.getenv("LLM_MAX_CALLS_SESSION", "10")), 
                1
            )
        
        with col3:
            max_calls_day = st.slider(
                "Daily budget (# answers)", 
                5, 500, 
                int(os.getenv("LLM_MAX_CALLS_DAY", "100")), 
                5
            )
        
        # Panic button
        if st.button("🚨 Panic: Disable LLM for this session", type="secondary"):
            st.session_state["llm_enabled"] = False
            st.session_state["provider_select"] = "Rule-based (OFF)"
            reset_session()
            st.success("LLM disabled and session budget reset!")
            st.rerun()
    
    # Status display
    usage = get_usage()
    session_calls = usage["session"]["calls"]
    daily_calls = usage["daily"]["calls"]
    active_model = get_env_or_secret("GEMINI_MODEL", "gemini-2.5-flash") if provider_id == "gemini" else ""
    
    if llm_enabled and provider_id != "off":
        if provider_id == "local":
            st.success(f"🟢 LLM: ON (Local) • Session: {session_calls}/{max_calls_session} • Today: {daily_calls}/{max_calls_day}")
        elif provider_id == "ollama":
            st.info(f"🔵 LLM: ON (Ollama) • Session: {session_calls}/{max_calls_session} • Today: {daily_calls}/{max_calls_day}")
        else:
            provider_label = "Gemini" if provider_id == "gemini" else "Online"
            model_suffix = f" • Model: {active_model}" if active_model else ""
            st.info(f"🔵 LLM: ON ({provider_label}) • Session: {session_calls}/{max_calls_session} • Today: {daily_calls}/{max_calls_day}{model_suffix}")
    else:
        st.info(f"⚪ LLM: OFF • Session: {session_calls} • Today: {daily_calls}")
    
    st.divider()
    
    # Try to get summary from session state or file
    summary = st.session_state.get("fusion_summary")
    if not summary:
        summary = load_fusion_summary("outputs/fusion_summary.json")
    
    if not summary:
        st.info("🔄 Run Fusion first to generate insights.")
        st.markdown("The Agentic Insights tab provides AI-powered analysis of your NDVI and vegetation health results.")
    else:
        # Display key metrics
        st.subheader("📊 Key Metrics")
        if summary.get("stats"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean NDVI", f"{summary['stats']['mean']:.3f}")
            with col2:
                st.metric("Area < 0.2", f"{summary['area']['below_02']*100:.1f}%")
            with col3:
                st.metric("Stressed Regions", summary['regions']['count'])
            with col4:
                st.metric("Largest Cluster", f"{summary['regions']['largest_px']} px")
        
        # Generate and display insights
        bullets = insight_bullets(summary)
        story = narrative(summary)
        
        st.subheader("💡 Key Insights")
        for bullet in bullets:
            st.markdown(f"• {bullet}")
        
        st.subheader("📝 Analysis Summary")
        st.markdown(story)
        
        # Q&A Section
        st.subheader("❓ Ask Questions")
        
        # Create provider based on UI selection
        provider = None
        if llm_enabled and provider_id != "off":
            try:
                if provider_id == "local":
                    model_path = os.getenv("LLM_MODEL_PATH")
                    if model_path and os.path.exists(model_path):
                        provider = LocalLlamaCpp(model_path)
                elif provider_id == "gemini":
                    api_key = get_env_or_secret("GEMINI_API_KEY", "")
                    model = get_env_or_secret("GEMINI_MODEL", "gemini-2.5-flash")
                    base_url = get_env_or_secret("GEMINI_BASE_URL", "")
                    if api_key:
                        provider = GeminiProvider(api_key=api_key, model=model, base_url=base_url or None)
                elif provider_id == "hf":
                    token = os.getenv("HF_API_TOKEN")
                    model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
                    if token:
                        provider = HuggingFaceProvider(token, model)
                elif provider_id == "openai":
                    base_url = os.getenv("LLM_BASE_URL")
                    api_key = os.getenv("LLM_API_KEY")
                    model = os.getenv("LLM_MODEL")
                    if all([base_url, api_key, model]):
                        provider = OpenAICompatible(base_url, api_key, model)
                elif provider_id == "ollama":
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    model = os.getenv("LLM_MODEL", "llama3.2:3b")
                    provider = OllamaProvider(base_url, model)
            except Exception as e:
                st.warning(f"⚠️ Provider error: {str(e)}")
                provider = None
        
        # Show configuration help if provider not available
        if llm_enabled and provider_id != "off" and not provider:
            if provider_id == "local":
                st.warning("⚠️ Set LLM_MODEL_PATH to use local mode")
            elif provider_id == "gemini":
                missing_gemini_key = (provider_id == "gemini") and (not has_env_or_secret("GEMINI_API_KEY"))
                if missing_gemini_key:
                    st.warning(
                        "Provider error: Gemini API key not configured. Set GEMINI_API_KEY via env or Streamlit secrets.",
                        icon="⚠️",
                    )
            elif provider_id == "hf":
                st.warning("⚠️ Set HF_API_TOKEN for Hugging Face mode")
            elif provider_id == "openai":
                st.warning("⚠️ Set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL for OpenAI mode")
        
        question = st.text_input("Ask a question about these results...", 
                                placeholder="e.g., What does the mean NDVI tell us about vegetation health?")
        
        if question:
            with st.spinner("Analyzing..."):
                # Build caps from UI or env
                caps = llm_caps_from_env()
                if llm_enabled and provider_id != "off":
                    caps.update({
                        "max_tokens": max_tokens,
                        "max_calls_session": max_calls_session,
                        "max_calls_day": max_calls_day
                    })
                
                answer = qa_answer(
                    question, 
                    {"summary": summary}, 
                    provider=provider,
                    caps=caps,
                    enabled=llm_enabled and provider_id != "off" and provider is not None
                )
                st.markdown(f"**Answer:** {answer}")
        
        # Download options
        st.subheader("📥 Downloads")
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate insights text
            insights_text = "VEGETATION INSIGHTS\n\n"
            insights_text += "KEY FINDINGS:\n"
            for bullet in bullets:
                insights_text += f"• {bullet}\n"
            insights_text += f"\nANALYSIS:\n{story}"
            
            st.download_button(
                "📄 Download Insights (TXT)",
                insights_text,
                file_name="vegetation_insights.txt",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                "📊 Download Summary (JSON)",
                json.dumps(summary, indent=2),
                file_name="fusion_summary.json",
                mime="application/json"
            )
    
    # Info section
    with st.expander("ℹ️ About Agentic Insights"):
        st.markdown("""
        **Features:**
        - **Rule-based Analysis**: Zero-cost vegetation health assessment
        - **Statistical Summary**: NDVI distribution, area fractions, region analysis
        - **Natural Language**: Human-readable insights and recommendations
        - **Q&A System**: Interactive questions about your results
        
        **Optional LLM Mode:**
        - Set `SPECTRA_USE_LLM=1` environment variable
        - Provide `LLM_MODEL_PATH` pointing to a GGUF model file
        - Enables AI-powered responses to complex questions
        - Runs locally (no external API calls)
        
        **Example Questions:**
        - "What does the mean NDVI tell us?"
        - "How much area shows vegetation stress?"
        - "What could cause low NDVI values?"
        
        **LLM Controls:**
        - Toggle LLM on/off per session
        - Select provider from dropdown
        - Adjust safety limits (tokens, session/daily budgets)
        - Panic button to instantly disable LLM
        
        **Usage Tracking:**
        - Per-session and daily call limits
        - Automatic daily reset at midnight
        - Response caching to avoid repeat charges
        """)

with news_tab:
    st.subheader("Environmental News for SPECTRA Analysis")
    st.caption("Live headlines on deforestation, land-use change, and climate / weather risks.")

    topic = st.radio(
        "Focus",
        ["All environment", "Deforestation & land use", "Weather & climate"],
        horizontal=True,
    )

    if st.button("Refresh news"):
        st.session_state["last_news_topic"] = topic

    active_topic = st.session_state.get("last_news_topic", topic)

    with st.spinner(f"Fetching {active_topic.lower()} news…"):
        articles = fetch_environment_news(active_topic)

    for article in articles:
        with st.container():
            st.markdown(
                f"**[{article['title']}]({article['url']})**  \n"
                f"*Source:* {article['source']}"
            )
            if article["published_at"]:
                try:
                    dt = datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
                    st.caption(dt.strftime("Published on %Y-%m-%d %H:%M UTC"))
                except Exception:
                    pass
            if article["summary"]:
                st.write(article["summary"])
            st.markdown("---")
