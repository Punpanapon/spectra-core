# ENTRYPOINT FOR STREAMLIT COMMUNITY CLOUD
# Chosen automatically by Codex based on project structure.

import os
import sys
import shutil
from datetime import datetime
import json
import requests
import streamlit as st

# Secrets/environment handling for Streamlit Cloud and local use
secrets_llm = st.secrets.get("llm", {}) if hasattr(st, "secrets") else {}
secrets_news = st.secrets.get("newsdata", {}) if hasattr(st, "secrets") else {}
secrets_ee = st.secrets.get("earthengine", {}) if hasattr(st, "secrets") else {}

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

# Mode selector
mode = st.sidebar.radio("Input Mode", ["Upload files", "Use server files"], index=1)

with efc_tab:
    red_file = None
    nir_file = None
    sar_c_file = None
    sar_l_file = None
    red_path = None
    nir_path = None
    sar_c_path = None
    sar_l_path = None

    if mode == "Upload files":
        red_file = st.sidebar.file_uploader("RED Band (B04) - Required", type=['tif', 'tiff'], key="red")
        nir_file = st.sidebar.file_uploader("NIR Band (B08) - Required", type=['tif', 'tiff'], key="nir")
        sar_c_file = st.sidebar.file_uploader("C-band SAR - Optional", type=['tif', 'tiff'], key="sar_c")
        sar_l_file = st.sidebar.file_uploader("L-band SAR - Optional", type=['tif', 'tiff'], key="sar_l")
        
        # Show file size warnings
        if red_file and red_file.size > 1e9:
            st.sidebar.warning("⚠️ Large file detected. Consider using server files for better performance.")
        if nir_file and nir_file.size > 1e9:
            st.sidebar.warning("⚠️ Large file detected. Consider using server files for better performance.")
    else:
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

    loader_placeholder = st.empty()

    if st.sidebar.button("🚀 Run Fusion", type="primary"):
        # Validate inputs based on mode
        if mode == "Upload files":
            if not red_file or not nir_file:
                st.error("❌ Please upload both RED and NIR bands to proceed.")
            else:
                valid_inputs = True
        else:
            if not red_path or not nir_path or not os.path.exists(red_path) or not os.path.exists(nir_path):
                st.error("❌ Please provide valid paths for both RED and NIR bands.")
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
                
                try:
                    if mode == "Upload files":
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
                    else:
                        status.write("Using server files...")
                        # Use server paths, clean empty strings
                        sar_c_path = sar_c_path if sar_c_path and os.path.exists(sar_c_path) else None
                        sar_l_path = sar_l_path if sar_l_path and os.path.exists(sar_l_path) else None
                    
                    status.write("Running fusion pipeline...")
                    # Run pipeline
                    efc_path, metrics, summary = run_pipeline(
                        red_path, nir_path, sar_c_path, sar_l_path, output_dir
                    )
                    
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
                    if mode == "Upload files":
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
                        st.image(efc_path, caption="EFC Visualization", use_column_width=True)
                    
                    with col2:
                        st.subheader("Summary")
                        st.text(summary)
                        
                        st.subheader("Metrics")
                        metrics_table = {
                            "NDVI Min": f"{metrics['ndvi_min']:.3f}",
                            "NDVI Max": f"{metrics['ndvi_max']:.3f}",
                            "NDVI Mean": f"{metrics['ndvi_mean']:.3f}",
                            "C-band SAR": "Yes" if metrics['has_sar_c'] else "No",
                            "L-band SAR": "Yes" if metrics['has_sar_l'] else "No"
                        }
                        st.table(metrics_table)
                    
                    # Download buttons
                    st.subheader("Downloads")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with open(efc_path, "rb") as f:
                            st.download_button(
                                "📥 Download EFC Image",
                                f.read(),
                                file_name="efc.png",
                                mime="image/png"
                            )
                    
                    with col2:
                        with open(report_path, "rb") as f:
                            st.download_button(
                                "📄 Download HTML Report",
                                f.read(),
                                file_name="report.html",
                                mime="text/html"
                            )
                    
                except Exception as e:
                    status.update(label="❌ Processing failed", state="error")
                    st.error(f"❌ Error processing files: {str(e)}")
                    st.info("Please ensure files are valid GeoTIFF format.")
                finally:
                    loader_placeholder.empty()

    # Info section
    with st.expander("ℹ️ About Enhanced Forest Composite"):
        st.markdown("""
        **EFC Formula:**
        - **R Channel**: 1 - NDVI (inverted vegetation)
        - **G Channel**: NDVI (vegetation index)
        - **B Channel**: Normalized SAR dB (C-band and/or L-band)
        
        **Required Files:**
        - RED band (Sentinel-2 B04)
        - NIR band (Sentinel-2 B08)
        
        **Optional Files:**
        - C-band SAR (Sentinel-1 or other)
        - L-band SAR (ALOS PALSAR or other)
        """)

with change_tab:
    from spectra_core.pipeline_change import align_and_stack, compute_change, write_artifacts
    from spectra_core.nl import make_change_brief
    
    st.header("🔍 Change Detection")
    st.markdown("Compare BEFORE vs AFTER imagery to detect vegetation changes.")
    
    # Change detection inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 BEFORE")
        if mode == "Upload files":
            before_red = st.file_uploader("RED Band (B04)", type=['tif', 'tiff'], key="before_red")
            before_nir = st.file_uploader("NIR Band (B08)", type=['tif', 'tiff'], key="before_nir")
            before_sar_c = st.file_uploader("C-band SAR (optional)", type=['tif', 'tiff'], key="before_sar_c")
        else:
            before_red_path = st.text_input("RED Path", value="data/before_S2_B04.tif", key="before_red_path")
            before_nir_path = st.text_input("NIR Path", value="data/before_S2_B08.tif", key="before_nir_path")
            before_sar_c_path = st.text_input("C-band SAR Path (optional)", value="", key="before_sar_c_path")
    
    with col2:
        st.subheader("🟢 AFTER")
        if mode == "Upload files":
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
        if mode == "Upload files":
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
                    if mode == "Upload files":
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
                    if mode == "Upload files":
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
                        st.image(artifacts['before_png'], caption="Before", use_column_width=True)
                    with col2:
                        st.image(artifacts['after_png'], caption="After", use_column_width=True)
                    with col3:
                        st.image(artifacts['delta_png'], caption="ΔNDVI", use_column_width=True)
                    with col4:
                        st.image(artifacts['mask_png'], caption="Change Mask", use_column_width=True)
                    
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
