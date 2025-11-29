import streamlit as st
import os
import sys
import tempfile
import shutil
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra_core.pipeline import run_pipeline
from spectra_core.report import generate_report

st.set_page_config(page_title="SPECTRA Fusion", page_icon="🛰️", layout="wide")

st.title("🛰️ SPECTRA Enhanced Forest Composite")
st.markdown("Upload Sentinel-2 optical bands and optional SAR data to generate Enhanced Forest Composite visualizations.")

# Sidebar for data input
st.sidebar.header("Data Input")

# Mode selector
mode = st.sidebar.radio("Input Mode", ["Upload files", "Use server files"])

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
                    st.image(efc_path, caption="EFC Visualization", use_container_width=True)
                
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
