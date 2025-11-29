"""
Snowflake Streamlit edition of SPECTRA.

Based on: app/streamlit_app.py (primary Spectra app). Other candidate: app/streamlit_app_backup.py.
Major stubs/changes for Snowflake:
- SPECTRA fusion/change pipelines replaced with static demo data (no raster I/O).
- LLM providers and external API calls (Gemini/OpenAI/etc.) replaced by simple rule-based responses.
- NewsData API replaced with static demo articles (no outbound HTTP).
- Keeps tab/layout structure similar to the original app for familiarity.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ORIGINAL (requires external network and local modules)
# from spectra_core.pipeline import run_pipeline
# from spectra_core.report import generate_report
# from spectra_core.agent.insights import summarize_ndvi, insight_bullets, narrative, qa_answer, load_fusion_summary, llm_caps_from_env
# from spectra_core.agent.llm_providers import LocalLlamaCpp, GeminiProvider, HuggingFaceProvider, OpenAICompatible, OllamaProvider
# import requests


def get_demo_fusion_results() -> Dict[str, object]:
    """Return deterministic demo fusion outputs."""
    size = 180
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xv, yv = np.meshgrid(x, y)
    ndvi = (np.sin(xv * np.pi) * np.cos(yv * np.pi) + 1) / 2
    # Build RGB with inverted NDVI in red channel
    img = np.dstack(
        [
            1 - ndvi,
            ndvi,
            0.35 + 0.15 * np.sin(xv * 4 * np.pi) * np.cos(yv * 4 * np.pi),
        ]
    )
    metrics = {
        "ndvi_min": float(ndvi.min()),
        "ndvi_max": float(ndvi.max()),
        "ndvi_mean": float(ndvi.mean()),
        "has_sar_c": True,
        "has_sar_l": False,
    }
    summary_text = (
        "Vegetation is healthy in most of the scene with limited stressed patches. "
        "SAR adds texture but no major moisture anomalies detected."
    )
    regions = pd.DataFrame(
        [
            {"Region": "North ridge", "Area (px)": 540, "NDVI mean": 0.71},
            {"Region": "Valley floor", "Area (px)": 320, "NDVI mean": 0.48},
            {"Region": "South spur", "Area (px)": 210, "NDVI mean": 0.36},
        ]
    )
    return {"image": img, "metrics": metrics, "summary": summary_text, "regions": regions}


def get_demo_change_results() -> Dict[str, object]:
    """Return demo change detection outputs."""
    metrics = {
        "changed_pixels": 1240,
        "pct_area": 3.7,
        "mean_dndvi": -0.18,
        "mean_dsar": -0.6,
        "confidence": 0.82,
    }
    polygons = pd.DataFrame(
        [
            {"ID": 1, "Area (ha)": 12.4},
            {"ID": 2, "Area (ha)": 7.1},
            {"ID": 3, "Area (ha)": 3.5},
        ]
    )
    delta_img = np.clip(np.random.normal(loc=0.5, scale=0.18, size=(140, 140)), 0, 1)
    mask_img = (delta_img < 0.45).astype(float)
    return {
        "metrics": metrics,
        "polygons": polygons,
        "delta_image": delta_img,
        "mask_image": mask_img,
        "brief": (
            "Vegetation loss clusters along access tracks; most patches are small and "
            "isolated with moderate confidence."
        ),
    }


def get_news_stub(topic: str) -> List[Dict[str, str]]:
    """Static news articles to avoid outbound HTTP."""
    topic_suffix = {
        "All environment": "Environment",
        "Deforestation & land use": "Deforestation",
        "Weather & climate": "Climate",
    }.get(topic, "Environment")
    return [
        {
            "title": f"{topic_suffix}: Community-led restoration shows early gains",
            "source": "SPECTRA Demo Desk",
            "url": "https://example.org/article1",
            "published_at": "2024-03-12T08:00:00Z",
            "summary": "Local partners report canopy recovery and reduced erosion after targeted planting.",
        },
        {
            "title": f"{topic_suffix}: Sentinel-2 captures seasonal greening trends",
            "source": "SPECTRA Demo Desk",
            "url": "https://example.org/article2",
            "published_at": "2024-02-05T14:20:00Z",
            "summary": "Composite imagery highlights strong vegetation rebound following monsoon onset.",
        },
    ]


def qa_stub(question: str) -> str:
    """Lightweight Q&A that keeps everything local."""
    # Guard against non-string inputs to avoid TypeError on .lower()
    lowered = question.lower() if isinstance(question, str) else ""
    if "ndvi" in lowered:
        return "Higher NDVI implies healthier vegetation; focus follow-up on low NDVI clusters."
    if "change" in lowered or "loss" in lowered:
        return "Most losses are small patches; verify field reports or cloud masking before action."
    return "This is a demo response. In production, the LLM would tailor insights to your uploaded scene."


st.set_page_config(page_title="SPECTRA (Snowflake edition)", page_icon="🛰️", layout="wide")

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
          SPECTRA Core (Snowflake)
        </div>
        <div style="font-size: 0.9rem; color: #CBD5F5;">
          Demo build with static data (no external APIs).
        </div>
      </div>
      <div style="font-size: 1.5rem;">
        🌍🛰️🌲
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🛰️ SPECTRA Enhanced Forest Composite (Snowflake)")
st.markdown("Upload bands or use placeholders to preview the Spectra experience. Processing is stubbed for Snowflake.")

tabs = st.tabs(["EFC Fusion", "Change Detection", "Agentic Insights", "News"])
efc_tab, change_tab, agentic_tab, news_tab = tabs

st.sidebar.header("Data Input")
mode = st.sidebar.radio("Input Mode", ["Upload files", "Use placeholders"], index=1)

with efc_tab:
    st.subheader("Fusion")
    st.write("This demo uses synthetic NDVI and SAR textures to mimic the original pipeline.")

    red_file = st.sidebar.file_uploader("RED Band (B04) - optional", type=["tif", "tiff"], key="red_demo")
    nir_file = st.sidebar.file_uploader("NIR Band (B08) - optional", type=["tif", "tiff"], key="nir_demo")
    sar_c_file = st.sidebar.file_uploader("C-band SAR - optional", type=["tif", "tiff"], key="sar_c_demo")

    if st.sidebar.button("🚀 Run Fusion", type="primary"):
        with st.status("Generating demo fusion...", expanded=True) as status:
            status.write("Loading inputs (stubbed)...")
            status.write("Computing NDVI and EFC visualization (static demo)...")
            demo = get_demo_fusion_results()
            status.update(label="✅ Demo fusion ready", state="complete")

        st.subheader("Enhanced Forest Composite")
        st.image(demo["image"], caption="Synthetic EFC preview", use_column_width=True)

        st.subheader("Summary")
        st.write(demo["summary"])

        st.subheader("Metrics")
        st.table(
            {
                "NDVI Min": f"{demo['metrics']['ndvi_min']:.3f}",
                "NDVI Max": f"{demo['metrics']['ndvi_max']:.3f}",
                "NDVI Mean": f"{demo['metrics']['ndvi_mean']:.3f}",
                "C-band SAR": "Yes" if demo["metrics"]["has_sar_c"] else "No",
                "L-band SAR": "Yes" if demo["metrics"]["has_sar_l"] else "No",
            }
        )

        st.subheader("Regions")
        st.dataframe(demo["regions"], use_container_width=True, hide_index=True)

with change_tab:
    st.header("🔍 Change Detection")
    st.write("Before/After processing is stubbed; outputs show representative metrics.")

    st.columns(2)[0].markdown("**Inputs** (upload optional; not processed in demo)")
    st.file_uploader("BEFORE RED (B04)", type=["tif", "tiff"], key="before_red_demo")
    st.file_uploader("BEFORE NIR (B08)", type=["tif", "tiff"], key="before_nir_demo")
    st.file_uploader("AFTER RED (B04)", type=["tif", "tiff"], key="after_red_demo")
    st.file_uploader("AFTER NIR (B08)", type=["tif", "tiff"], key="after_nir_demo")

    if st.button("🔍 Run Change Detection", type="primary"):
        with st.status("Running demo change detection...", expanded=True) as status:
            status.write("Aligning imagery (stub)...")
            status.write("Computing ΔNDVI (stub)...")
            result = get_demo_change_results()
            status.update(label="✅ Demo change results ready", state="complete")

        st.subheader("📊 Metrics")
        st.table(
            {
                "Changed Pixels": result["metrics"]["changed_pixels"],
                "Area Changed": f"{result['metrics']['pct_area']:.2f}%",
                "Mean ΔNDVI": f"{result['metrics']['mean_dndvi']:.3f}",
                "Mean ΔSAR": f"{result['metrics']['mean_dsar']:.2f} dB",
                "Confidence": f"{result['metrics']['confidence']:.2f}",
            }
        )

        st.subheader("Brief")
        st.info(result["brief"])

        col1, col2 = st.columns(2)
        with col1:
            st.image(result["delta_image"], caption="ΔNDVI (demo)", use_column_width=True)
        with col2:
            st.image(result["mask_image"], caption="Change mask (demo)", use_column_width=True)

        st.subheader("Polygons (demo)")
        st.dataframe(result["polygons"], use_container_width=True, hide_index=True)

with agentic_tab:
    st.header("🤖 Agentic Insights (Demo)")
    st.markdown("LLM calls are disabled; responses are rule-based stubs.")
    summary_text = st.session_state.get("fusion_summary_demo")

    if not summary_text:
        demo = get_demo_fusion_results()
        summary_text = demo["summary"]
        st.session_state["fusion_summary_demo"] = summary_text

    st.subheader("Key Insights")
    st.markdown(f"• {summary_text}")
    st.markdown("• NDVI range is broad; focus on the lowest 10% for targeted checks.")
    st.markdown("• SAR backscatter indicates stable moisture, no flood signals detected.")

    question = st.text_input(
        "Ask a question about these results...",
        placeholder="e.g., Where should I inspect for vegetation loss?",
    )
    if question:
        st.markdown(f"**Answer:** {qa_stub(question)}")

with news_tab:
    st.subheader("Environmental News (Demo)")
    topic = st.radio(
        "Focus",
        ["All environment", "Deforestation & land use", "Weather & climate"],
        horizontal=True,
    )
    if st.button("Refresh news"):
        st.session_state["last_news_topic"] = topic
    active_topic = st.session_state.get("last_news_topic", topic)

    articles = get_news_stub(active_topic)
    for article in articles:
        with st.container():
            st.markdown(f"**[{article['title']}]({article['url']})**  \n*Source:* {article['source']}")
            published_at = article.get("published_at")
            # Safeguard: fromisoformat/replace expects a string; avoid TypeError when None/other types.
            if isinstance(published_at, str):
                try:
                    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    st.caption(dt.strftime("Published on %Y-%m-%d %H:%M UTC"))
                except Exception:
                    pass
            if article["summary"]:
                st.write(article["summary"])
            st.markdown("---")
