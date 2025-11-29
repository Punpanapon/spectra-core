import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


@st.cache_data
def load_earth_image_base64() -> str:
    """
    Load the realistic Earth PNG from disk and return it as a base64 string
    for embedding into an HTML <img> as a data URL.
    """
    img_path = Path(__file__).parent / "assets" / "spectra_earth_realistic.png"
    with img_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def show_spectra_loader(
    message: str = "RUNNING SPECTRA FUSION ON SATELLITE DATA…",
) -> None:
    earth_b64 = load_earth_image_base64()

    loader_html = f"""
    <div class="spectra-loader">
      <div class="spectra-orbit-container">
        <div class="spectra-glow"></div>

        <div class="spectra-planet">
          <img src="data:image/png;base64,{earth_b64}" alt="Earth" class="spectra-earth-img" />
        </div>

        <div class="spectra-orbit orbit-inner">
          <div class="spectra-satellite sat-1">
            <div class="sat-panel left"></div>
            <div class="sat-body"></div>
            <div class="sat-panel right"></div>
          </div>
        </div>

        <div class="spectra-orbit orbit-outer">
          <div class="spectra-satellite sat-2">
            <div class="sat-panel left"></div>
            <div class="sat-body"></div>
            <div class="sat-panel right"></div>
          </div>
        </div>
      </div>

      <div class="spectra-text">{message}</div>
    </div>

    <style>
    .spectra-loader {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2.5rem 1rem 1.5rem;
      gap: 1rem;
    }}

    .spectra-orbit-container {{
      position: relative;
      width: 180px;
      height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .spectra-glow {{
      position: absolute;
      width: 150px;
      height: 150px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(45, 212, 191, 0.35) 0%, rgba(15, 23, 42, 0) 70%);
      filter: blur(2px);
    }}

    .spectra-planet {{
      position: relative;
      width: 92px;
      height: 92px;
      border-radius: 50%;
      overflow: hidden;
      box-shadow:
        0 10px 16px rgba(15, 23, 42, 0.9),
        0 0 30px rgba(56, 189, 248, 0.55);
    }}

    .spectra-earth-img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}

    .spectra-orbit {{
      position: absolute;
      border-radius: 50%;
      border: 2px dashed rgba(148, 163, 184, 0.85);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }}

    .orbit-inner {{
      width: 130px;
      height: 130px;
      top: 50%;
      left: 50%;
      transform-origin: center center;
      animation: orbit-spin-inner 10s linear infinite;
    }}

    .orbit-outer {{
      width: 160px;
      height: 160px;
      top: 50%;
      left: 50%;
      transform-origin: center center;
      animation: orbit-spin-outer 16s linear infinite;
    }}

    .spectra-satellite {{
      position: absolute;
      top: 50%;
      left: 100%;
      transform: translate(-50%, -50%);
      display: flex;
      align-items: center;
      gap: 2px;
    }}

    .sat-body {{
      width: 18px;
      height: 11px;
      border-radius: 3px;
      background: linear-gradient(135deg, #e5e7eb, #cbd5f5);
      box-shadow:
        0 0 6px rgba(248, 250, 252, 0.9),
        0 0 14px rgba(56, 189, 248, 0.7);
    }}

    .sat-panel {{
      width: 13px;
      height: 8px;
      border-radius: 2px;
      background: linear-gradient(90deg, #0f172a, #1f2937, #0f172a);
      box-shadow: 0 0 4px rgba(59, 130, 246, 0.85);
    }}

    .sat-panel.left {{
      margin-right: 2px;
    }}

    .sat-panel.right {{
      margin-left: 2px;
    }}

    .spectra-text {{
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #e5e7eb;
      text-align: center;
    }}

    @keyframes orbit-spin-inner {{
      from {{ transform: translate(-50%, -50%) rotate(0deg); }}
      to   {{ transform: translate(-50%, -50%) rotate(360deg); }}
    }}

    @keyframes orbit-spin-outer {{
      from {{ transform: translate(-50%, -50%) rotate(0deg); }}
      to   {{ transform: translate(-50%, -50%) rotate(-360deg); }}
    }}
    </style>
    """
    components.html(loader_html, height=260, width=260)
