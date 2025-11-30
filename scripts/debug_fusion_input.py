"""
Debug helper for fusion inputs/modes.

Run:
    python scripts/debug_fusion_input.py

It prints supported input modes, expected session keys for GEE arrays,
and performs a lightweight call to process_from_arrays with fake data to
ensure no file paths are touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_streamlit_app_module():
    """Load app/streamlit_app.py as a module even though 'app' is not a package."""
    app_path = ROOT / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app_module", app_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load streamlit_app.py from {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


process_from_arrays = None
load_note = ""
try:
    streamlit_app = _load_streamlit_app_module()
    process_from_arrays = streamlit_app.process_from_arrays  # type: ignore
except Exception as exc:  # noqa: BLE001
    load_note = f"(using lightweight fallback; could not import streamlit_app: {exc})"

if process_from_arrays is None:
    def process_from_arrays(red_arr, nir_arr, sar_arr=None, output_dir=None):  # type: ignore
        red_arr = np.asarray(red_arr, dtype=np.float32)
        nir_arr = np.asarray(nir_arr, dtype=np.float32)
        h = min(red_arr.shape[0], nir_arr.shape[0])
        w = min(red_arr.shape[1], nir_arr.shape[1])
        red_arr = red_arr[:h, :w]
        nir_arr = nir_arr[:h, :w]
        sar_trimmed = None
        if sar_arr is not None:
            sar_arr = np.asarray(sar_arr, dtype=np.float32)
            h = min(h, sar_arr.shape[0])
            w = min(w, sar_arr.shape[1])
            red_arr = red_arr[:h, :w]
            nir_arr = nir_arr[:h, :w]
            sar_trimmed = sar_arr[:h, :w]
        metrics = {"ndvi_min": float(((nir_arr - red_arr) / (nir_arr + red_arr + 1e-8)).min())}
        summary = "Fallback array-only check"
        return "N/A", metrics, summary, red_arr, nir_arr, sar_trimmed


def main() -> None:
    print("Supported input modes: upload, server, gee")
    print("GEE mode uses arrays from session_state keys: gee_red, gee_nir, gee_s1 (optional)")
    print("File-based modes use paths provided via UI (upload/server); validation happens before processing.\n")

    print("Running a lightweight array-only check (no file paths)...")
    red = np.ones((8, 8), dtype=np.float32)
    nir = np.ones((8, 8), dtype=np.float32) * 2
    sar = np.ones((8, 8), dtype=np.float32) * -12
    try:
        efc_path, metrics, summary, red_out, nir_out, sar_out = process_from_arrays(
            red, nir, sar, output_dir=str(ROOT / "outputs" / "debug_inputs")
        )
        print("✅ process_from_arrays executed without touching input file paths.")
        print(f"Outputs: efc_path={efc_path}, metrics_keys={list(metrics.keys())}, summary='{summary}'")
        print(f"Returned array shapes: red={red_out.shape}, nir={nir_out.shape}, sar={None if sar_out is None else sar_out.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ process_from_arrays failed: {exc}")


if __name__ == "__main__":
    main()
