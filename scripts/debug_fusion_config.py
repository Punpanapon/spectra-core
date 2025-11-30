"""
Debug helper to inspect fusion model configuration and loadability.

Run:
    python scripts/debug_fusion_config.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectra_core.models.optical_sar_fusion import OpticalSarFusionModel  # noqa: E402
from spectra_core.util.config import get_fusion_model_path  # noqa: E402


def main() -> None:
    path = get_fusion_model_path()
    if not path:
        print("Fusion model path: None (built-in fusion will be used).")
        return

    print(f"Fusion model path: {path}")
    if not os.path.exists(path):
        print(f"⚠️ Path not found: {path}")
        return

    try:
        model = OpticalSarFusionModel(path)
        print("✅ Fusion model loaded successfully.")
        if getattr(model, "feature_names_", None):
            print(f"Feature names: {model.feature_names_}")
    except ImportError as exc:  # noqa: BLE001
        print(f"❌ Missing dependency while loading model: {exc}")
        print("   Install requirements (e.g., `pip install -r requirements.txt`) or rely on built-in fusion.")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to load fusion model at {path}: {exc}")


if __name__ == "__main__":
    main()
