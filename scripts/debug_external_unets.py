"""
Quick sanity check for the external optical/SAR U-Nets.

Run from the spectra repo root:
    python spectra-core/scripts/debug_external_unets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectra_core.models.external_unet_wrappers import (  # noqa: E402
    get_wrapper_metadata,
    run_optical_unet_defmapping_debug,
    run_sar_unet_sentinel,
)


def _print_metadata(label: str, meta: dict):
    candidates = meta.get(f"{label}_candidates") or []
    weight = meta.get(f"{label}_weight")
    if weight:
        print(f"{label} weight: {weight}")
    else:
        print(f"{label} weight: None found")
    if candidates:
        print(f"{label} candidates: {candidates}")


def main() -> None:
    meta_before = get_wrapper_metadata()
    print("=== Weight discovery (before running) ===")
    _print_metadata("optical", meta_before)
    _print_metadata("sar", meta_before)

    red = np.random.rand(64, 64).astype(np.float32)
    green = np.random.rand(64, 64).astype(np.float32)
    blue = np.random.rand(64, 64).astype(np.float32)
    nir = np.random.rand(64, 64).astype(np.float32)
    sar = np.random.rand(64, 64, 2).astype(np.float32)

    print("\n=== Optical UNet (UNet-defmapping) ===")
    try:
        p_opt = run_optical_unet_defmapping_debug(red, green, blue, nir)
        print(f"Shape: {p_opt.shape}, min={p_opt.min():.4f}, max={p_opt.max():.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"Optical UNet failed: {exc}")
        p_opt = None

    print("\n=== SAR UNet (unet-sentinel) ===")
    try:
        p_sar = run_sar_unet_sentinel(sar, device="cpu")
        print(f"Shape: {p_sar.shape}, min={p_sar.min():.4f}, max={p_sar.max():.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"SAR UNet failed: {exc}")
        p_sar = None

    meta_after = get_wrapper_metadata()
    print("\n=== Weight discovery (after running) ===")
    _print_metadata("optical", meta_after)
    _print_metadata("sar", meta_after)

    def _loaded(arr: np.ndarray | None) -> str:
        return "non-zero output" if arr is not None and np.any(arr > 0.0) else "zeros"

    opt_weight = meta_after.get("optical_weight") or "None"
    sar_weight = meta_after.get("sar_weight") or "None"
    print(f"\nSelected optical weight: {opt_weight}")
    print(f"Selected SAR weight: {sar_weight}")
    if p_opt is not None:
        print(f"Optical output shape: {p_opt.shape}, min={p_opt.min():.4f}, max={p_opt.max():.4f}")
    if p_sar is not None:
        print(f"SAR output shape: {p_sar.shape}, min={p_sar.min():.4f}, max={p_sar.max():.4f}")
    print(f"Optical model output status: {_loaded(p_opt)}")
    print(f"SAR model output status: {_loaded(p_sar)}")


if __name__ == "__main__":
    main()
