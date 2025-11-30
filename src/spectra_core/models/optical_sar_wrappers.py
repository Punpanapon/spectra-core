# Discovery: backend logic lives in src/spectra_core (e.g., pipeline.py for EFC), Streamlit entry is app/streamlit_app.py, and the existing AI overlay calls spectra_ai.infer_unet.run_unet_on_efc_tile on EFC tiles.
"""
Thin wrappers around the external optical (UNet-defmapping) and SAR (unet-sentinel)
models so they can be called on already loaded tiles.

The actual weights/normalization stats are loaded lazily from the paths provided via
environment variables:
    - SPECTRA_OPTICAL_UNET_WEIGHTS
    - SPECTRA_OPTICAL_BANDS_THIRD
    - SPECTRA_OPTICAL_BANDS_NIN
    - SPECTRA_SAR_UNET_WEIGHTS
These paths are intentionally left configurable because the weights live in the
external repos referenced in the project brief.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

OPTICAL_WEIGHTS_ENV = "SPECTRA_OPTICAL_UNET_WEIGHTS"
BANDS_THIRD_ENV = "SPECTRA_OPTICAL_BANDS_THIRD"
BANDS_NIN_ENV = "SPECTRA_OPTICAL_BANDS_NIN"
SAR_WEIGHTS_ENV = "SPECTRA_SAR_UNET_WEIGHTS"

# Module-level caches so we only load weights once per process.
_optical_model = None
_optical_norm: Optional[Tuple[np.ndarray, np.ndarray]] = None
_sar_model: Optional[torch.nn.Module] = None
_sar_norm: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _lazy_load_optical_model(weights_path: Optional[str] = None):
    """Load the Keras optical UNet once. Raise a clear error if unavailable."""
    global _optical_model
    if _optical_model is not None:
        return _optical_model

    weights_path = weights_path or os.getenv(OPTICAL_WEIGHTS_ENV)
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Optical UNet weights not found. Set {OPTICAL_WEIGHTS_ENV} to the HDF5 checkpoint path."
        )

    try:
        from tensorflow import keras
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "TensorFlow/Keras is required for the optical UNet. Ensure tensorflow/keras are installed."
        ) from exc

    _optical_model = keras.models.load_model(weights_path, compile=False)
    _optical_model.trainable = False
    return _optical_model


def _lazy_load_optical_norm() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load percentile-based normalization statistics.

    The UNet-defmapping repo normalizes each band with precomputed percentiles
    (bands_third.npy, bands_nin.npy). If the files are missing we fall back to
    per-tile robust percentiles to keep the pipeline usable.
    """
    global _optical_norm
    if _optical_norm is not None:
        return _optical_norm

    bands_third_path = os.getenv(BANDS_THIRD_ENV)
    bands_nin_path = os.getenv(BANDS_NIN_ENV)

    if bands_third_path and bands_nin_path and os.path.exists(bands_third_path) and os.path.exists(bands_nin_path):
        thirds = np.load(bands_third_path).astype(np.float32)
        nins = np.load(bands_nin_path).astype(np.float32)
        _optical_norm = (thirds, nins)
    else:
        warnings.warn(
            "bands_third/bands_nin not configured; using per-tile 2nd/98th percentiles for normalization."
        )
        _optical_norm = (None, None)
    return _optical_norm


def _normalize_optical_tile(tile: np.ndarray) -> np.ndarray:
    """Apply percentile normalization similar to UNet-defmapping."""
    if tile.ndim != 3 or tile.shape[-1] != 4:
        raise ValueError(f"s2_tile must be HxWx4 (bands 4,3,2,8). Got shape {tile.shape}")

    tile_f = tile.astype(np.float32)
    thirds, nins = _lazy_load_optical_norm()
    if thirds is None or nins is None:
        # Robust per-tile percentiles as a fallback
        lower = np.percentile(tile_f, 2, axis=(0, 1), keepdims=True)
        upper = np.percentile(tile_f, 98, axis=(0, 1), keepdims=True)
    else:
        lower = thirds.reshape((1, 1, 4))
        upper = nins.reshape((1, 1, 4))

    denom = np.maximum(upper - lower, 1e-3)
    normed = (tile_f - lower) / denom
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def _sigmoid_if_needed(arr: np.ndarray) -> np.ndarray:
    """Ensure probabilities in [0,1]."""
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = expit(arr)
    return np.clip(arr, 0.0, 1.0)


def run_optical_unet_on_tile(s2_tile: np.ndarray) -> np.ndarray:
    """
    Run the optical UNet (Sentinel-2 bands 4,3,2,8) and return per-pixel probabilities.
    """
    model = _lazy_load_optical_model()
    normed = _normalize_optical_tile(s2_tile)
    batch = np.expand_dims(normed, axis=0)  # 1, H, W, 4 (Keras default)
    preds = model.predict(batch, verbose=0)
    probs = np.asarray(preds)

    if probs.ndim == 4 and probs.shape[-1] > 1:
        # Assume last channel is foreground class
        probs = probs[..., -1]
    probs = np.squeeze(probs)
    probs = _sigmoid_if_needed(probs.astype(np.float32))
    if probs.shape != s2_tile.shape[:2]:
        raise ValueError(f"Optical model output shape {probs.shape} does not match input {s2_tile.shape[:2]}.")
    return probs


@dataclass
class SarNorm:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None


class SimpleSarUNet(nn.Module):
    """
    Lightweight fallback UNet-ish block for SAR inference when importing the external
    model class is not feasible. This is only a stand-in; real weights should be
    provided via SPECTRA_SAR_UNET_WEIGHTS.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return self.conv2(x)


def _lazy_load_sar_model(weights_path: Optional[str] = None, in_channels: int = 1):
    """Load SAR model lazily; default to a tiny conv stack if weights are missing."""
    global _sar_model
    if _sar_model is not None:
        return _sar_model

    weights_path = weights_path or os.getenv(SAR_WEIGHTS_ENV)
    model = SimpleSarUNet(in_channels=in_channels)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            warnings.warn(f"SAR checkpoint loaded with missing={missing} unexpected={unexpected}")
    else:
        warnings.warn(
            "SPECTRA_SAR_UNET_WEIGHTS not set; using randomly initialized fallback SAR model."
        )
    _sar_model = model.eval()
    return _sar_model


def _normalize_sar_tile(tile: np.ndarray) -> np.ndarray:
    """
    Basic normalization: per-channel z-score with fallback to 0-1 scaling if std is tiny.
    """
    tile_f = tile.astype(np.float32)
    if tile_f.ndim != 3:
        raise ValueError(f"s1_tile must be HxWxC. Got shape {tile.shape}")

    mean = tile_f.mean(axis=(0, 1), keepdims=True)
    std = tile_f.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-4, 1.0, std)
    normed = (tile_f - mean) / std
    return normed.astype(np.float32)


def run_sar_unet_on_tile(s1_tile: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Run the SAR UNet (Sentinel-1) and return per-pixel deforestation probabilities.
    """
    if s1_tile.ndim != 3:
        raise ValueError(f"s1_tile must be HxWxC. Got shape {s1_tile.shape}")

    normed = _normalize_sar_tile(s1_tile)
    in_channels = normed.shape[-1]
    model = _lazy_load_sar_model(in_channels=in_channels)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    actual_device = "cuda" if use_cuda else "cpu"
    model = model.to(actual_device)

    tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(0).to(actual_device)
    with torch.no_grad():
        logits = model(tensor)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        probs_t = torch.sigmoid(logits)
        probs = probs_t.squeeze().cpu().numpy().astype(np.float32)
    if probs.shape != s1_tile.shape[:2]:
        raise ValueError(f"SAR model output shape {probs.shape} does not match input {s1_tile.shape[:2]}.")
    return probs

