"""
Thin wrappers around external U-Nets for optical (Sentinel-2) and SAR (Sentinel-1)
deforestation inference.

Behaviour:
- Automatically scans the sibling external repos for pretrained weights.
- Loads models lazily and reuses them across calls.
- If weights/repos/dependencies are missing, logs a warning and returns zeros so
  upstream fusion can fall back to optical-only without crashing.

External repos discovered by name relative to this file:
    UNet-defmapping   -> optical Sentinel-2 UNet (Keras, .hdf5)
    unet-sentinel     -> SAR Sentinel-1 UNet (PyTorch, .pth)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import os
import traceback
from scipy.special import expit

# Force TensorFlow to run on CPU only (avoid GPU/XLA PTX issues)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
try:
    import tensorflow as tf  # noqa: WPS433 E402
except Exception as exc:  # noqa: BLE001
    tf = None  # type: ignore
    _tf_import_error = exc
else:
    _tf_import_error = None
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        # Either no GPU is present or TF runtime is already initialized; ignore.
        pass

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch might be absent in minimal envs
    torch = None  # type: ignore
    nn = None  # type: ignore

_logger = logging.getLogger(__name__)

_OPTICAL_MODEL = None
_OPTICAL_WEIGHTS_USED: Optional[Path] = None
_OPTICAL_WEIGHT_CANDIDATES: list[Path] = []
_OPTICAL_NORM: Optional[Tuple[np.ndarray | None, np.ndarray | None]] = None

_SAR_MODEL = None
_SAR_WEIGHTS_USED: Optional[Path] = None
_SAR_WEIGHT_CANDIDATES: list[Path] = []


def _iter_repo_search_roots(max_depth: int = 6) -> List[Path]:
    """
    Yield parent directories to search for external repos.

    Starts at this file's directory and walks up `max_depth` parents.
    """
    roots: List[Path] = []
    here = Path(__file__).resolve()
    for idx, parent in enumerate(here.parents):
        if idx > max_depth:
            break
        roots.append(parent)
    return roots


def _find_repo_dir(repo_name: str) -> Optional[Path]:
    """Find an external repo directory by name by walking up the tree."""
    for parent in _iter_repo_search_roots():
        candidate = parent / repo_name
        if candidate.exists():
            return candidate
    return None


def _relative(path: Path, root: Optional[Path]) -> str:
    """Return a path relative to root when possible."""
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _choose_weight_file(candidates: list[Path]) -> Optional[Path]:
    """
    Choose the most plausible weight file.

    Heuristic:
      - prefer unet_forest.hdf5 (default deforestation model), then deforestation_unet.hdf5,
        with cloud checkpoints left available but not default
      - prefer filenames containing "best"
      - then ones containing "unet"
      - then most recent mtime
    """
    if not candidates:
        return None

    def _score(p: Path) -> tuple[int, int, int, float]:
        name = p.name.lower()
        forest = 0 if "unet_forest" in name else 1
        defo = 0 if "deforestation_unet" in name else 1
        has_best = 0 if "best" in name else 1
        has_unet = 0 if "unet" in name else 1
        return (forest, defo, has_best, has_unet, -p.stat().st_mtime)

    candidates_sorted = sorted(candidates, key=_score)
    return candidates_sorted[0]


def _discover_optical_assets() -> tuple[Optional[Path], list[Path], Optional[Path], Optional[Path]]:
    """Locate optical weights and normalization npy files."""
    repo = _find_repo_dir("UNet-defmapping")
    if repo is None:
        return None, [], None, None

    weight_candidates: list[Path] = []
    for pattern in ("*.hdf5", "*.h5", "*.keras"):
        weight_candidates.extend(repo.rglob(pattern))
    weight_file = _choose_weight_file(weight_candidates)

    # Prefer non-cloud stats for deforestation
    thirds = [p for p in repo.rglob("bands_third*.npy") if "cloud" not in p.name.lower()]
    nins = [p for p in repo.rglob("bands_nin*.npy") if "cloud" not in p.name.lower()]
    bands_third_path = sorted(thirds)[0] if thirds else None
    bands_nin_path = sorted(nins)[0] if nins else None

    return weight_file, weight_candidates, bands_third_path, bands_nin_path


def _lazy_load_optical_norm() -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Load percentile-based normalization stats for the optical UNet."""
    global _OPTICAL_NORM
    if _OPTICAL_NORM is not None:
        return _OPTICAL_NORM

    _, _, bands_third_path, bands_nin_path = _discover_optical_assets()
    thirds = np.load(bands_third_path).astype(np.float32) if bands_third_path and bands_third_path.exists() else None
    nins = np.load(bands_nin_path).astype(np.float32) if bands_nin_path and bands_nin_path.exists() else None

    if thirds is None or nins is None:
        _logger.warning(
            "Optical UNet percentiles missing; falling back to per-tile robust percentiles. "
            "Place bands_third.npy and bands_nin.npy under UNet-defmapping/Files."
        )
    _OPTICAL_NORM = (thirds, nins)
    return _OPTICAL_NORM


def _lazy_load_optical_model():
    """Load the Keras optical UNet once; return None on failure."""
    global _OPTICAL_MODEL, _OPTICAL_WEIGHTS_USED, _OPTICAL_WEIGHT_CANDIDATES
    if _OPTICAL_MODEL is not None:
        return _OPTICAL_MODEL

    weight_file, candidates, _, _ = _discover_optical_assets()
    _OPTICAL_WEIGHT_CANDIDATES = candidates
    repo_root = _find_repo_dir("UNet-defmapping")
    if weight_file is None:
        _logger.warning(
            "No optical UNet weights (.hdf5/.h5/.keras) found under UNet-defmapping; returning zeros."
        )
        # TODO: Drop a pretrained optical UNet checkpoint into UNet-defmapping (see Files/Link download .hdf5 files.txt).
        return None

    try:
        from tensorflow import keras
    except Exception as exc:  # pragma: no cover - environment dependent
        _logger.error("TensorFlow/Keras not available for optical UNet: %s", exc)
        return None

    try:
        model = keras.models.load_model(weight_file, compile=False)
        model.trainable = False
        _OPTICAL_MODEL = model
        _OPTICAL_WEIGHTS_USED = weight_file
        return _OPTICAL_MODEL
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to load optical UNet weights at %s: %s", weight_file, exc)
        return None


def _normalize_optical_inputs(red: np.ndarray, green: np.ndarray, blue: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Scale and normalize S2 bands to match UNet-defmapping preprocessing."""
    red = np.asarray(red, dtype=np.float32)
    green = np.asarray(green, dtype=np.float32)
    blue = np.asarray(blue, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    h = min(red.shape[0], green.shape[0], blue.shape[0], nir.shape[0])
    w = min(red.shape[1], green.shape[1], blue.shape[1], nir.shape[1])
    stack = np.stack([red[:h, :w], green[:h, :w], blue[:h, :w], nir[:h, :w]], axis=-1)

    stack = np.clip(stack, 0.0, 10000.0) / 10000.0
    thirds, nins = _lazy_load_optical_norm()
    if thirds is not None and nins is not None and len(thirds) == stack.shape[-1]:
        lower = thirds.reshape((1, 1, -1))
        upper = nins.reshape((1, 1, -1))
    else:
        lower = np.percentile(stack, 2, axis=(0, 1), keepdims=True)
        upper = np.percentile(stack, 98, axis=(0, 1), keepdims=True)

    denom = np.maximum(upper - lower, 1e-3)
    normed = (stack - lower) / denom
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def _sigmoid_if_needed(arr: np.ndarray) -> np.ndarray:
    """Ensure outputs are probabilities."""
    if arr.size == 0:
        return arr.astype(np.float32)
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = expit(arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def run_optical_unet_defmapping(red: np.ndarray, green: np.ndarray, blue: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Run the optical UNet (Sentinel-2 B4/B3/B2/B8) from UNet-defmapping.

    Parameters
    ----------
    red, green, blue, nir : np.ndarray
        Co-registered float32 arrays (H x W).

    Returns
    -------
    np.ndarray
        Per-pixel deforestation probabilities in [0, 1].
    """
    if tf is None:
        _logger.warning("TensorFlow is not available; optical UNet disabled. Install tensorflow in the active venv.")
        h = min(getattr(red, "shape", [0])[0] or 0, getattr(nir, "shape", [0])[0] or 0)
        w = min(getattr(red, "shape", [0])[1] or 0, getattr(nir, "shape", [0])[1] or 0)
        return np.zeros((h, w), dtype=np.float32)
    try:
        model = _lazy_load_optical_model()
        normed = _normalize_optical_inputs(red, green, blue, nir)
        orig_h, orig_w, _ = normed.shape
        if model is None:
            return np.zeros((orig_h, orig_w), dtype=np.float32)

        # Determine expected input size from the Keras model
        input_shape = getattr(model, "input_shape", None)
        expected_h = input_shape[1] if input_shape and len(input_shape) > 2 else None
        expected_w = input_shape[2] if input_shape and len(input_shape) > 3 else None
        resize_needed = (
            expected_h is not None
            and expected_w is not None
            and expected_h > 0
            and expected_w > 0
            and (expected_h != orig_h or expected_w != orig_w)
        )

        # Resize to model input if required
        model_in = normed
        if resize_needed:
            model_in = (
                tf.image.resize(
                    normed,
                    size=(int(expected_h), int(expected_w)),
                    method="bilinear",
                    antialias=False,
                )
                .numpy()
                .astype(np.float32)
            )

        batch = np.expand_dims(model_in, axis=0)
        preds = model.predict(batch, verbose=0)
        probs = np.asarray(preds)
        if probs.ndim == 4:
            if probs.shape[-1] == 1:
                probs = probs[..., 0]
            elif probs.shape[-1] >= 2:
                idx = 1 if probs.shape[-1] == 2 else -1
                probs = probs[..., idx]
        probs = np.squeeze(probs)
        probs = _sigmoid_if_needed(probs.astype(np.float32))

        # Resize back to original tile size if we adjusted the input
        if resize_needed and probs.shape != (orig_h, orig_w):
            probs = (
                tf.image.resize(
                    probs[np.newaxis, ..., np.newaxis],
                    size=(orig_h, orig_w),
                    method="bilinear",
                    antialias=False,
                )
                .numpy()
                .squeeze()
                .astype(np.float32)
            )

        if probs.shape != (orig_h, orig_w):
            probs = np.reshape(probs, (orig_h, orig_w))
        return np.clip(probs, 0.0, 1.0).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _logger.warning("Optical UNet inference failed: %s\n%s", exc, tb)
        h = min(getattr(red, "shape", [0])[0] or 0, getattr(nir, "shape", [0])[0] or 0)
        w = min(getattr(red, "shape", [0])[1] or 0, getattr(nir, "shape", [0])[1] or 0)
        return np.zeros((h, w), dtype=np.float32)


def run_optical_unet_defmapping_debug(red: np.ndarray, green: np.ndarray, blue: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Debug-friendly optical UNet runner that surfaces shape info and exceptions.
    """
    if tf is None:
        raise RuntimeError(
            f"TensorFlow is not available in this environment; install tensorflow. Import error: {_tf_import_error}"
        )
    model = _lazy_load_optical_model()
    if model is None:
        raise RuntimeError("Optical UNet model unavailable (no weights or failed load).")

    input_shape = getattr(model, "input_shape", None)
    output_shape = getattr(model, "output_shape", None)
    print(f"[optical_debug] model.input_shape={input_shape}, model.output_shape={output_shape}")

    normed = _normalize_optical_inputs(red, green, blue, nir)
    orig_h, orig_w, _ = normed.shape
    print(f"[optical_debug] stacked input shape: {normed.shape}")

    expected_h = input_shape[1] if input_shape and len(input_shape) > 2 else None
    expected_w = input_shape[2] if input_shape and len(input_shape) > 3 else None
    resize_needed = (
        expected_h is not None
        and expected_w is not None
        and expected_h > 0
        and expected_w > 0
        and (expected_h != orig_h or expected_w != orig_w)
    )

    model_in = normed
    if resize_needed:
        model_in = (
            tf.image.resize(
                normed,
                size=(int(expected_h), int(expected_w)),
                method="bilinear",
                antialias=False,
            )
            .numpy()
            .astype(np.float32)
        )
        print(f"[optical_debug] resized input shape: {model_in.shape}")

    batch = np.expand_dims(model_in, axis=0)
    preds = model.predict(batch, verbose=0)
    print(f"[optical_debug] raw model output shape: {np.asarray(preds).shape}")
    probs = np.asarray(preds)
    if probs.ndim == 4:
        if probs.shape[-1] == 1:
            probs = probs[..., 0]
        elif probs.shape[-1] >= 2:
            # Assume channel 1 is deforestation (binary softmax) or use last channel otherwise.
            idx = 1 if probs.shape[-1] == 2 else -1
            probs = probs[..., idx]
    probs = np.squeeze(probs)
    probs = _sigmoid_if_needed(probs.astype(np.float32))

    if resize_needed and probs.shape != (orig_h, orig_w):
        probs = (
            tf.image.resize(
                probs[np.newaxis, ..., np.newaxis],
                size=(orig_h, orig_w),
                method="bilinear",
                antialias=False,
            )
            .numpy()
            .squeeze()
            .astype(np.float32)
        )
        print(f"[optical_debug] resized output shape: {probs.shape}")

    if probs.shape != (orig_h, orig_w):
        probs = np.reshape(probs, (orig_h, orig_w))

    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    print(f"[optical_debug] final output shape: {probs.shape}, min={probs.min():.4f}, max={probs.max():.4f}")
    return probs


def _discover_sar_weights() -> tuple[Optional[Path], list[Path]]:
    """Locate SAR .pth files under the unet-sentinel repo."""
    repo = _find_repo_dir("unet-sentinel")
    if repo is None:
        return None, []
    candidates = list(repo.rglob("*.pth"))
    return _choose_weight_file(candidates), candidates


def _ensure_repo_on_path(repo: Path):
    """Add repo to sys.path once for imports."""
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _build_sar_unet(in_channels: int) -> Optional[nn.Module]:
    """Instantiate the SAR UNet architecture from the external repo."""
    repo = _find_repo_dir("unet-sentinel")
    if repo is None or torch is None:
        return None
    try:
        _ensure_repo_on_path(repo)
        from model import UNet  # type: ignore
        import config as sar_config  # type: ignore

        enc_channels = getattr(sar_config, "ENC_CHANNELS", (2, 16, 32, 64))
        dec_channels = getattr(sar_config, "DEC_CHANNELS", (64, 32, 16))
        if enc_channels:
            enc_channels = (in_channels,) + tuple(enc_channels[1:])
        out_size = (
            getattr(sar_config, "INPUT_IMAGE_HEIGHT", 512),
            getattr(sar_config, "INPUT_IMAGE_WIDTH", 512),
        )
        model = UNet(
            encChannels=enc_channels,
            decChannels=dec_channels,
            nbClasses=1,
            retainDim=True,
            outSize=out_size,
        )
        return model
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to import SAR UNet definition: %s", exc)
        return None


def _lazy_load_sar_model(in_channels: int) -> Optional[nn.Module]:
    """Load the SAR UNet checkpoint lazily; return None if unavailable."""
    global _SAR_MODEL, _SAR_WEIGHTS_USED, _SAR_WEIGHT_CANDIDATES
    if _SAR_MODEL is not None:
        return _SAR_MODEL
    if torch is None:
        _logger.warning("PyTorch not available; SAR UNet disabled.")
        return None

    weight_file, candidates = _discover_sar_weights()
    _SAR_WEIGHT_CANDIDATES = candidates
    repo_root = _find_repo_dir("unet-sentinel")
    if weight_file is None:
        _logger.warning("No pretrained SAR UNet weights found; returning zeros.")
        # TODO: Drop a trained SAR checkpoint (.pth) into unet-sentinel/checkpoints or similar.
        return None

    try:
        checkpoint = torch.load(weight_file, map_location="cpu")
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            model = _build_sar_unet(in_channels) or nn.Identity()
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)  # type: ignore[arg-type]
            if missing or unexpected:
                _logger.warning(
                    "Loaded SAR checkpoint with missing=%s unexpected=%s", missing, unexpected
                )
        else:
            _logger.warning("Unrecognized SAR checkpoint format at %s", weight_file)
            return None
        model.eval()
        _SAR_MODEL = model
        _SAR_WEIGHTS_USED = weight_file
        return _SAR_MODEL
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to load SAR UNet weights at %s: %s", weight_file, exc)
        return None


def _normalize_sar_bands(sar_bands: np.ndarray) -> np.ndarray:
    """
    Normalize SAR bands roughly following the training script (0-255 inputs).

    - Accepts HxW or HxWxC
    - Clips extreme values and scales to [0, 1]
    """
    sar = np.asarray(sar_bands, dtype=np.float32)
    if sar.ndim == 2:
        sar = sar[..., None]
    sar = np.nan_to_num(sar, nan=0.0)
    # If values look like dB, shift to a 0-1 span
    if sar.min() < -1.0:
        sar = np.clip(sar, -30.0, 5.0)
        sar = (sar - sar.min()) / (sar.max() - sar.min() + 1e-6)
    elif sar.max() > 2.0:
        # Assume 0-255 style scaling
        sar = np.clip(sar, 0.0, np.percentile(sar, 99.5))
        sar = sar / np.maximum(255.0, sar.max())
    return np.clip(sar, 0.0, 1.0).astype(np.float32)


def run_sar_unet_sentinel(sar_bands: np.ndarray, device: str = "cuda") -> np.ndarray:
    """
    Run the SAR UNet (Sentinel-1) from unet-sentinel.

    Parameters
    ----------
    sar_bands : np.ndarray
        HxW or HxWxC float array containing co-registered SAR bands (VV, VH).
    device : str
        Preferred device ("cuda" or "cpu").

    Returns
    -------
    np.ndarray
        Per-pixel deforestation probabilities in [0, 1].
    """
    sar = np.asarray(sar_bands, dtype=np.float32)
    if sar.ndim == 2:
        sar = sar[..., None]
    sar = _normalize_sar_bands(sar)
    h, w, c = sar.shape

    try:
        model = _lazy_load_sar_model(in_channels=c)
        if model is None or torch is None:
            return np.zeros((h, w), dtype=np.float32)

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        actual_device = "cuda" if use_cuda else "cpu"
        model = model.to(actual_device)
        # Ensure the output is resized back to the incoming tile size
        if hasattr(model, "outSize"):
            model.outSize = (h, w)  # type: ignore[attr-defined]

        tensor = torch.from_numpy(sar).permute(2, 0, 1).unsqueeze(0).to(actual_device)
        with torch.no_grad():
            logits = model(tensor)
            if logits.ndim == 3:
                logits = logits.unsqueeze(1)
            probs_t = torch.sigmoid(logits)
            probs = probs_t.squeeze().cpu().numpy().astype(np.float32)
        if probs.shape != (h, w):
            probs = np.reshape(probs, (h, w))
        return _sigmoid_if_needed(probs)
    except Exception as exc:  # noqa: BLE001
        _logger.error("SAR UNet inference failed: %s", exc)
        return np.zeros((h, w), dtype=np.float32)


def get_wrapper_metadata() -> dict:
    """Expose which weight files were selected for debugging/logging."""
    optical_root = _find_repo_dir("UNet-defmapping")
    sar_root = _find_repo_dir("unet-sentinel")
    return {
        "optical_weight": _relative(_OPTICAL_WEIGHTS_USED, optical_root) if _OPTICAL_WEIGHTS_USED else None,
        "optical_candidates": [_relative(p, optical_root) for p in _OPTICAL_WEIGHT_CANDIDATES],
        "sar_weight": _relative(_SAR_WEIGHTS_USED, sar_root) if _SAR_WEIGHTS_USED else None,
        "sar_candidates": [_relative(p, sar_root) for p in _SAR_WEIGHT_CANDIDATES],
    }
