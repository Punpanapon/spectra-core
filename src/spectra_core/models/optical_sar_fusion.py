# Discovery: backend lives in src/spectra_core (pipeline.py etc.), Streamlit entry is app/streamlit_app.py, and the current AI overlay uses spectra_ai.infer_unet.run_unet_on_efc_tile on EFC tiles.
"""
Late-fusion head that combines optical and SAR UNet probabilities using a
logistic regression model trained offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

try:
    from sklearn.base import ClassifierMixin
except ImportError:
    ClassifierMixin = object  # type: ignore
    _SKLEARN_AVAILABLE = False
else:
    _SKLEARN_AVAILABLE = True


def fallback_fusion(probs_opt: np.ndarray, probs_sar: np.ndarray) -> np.ndarray:
    """
    Simple deterministic fusion for when no trained model is available.
    Returns 0.5 * probs_opt + 0.5 * probs_sar with shape/dtype preserved.
    """
    if probs_opt.shape != probs_sar.shape:
        raise ValueError("probs_opt and probs_sar must have identical shapes for fallback fusion.")
    fused = 0.5 * probs_opt.astype(np.float32) + 0.5 * probs_sar.astype(np.float32)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def _flatten_features(
    probs_opt: np.ndarray, probs_sar: np.ndarray, extra_features: Optional[Dict[str, np.ndarray]]
) -> tuple[np.ndarray, List[str]]:
    """Flatten the per-pixel feature planes into a design matrix."""
    h, w = probs_opt.shape
    if probs_sar.shape != (h, w):
        raise ValueError("probs_opt and probs_sar must have identical shapes.")

    planes: List[np.ndarray] = [
        probs_opt.reshape(-1),
        probs_sar.reshape(-1),
        np.ones((h * w,), dtype=np.float32),  # bias term
    ]
    names = ["p_opt", "p_sar", "bias"]

    if extra_features:
        for key, arr in extra_features.items():
            if arr.shape[:2] != (h, w):
                raise ValueError(f"Extra feature '{key}' shape {arr.shape} does not match {h}x{w}.")
            planes.append(arr.reshape(-1))
            names.append(key)

    design = np.stack(planes, axis=1).astype(np.float32)
    return design, names


@dataclass
class OpticalSarFusionModel:
    """
    Lightweight wrapper around a scikit-learn LogisticRegression (or compatible)
    classifier trained to fuse optical and SAR probabilities.
    """

    model_path: str
    model: ClassifierMixin | None = None
    feature_names_: List[str] | None = None

    def __post_init__(self):
        self._load_model()

    def _load_model(self):
        if joblib is None:  # pragma: no cover - environment dependent
            raise ImportError(
                "joblib is required to load the fusion model. Install with `pip install joblib` "
                "or rely on the built-in fallback fusion."
            )
        if not _SKLEARN_AVAILABLE:  # pragma: no cover - environment dependent
            raise ImportError(
                "scikit-learn is required to load the fusion model. Install with `pip install scikit-learn` "
                "or rely on the built-in fallback fusion."
            )
        obj = joblib.load(self.model_path)
        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.feature_names_ = obj.get("feature_names")
        else:
            self.model = obj
            self.feature_names_ = getattr(obj, "feature_names_in_", None)
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Fusion model must implement predict_proba.")

    def fuse(
        self,
        probs_opt: np.ndarray,
        probs_sar: np.ndarray,
        extra_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Fuse optical and SAR probabilities into a single deforestation probability map.
        """
        design, names = _flatten_features(probs_opt, probs_sar, extra_features)
        probs = self.model.predict_proba(design)[:, 1]
        fused = probs.reshape(probs_opt.shape).astype(np.float32)
        self.feature_names_ = self.feature_names_ or names
        return fused
