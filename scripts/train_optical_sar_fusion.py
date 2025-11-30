# Discovery: core processing lives in src/spectra_core (pipeline.py) with Streamlit entry at app/streamlit_app.py; current AI overlay uses spectra_ai.infer_unet.run_unet_on_efc_tile.
"""
Train a logistic regression fusion head that blends optical and SAR UNet probabilities.

Usage example:
    python scripts/train_optical_sar_fusion.py \\
        --manifest data/fusion_manifest.csv \\
        --model-out models/optical_sar_fusion.joblib \\
        --max-samples 200000 --class-balance

The manifest CSV should have columns:
    probs_opt, probs_sar, label[, ndvi, vv_vh_ratio, tile_id]
Paths can be .npy or GeoTIFF rasters. Labels should be binary (0/1).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, jaccard_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from spectra_core.models.optical_sar_fusion import _flatten_features


def _load_plane(path: str) -> np.ndarray:
    """Load a probability or label raster from .npy or GeoTIFF."""
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        with rasterio.open(path) as src:
            arr = src.read(1)
    return np.asarray(arr, dtype=np.float32)


def _collect_samples(
    opt_path: str,
    sar_path: str,
    label_path: str,
    extra_paths: Dict[str, str],
    max_samples: int,
    positive_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Build a sampled design matrix + labels for one tile."""
    probs_opt = _load_plane(opt_path)
    probs_sar = _load_plane(sar_path)
    labels = _load_plane(label_path)
    extras = {k: _load_plane(v) for k, v in extra_paths.items() if v}

    design, names = _flatten_features(probs_opt, probs_sar, extras)
    labels_flat = labels.reshape(-1).astype(np.int64)

    valid_mask = np.isfinite(design).all(axis=1) & np.isfinite(labels_flat)
    design = design[valid_mask]
    labels_flat = labels_flat[valid_mask]

    if max_samples and design.shape[0] > max_samples:
        pos_idx = np.where(labels_flat == 1)[0]
        neg_idx = np.where(labels_flat == 0)[0]
        n_pos = int(max_samples * positive_fraction)
        n_neg = max_samples - n_pos
        pos_sel = rng.choice(pos_idx, size=min(len(pos_idx), n_pos), replace=len(pos_idx) < n_pos)
        neg_sel = rng.choice(neg_idx, size=min(len(neg_idx), n_neg), replace=len(neg_idx) < n_neg)
        keep_idx = np.concatenate([pos_sel, neg_sel])
        rng.shuffle(keep_idx)
        design = design[keep_idx]
        labels_flat = labels_flat[keep_idx]

    return design, labels_flat, names


def main():
    parser = argparse.ArgumentParser(description="Train logistic fusion head for optical+SAR probabilities.")
    parser.add_argument("--manifest", required=True, help="CSV with columns: probs_opt, probs_sar, label[, ndvi, vv_vh_ratio]")
    parser.add_argument("--model-out", required=True, help="Output path for joblib model.")
    parser.add_argument("--max-samples", type=int, default=200000, help="Max pixels to sample per tile.")
    parser.add_argument("--positive-fraction", type=float, default=0.5, help="Fraction of positives in the sampled set.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")
    parser.add_argument("--class-balance", action="store_true", help="Use class_weight='balanced'.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    df = pd.read_csv(args.manifest)
    required_cols = {"probs_opt", "probs_sar", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Manifest must include columns: {required_cols}")

    design_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    feature_names: List[str] = []

    for _, row in df.iterrows():
        extra_paths = {}
        for col in ("ndvi", "vv_vh_ratio"):
            if col in df.columns and isinstance(row[col], str) and row[col]:
                extra_paths[col] = row[col]
        design, labels_flat, names = _collect_samples(
            row["probs_opt"],
            row["probs_sar"],
            row["label"],
            extra_paths,
            max_samples=args.max_samples,
            positive_fraction=args.positive_fraction,
            rng=rng,
        )
        design_list.append(design)
        labels_list.append(labels_flat)
        feature_names = names  # consistent across tiles

    X = np.concatenate(design_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.random_seed, stratify=y
    )

    clf = LogisticRegression(
        max_iter=400,
        class_weight="balanced" if args.class_balance else None,
        C=args.C,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_probs >= 0.5).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(y_val, val_pred, average="binary", zero_division=0)
    iou = jaccard_score(y_val, val_pred, zero_division=0)

    print("Validation metrics")
    print(json.dumps({"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou)}, indent=2))
    print("Classification report")
    print(classification_report(y_val, val_pred, digits=3))

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": feature_names}, out_path)
    print(f"Saved fusion model to {out_path}")


if __name__ == "__main__":
    main()
