import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from rasterio.transform import Affine
from sklearn.linear_model import LogisticRegression

# Ensure src/ is on path for test discovery without installation
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectra_core.models import optical_sar_wrappers as wrappers  # noqa: E402
from spectra_core.models.optical_sar_fusion import OpticalSarFusionModel, fallback_fusion  # noqa: E402
from spectra_core.pipelines.optical_sar_fusion_pipeline import (  # noqa: E402
    run_optical_sar_fusion_for_aoi,
)


class DummyOpticalModel:
    def predict(self, batch, verbose: int = 0):
        _, h, w, _ = batch.shape
        return np.full((1, h, w, 1), 0.5, dtype=np.float32)


class DummySarModel(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        return torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)


def test_run_optical_unet_on_tile_shape_and_range(monkeypatch):
    wrappers._optical_model = DummyOpticalModel()
    wrappers._optical_norm = (np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    tile = np.ones((8, 6, 4), dtype=np.float32)
    probs = wrappers.run_optical_unet_on_tile(tile)
    assert probs.shape == (8, 6)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_run_sar_unet_on_tile_shape(monkeypatch):
    wrappers._sar_model = DummySarModel()
    tile = np.ones((5, 4, 1), dtype=np.float32)
    probs = wrappers.run_sar_unet_on_tile(tile, device="cpu")
    assert probs.shape == (5, 4)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_fusion_model_preserves_shape(tmp_path):
    model_path = tmp_path / "fusion.joblib"
    X = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    clf = LogisticRegression(max_iter=100).fit(X, y)
    joblib.dump({"model": clf, "feature_names": ["p_opt", "p_sar", "bias"]}, model_path)

    fusion = OpticalSarFusionModel(str(model_path))
    probs_opt = np.full((4, 4), 0.2, dtype=np.float32)
    probs_sar = np.full((4, 4), 0.8, dtype=np.float32)
    fused = fusion.fuse(probs_opt, probs_sar)
    assert fused.shape == probs_opt.shape
    assert np.all((fused >= 0.0) & (fused <= 1.0))


def test_fallback_fusion_average():
    probs_opt = np.full((2, 2), 0.2, dtype=np.float32)
    probs_sar = np.full((2, 2), 0.8, dtype=np.float32)
    fused = fallback_fusion(probs_opt, probs_sar)
    assert fused.shape == probs_opt.shape
    assert np.allclose(fused, 0.5)


def test_pipeline_runs_with_dummy_models(tmp_path, monkeypatch):
    wrappers._optical_model = DummyOpticalModel()
    wrappers._optical_norm = (np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    wrappers._sar_model = DummySarModel()

    fusion_model_path = tmp_path / "fusion.joblib"
    clf = LogisticRegression(max_iter=50).fit(np.array([[0, 0, 1], [1, 1, 1]]), np.array([0, 1]))
    joblib.dump({"model": clf, "feature_names": ["p_opt", "p_sar", "bias"]}, fusion_model_path)

    def tile_fetcher(aoi, t0, t1):
        s2 = np.ones((4, 4, 4), dtype=np.float32)
        s1 = np.ones((4, 4, 1), dtype=np.float32)
        yield {"tile_id": "dummy", "s2_tile": s2, "s1_tile": s1, "transform": Affine.identity(), "crs": "EPSG:4326"}

    out_dir = run_optical_sar_fusion_for_aoi(
        aoi={"type": "bbox", "coords": [0, 0, 1, 1]},
        t0="2024-01-01",
        t1="2024-02-01",
        fusion_model_path=str(fusion_model_path),
        tile_fetcher=tile_fetcher,
        output_dir=str(tmp_path / "out"),
        device="cpu",
    )
    tif_files = list(Path(out_dir).glob("*.tif"))
    assert tif_files, "Pipeline should write at least one fused GeoTIFF."
