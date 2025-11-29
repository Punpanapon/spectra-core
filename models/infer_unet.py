import os
import warnings
import numpy as np
import torch

from .unet_efc import EfcUNet


def _load_state_with_fallback(model: EfcUNet, model_path: str, device: torch.device):
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        missing = [k for k in model_state if k not in filtered]
        warnings.warn(
            f"State dict mismatch when loading {model_path}: {exc}. "
            f"Dropping mismatched keys. Missing keys now: {missing}"
        )
        model.load_state_dict(filtered, strict=False)


def load_unet_model(model_path: str, device: str = "cpu") -> EfcUNet:
    """
    Load an EfcUNet model from a .pt checkpoint. If missing or torch unavailable, raise RuntimeError.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"UNet model file not found at: {model_path}")
    device_t = torch.device(device)
    try:
        model = EfcUNet(n_channels=3, n_classes=2)
        _load_state_with_fallback(model, model_path, device_t)
        model.to(device_t)
        model.eval()
        return model
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model {model_path}: {exc}") from exc


def run_unet_on_efc_tile(
    tile_np,
    model_path: str = "models/efc_unet.pt",
    device: str = "cpu",
    **_,
) -> np.ndarray:
    """
    efc_rgb: HxWx3 uint8 or float, the EFC tile.
    Returns: HxW integer mask with values 0 or 1 (background / deforested).
    """
    if tile_np.ndim != 3 or tile_np.shape[-1] != 3:
        raise ValueError("tile_np must be HxWx3.")

    device_t = torch.device(device)
    arr = tile_np.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # to C,H,W
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device_t)

    model = EfcUNet(n_channels=3, n_classes=2)
    _load_state_with_fallback(model, model_path, device_t)
    model.to(device_t)
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return preds
