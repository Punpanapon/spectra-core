import numpy as np
import torch
import warnings

from .unet_efc import EfcUNet


def run_unet_on_efc_tile(
    tile_np,
    model_path: str = "models/efc_unet.pt",
    device: str = "cpu",
    **_,
):
    """
    Run the EFC UNet on a single EFC tile.

    Parameters
    ----------
    tile_np : np.ndarray
        H x W x 3 RGB array (uint8 or float32).
    model_path : str
        Path to the .pt weights file.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    preds : np.ndarray
        H x W uint8 mask (0 = background, 1 = vegetation/forest, 2 = deforested).
    """
    device = torch.device(device)

    # --- prepare input -------------------------------------------------
    x = tile_np.astype("float32")
    if x.max() > 1.0:
        x = x / 255.0  # scale to [0,1]

    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(device)  # [1, 3, H, W]

    # --- load model ----------------------------------------------------
    model = EfcUNet(n_channels=3, n_classes=2)
    state = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        model_state = model.state_dict()
        filtered_state = {
            k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape
        }
        missing = [k for k in model_state.keys() if k not in filtered_state]
        warnings.warn(
            f"State dict mismatch when loading {model_path}: {exc}. "
            f"Dropping mismatched keys and continuing. Missing keys: {missing}"
        )
        model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()

    # --- inference -----------------------------------------------------
    with torch.no_grad():
        logits = model(x)             # [1, 2, H, W]
        preds = logits.argmax(dim=1)  # [1, H, W]
        preds = preds[0].cpu().numpy().astype("uint8")

    # For display/debug, expand binary classes to 0/1/2 by mapping class1->2 (deforested).
    preds_display = preds.copy()
    preds_display[preds_display == 1] = 2

    # DEBUG: see how many pixels of each class we get
    vals, cnts = np.unique(preds_display, return_counts=True)
    print("UNet prediction histogram:", dict(zip(vals.tolist(), cnts.tolist())))

    return preds_display
