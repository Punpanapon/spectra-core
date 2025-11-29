#!/usr/bin/env python3
"""
Small launcher to run the EFC U-Net training with default paths/params.

Prefers spectra_ai.train_unet if present; falls back to the template.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_train_fn() -> Tuple[Callable, str]:
    """Load train_unet from the primary module, fallback to template."""
    try:
        from spectra_ai.train_unet import train_unet  # type: ignore

        return train_unet, "spectra_ai.train_unet"
    except Exception as primary_exc:  # noqa: BLE001
        try:
            from spectra_ai.train_unet_template import train_unet  # type: ignore

            print(
                "[WARN] spectra_ai.train_unet not found/usable; "
                "falling back to spectra_ai.train_unet_template. "
                f"Primary import error: {primary_exc}"
            )
            return train_unet, "spectra_ai.train_unet_template"
        except Exception as template_exc:  # noqa: BLE001
            raise ImportError(
                f"Unable to import train_unet from spectra_ai. Errors: primary={primary_exc}, template={template_exc}"
            ) from template_exc


def main() -> None:
    train_unet_fn, source = load_train_fn()

    data_root = "data/efc_tiles"
    models_dir = "models"
    num_epochs = 10
    batch_size = 4

    try:
        train_unet_fn(
            data_root=data_root,
            models_dir=models_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )
    except TypeError as exc:
        # For templates that take no args, fall back to bare call.
        print(
            f"[INFO] train_unet from {source} did not accept the default kwargs "
            f"(data_root/models_dir/num_epochs/batch_size). Calling without kwargs. ({exc})"
        )
        train_unet_fn()


if __name__ == "__main__":
    main()
