# Fusion Model Configuration (Optical + SAR)

The fusion overlay now always renders, even without an external fusion model:
- If **no model path** is configured, the app uses a built-in simple fusion: `0.5 * p_opt + 0.5 * p_sar`.
- If a **model path** is configured and loads successfully, the trained fusion head is used.
- If a **model path** is set but missing or fails to load, the app logs the error and falls back to the built-in fusion with a warning banner.

## Configure a real fusion model

Option A: Streamlit secrets (`.streamlit/secrets.toml`)
```toml
[ai]
fusion_model_path = "/absolute/path/to/optical_sar_fusion.joblib"
```

Option B: Environment variable
```bash
export SPECTRA_FUSION_MODEL_PATH="/absolute/path/to/optical_sar_fusion.joblib"
```

Precedence: secrets (`ai.fusion_model_path`) first, then `SPECTRA_FUSION_MODEL_PATH`. Blank strings are treated as not set.

## Behavior summary
- No config: banner shows “Using built-in fusion (no external model configured)”, overlay uses the simple average.
- Configured & loads: banner shows “Fusion model loaded from: <path>”.
- Configured but fails/missing: banner shows “Fusion model failed to load; using built-in fusion (see logs)” and the overlay uses the simple average.

## Debug
Run:
```bash
python scripts/debug_fusion_config.py
```
This prints the effective fusion model path and attempts to load it (reporting success/failure).
