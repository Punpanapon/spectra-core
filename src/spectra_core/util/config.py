from __future__ import annotations

import os

# Optional import of Streamlit and its secret error type.
try:
    import streamlit as st  # type: ignore

    try:
        from streamlit.runtime.secrets import StreamlitSecretNotFoundError  # type: ignore
    except Exception:  # Streamlit < 1.25 fallback

        class StreamlitSecretNotFoundError(Exception):
            pass
except Exception:
    st = None  # type: ignore

    class StreamlitSecretNotFoundError(Exception):  # type: ignore
        pass


def get_env_or_secret(key: str, default: str = "") -> str:
    """
    Return value from environment if present; otherwise (best-effort) from Streamlit secrets.
    Never raises if secrets.toml is missing.
    """
    val = os.getenv(key)
    if val not in (None, ""):
        return val
    if st is None:
        return default
    try:
        return st.secrets.get(key, default)  # type: ignore[attr-defined]
    except StreamlitSecretNotFoundError:
        return default
    except Exception:
        return default


def has_env_or_secret(key: str) -> bool:
    return get_env_or_secret(key, "") not in ("", None)


def get_fusion_model_path() -> str | None:
    """
    Resolve the fusion model path for the optical+SAR overlay.

    Order of precedence:
      1) st.secrets["ai"]["fusion_model_path"] (if Streamlit is available)
      2) os.environ["SPECTRA_FUSION_MODEL_PATH"]
    Strips whitespace and returns None if the final value is empty.
    """
    path: str | None = None
    if st is not None:
        try:
            path = st.secrets.get("ai", {}).get("fusion_model_path")  # type: ignore[attr-defined]
        except StreamlitSecretNotFoundError:
            path = None
        except Exception:
            path = None
    if not path:
        path = os.getenv("SPECTRA_FUSION_MODEL_PATH")
    if path is None:
        return None
    path = str(path).strip()
    return path or None
