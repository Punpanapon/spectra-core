#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Activate env if available
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate spectra || true
fi

# LLM env are optional; user can export outside
PORT="${1:-}"
if [ -z "${PORT}" ]; then
  PORT=$(python tools/find_free_port.py)
fi
echo "Starting Streamlit on port ${PORT}"
streamlit run app/streamlit_app.py --server.port "${PORT}"
