#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Activate project venv if available (preferred over system/conda)
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
else
  echo "⚠️  .venv not found; using current Python. Run: python -m venv .venv && source .venv/bin/activate"
fi

# LLM env are optional; user can export outside
PORT="${1:-}"
if [ -z "${PORT}" ]; then
  PORT=$(python tools/find_free_port.py)
fi
echo "Starting Streamlit on port ${PORT}"
python -m streamlit run app/streamlit_app.py --server.port "${PORT}"
