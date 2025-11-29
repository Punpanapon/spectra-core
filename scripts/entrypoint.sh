#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8501}"
echo "[entrypoint] Starting Streamlit on 0.0.0.0:${PORT}"
exec streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port "${PORT}" --server.headless true