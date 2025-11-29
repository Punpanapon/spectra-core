#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8501}"
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}"/tcp || true
else
  pkill -f "streamlit run .*--server.port ${PORT}" || true
  pkill -f "streamlit run" || true
fi
echo "Freed port ${PORT} (if any process was using it)."
