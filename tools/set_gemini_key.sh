#!/usr/bin/env bash
set -euo pipefail
KEY="${1:-}"
MODE="${2:-gemini-2.5-flash}"
if [ -z "${KEY}" ]; then
  echo "Usage: tools/set_gemini_key.sh <GEMINI_API_KEY> [model]" >&2
  exit 2
fi
# Export for current shell
export GEMINI_API_KEY="${KEY}"
export SPECTRA_USE_LLM=1
export LLM_MODE=gemini
export GEMINI_MODEL="${MODE}"
echo "Exported key to current shell."

# Persist to ~/.bashrc (idempotent)
for VAR in GEMINI_API_KEY SPECTRA_USE_LLM LLM_MODE GEMINI_MODEL; do
  sed -i "/^export ${VAR}=/d" "$HOME/.bashrc" 2>/dev/null || true
done
{
  echo "export GEMINI_API_KEY='${KEY}'"
  echo "export SPECTRA_USE_LLM=1"
  echo "export LLM_MODE=gemini"
  echo "export GEMINI_MODEL='${MODE}'"
} >> "$HOME/.bashrc"
echo "Appended to ~/.bashrc. Open a new terminal or 'source ~/.bashrc'."
