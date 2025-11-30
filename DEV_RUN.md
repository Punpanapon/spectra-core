## Running SPECTRA locally (venv, no conda)

1) From repo root:
```bash
cd ~/wsl-setup-spectra/spectra/spectra
python -m venv .venv  # if not created yet
source .venv/bin/activate
python -m pip install -r spectra-core/requirements.txt
```

2) Debug external UNets (CPU-only):
```bash
python spectra-core/scripts/debug_external_unets.py
```

3) Run Streamlit app:
```bash
python -m streamlit run spectra-core/app/streamlit_app.py
```

Notes:
- Avoid `conda activate base`; use the project `.venv` above.
- Optical UNet falls back to zeros if TensorFlow is missing; install `tensorflow` in the venv.
