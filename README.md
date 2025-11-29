# SPECTRA Core

A zero-API geospatial fusion pipeline for creating Enhanced Forest Composite (EFC) visualizations from Sentinel-2 optical and SAR data.

## Enhanced Forest Composite (EFC) Formula

The EFC visualization uses the following channel mapping:
- **R**: 1 - NDVI (inverted vegetation index)
- **G**: NDVI (vegetation index)
- **B**: Normalized SAR dB (C-band and/or L-band)

Where NDVI = (NIR - RED) / (NIR + RED)

## Quickstart

```bash
conda env create -f env/environment.yml
conda activate spectra
```

## Usage

```bash
python cli/run_fusion.py --red data/S2_B04.tif --nir data/S2_B08.tif --sar_c data/S1_C.tif --out outputs/
python cli/report.py --in outputs/ --open
```

## Fetch Sample S2/S1 via STAC

Download aligned Sentinel-2 and Sentinel-1 data for any area:

```bash
# Fetch data for Khao Yai area (Thailand)
python -m spectra_core.ai.stac_fetch --bbox "101.2,14.3,101.7,14.8" --t0 2024-01-01 --t1 2024-01-31 --out data --cloud 25

# Outputs: data/S2_B04.tif, data/S2_B08.tif, data/S1_VV.tif (if available)
# All files aligned to same 10m grid, COG format with DEFLATE compression
```

**Options:**
- `--bbox`: Bounding box in WGS84 (minx,miny,maxx,maxy)
- `--cloud`: Maximum cloud cover percentage (default: 30)
- `--max-items`: Items to search per collection (default: 3)

**Technical Notes:**
- Stackstac requires bounds in the target EPSG; we convert bbox to EPSG automatically
- S1 alignment only occurs after S2 processing succeeds

⚠️ **Note**: If no Sentinel-2 data is found, try widening the date range or increasing `--cloud` (e.g., 70).

## AI Data → Chips

Generate training chips for AI models from multiple data sources:

### STAC Data (Thailand Example)
```bash
# Fetch Sentinel-2/1 data for Bangkok area
python -m spectra_core.ai.stac_fetch --bbox "100.1,14.9,100.4,15.2" --t0 2024-01-01 --t1 2024-01-15 --out data

# Generate chips (requires label.tif)
python -m spectra_core.ai.chipper --red data/S2_B04.tif --nir data/S2_B08.tif --sar data/S1_VV.tif --label data/label.tif --out outputs/chips/thailand
```

### Local Files
```bash
# Generate chips from local GeoTIFFs
python -m spectra_core.ai.chipper --red data/S2_B04.tif --nir data/S2_B08.tif --label data/label.tif --out outputs/chips/local
```

### S3 Sources
```bash
# Generate chips from S3-hosted rasters
python -m spectra_core.ai.chipper --red s3://bucket/S2_B04.tif --nir s3://bucket/S2_B08.tif --label s3://bucket/label.tif --out outputs/chips/s3
```

### Dummy Labels (for smoke tests only)
```bash
# Create dummy labels from NDVI thresholding
python -m spectra_core.ai.make_dummy_label --red data/S2_B04.tif --nir data/S2_B08.tif --out data/label.tif --method otsu

# Then generate chips
python -m spectra_core.ai.chipper --red data/S2_B04.tif --nir data/S2_B08.tif --label data/label.tif --out outputs/chips/demo
```

⚠️ **Warning**: Dummy labels are for pipeline testing only. Use real ground truth labels for actual model training.

### Output Structure
```
outputs/chips/
├── images/          # .npy feature arrays [H,W,C]
├── masks/           # .npy label arrays [H,W]
├── thumbnails/      # .png preview images
└── manifest.json    # Metadata (chip count, channels, CRS)
```

## Deploy to AWS App Runner

### Prerequisites
- Docker Desktop with WSL2 integration
- AWS CLI configured with appropriate credentials
- IAM permissions for ECR and App Runner

### Deployment Steps

```bash
# 1. Build Docker image locally
docker build -t spectra-core:local .

# 2. Test locally
docker run --rm -p 8501:8501 spectra-core:local
# Visit http://localhost:8501

# 3. Configure AWS credentials
aws configure  # Set your access key, secret, and region

# 4. Push to ECR
bash scripts/ecr_push.sh

# 5. Create App Runner service
# - Go to AWS App Runner Console
# - Create service → Container image → Amazon ECR
# - Select the pushed image
# - Set port to 8501
# - Deploy
```

### Configuration
- **Port**: 8501 (configured in Streamlit)
- **Upload Limits**: 2GB per file
- **Memory**: Recommend 4GB+ for large raster processing
- **CPU**: 2+ vCPUs recommended

## Agentic Insights (Rule-based + Optional Local LLM)

The Streamlit app includes an "Agentic Insights" tab that provides AI-powered analysis of vegetation health and fusion results.
Gemini credentials can be provided via env or Streamlit secrets; the app prefers env and never crashes if secrets.toml is missing.


### Features
- **Statistical Analysis**: NDVI distribution, area fractions, stressed region detection
- **Natural Language Insights**: Human-readable bullets and narrative summaries
- **Interactive Q&A**: Ask questions about your results
- **Zero Cost**: Default rule-based mode requires no external APIs

### LLM Modes

Choose from multiple LLM backends for advanced AI responses:

#### 1. Local LLM (llama-cpp)
```bash
export SPECTRA_USE_LLM=1
export LLM_MODE=local
export LLM_MODEL_PATH=./models/tinyllama.gguf
```

#### 2. Ollama (Local HTTP)
```bash
# Start Ollama server
ollama serve
ollama pull llama3.2:3b

# Configure SPECTRA
export SPECTRA_USE_LLM=1
export LLM_MODE=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export LLM_MODEL=llama3.2:3b
```

#### 3. Online APIs (OpenAI-compatible)
```bash
export SPECTRA_USE_LLM=1
export LLM_MODE=openai
export LLM_BASE_URL=https://api.openrouter.ai/v1
export LLM_API_KEY=sk-your-key-here
export LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

#### 4. Google Gemini
```bash
export SPECTRA_USE_LLM=1
export LLM_MODE=gemini
export GEMINI_API_KEY=AIzaSyCWoO3Pb-Z-oRlQ9_5G5PYBhwNtF6M1S6k
export GEMINI_MODEL=gemini-2.5-flash
```

#### 5. Hugging Face Inference API
```bash
export SPECTRA_USE_LLM=1
export LLM_MODE=hf
export HF_API_TOKEN=your-hf-token
export HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

**Safety Caps (all online modes):**
```bash
export LLM_ONLINE_MAX_TOKENS=256
export LLM_ONLINE_TEMPERATURE=0.2
export LLM_ONLINE_TIMEOUT=15
```

**Supported Online Providers:**
- **OpenRouter** (https://openrouter.ai) - Many models, free tier available
- **Together AI** (https://together.ai) - Fast inference, free credits
- **Fireworks AI** (https://fireworks.ai) - High performance, free tier
- **OpenAI** (https://openai.com) - GPT models
- **Google Gemini** (https://ai.google.dev) - Fast and capable, free tier
- **Hugging Face** (https://huggingface.co) - Open models, free inference API
- Any OpenAI-compatible API

## LLM On/Off & Safe Limits

The Agentic Insights tab includes comprehensive controls for LLM usage:

### Interactive Controls
- **LLM Toggle**: Enable/disable AI responses per session
- **Provider Selection**: Choose from 6 options via dropdown
- **Safety Sliders**: Adjust token limits and call budgets
- **Panic Button**: Instantly disable LLM and reset session budget

### Usage Limits
```bash
# Environment defaults (optional)
export LLM_ONLINE_MAX_TOKENS=256
export LLM_MAX_CALLS_SESSION=10
export LLM_MAX_CALLS_DAY=100
export LLM_ONLINE_TEMPERATURE=0.2
export LLM_ONLINE_TIMEOUT=15
```

### Budget Tracking
- **Session Budget**: Resets when you restart the app
- **Daily Budget**: Automatically resets at midnight
- **Usage Display**: Shows current usage vs limits in real-time
- **Persistent Storage**: Tracks usage in `outputs/cache/usage.json`

**Safety Features:**
- Response caching to avoid repeat charges
- Per-session and daily call limits
- Token limits and timeouts
- Graceful fallback to rule-based answers
- Local-first: no data sent unless explicitly configured

### Gemini API quick setup
- Set: `bash tools/set_gemini_key.sh <KEY> [model]`
- Verify: `python -m spectra_core.cli.gemini_check`
- Launch: `streamlit run app/streamlit_app.py`

**Notes:**
- Default mode is $0 rule-based analysis
- UI controls override environment variables
- Always check your provider's current quotas and pricing
- Free tiers vary by provider and may have rate limits

### CLI Usage

Generate insights offline:

```bash
python -m spectra_core.agent.cli --red data/S2_B04.tif --nir data/S2_B08.tif --out outputs/insights
```
