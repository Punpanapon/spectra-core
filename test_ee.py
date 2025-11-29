import toml
import ee

# Load secrets from local .streamlit/secrets.toml
secrets = toml.load(".streamlit/secrets.toml")
ee_cfg = secrets.get("earthengine", {})

service_account = ee_cfg.get("service_account")
key_json = ee_cfg.get("key_json") or ee_cfg.get("private_key")
project_id = ee_cfg.get("project_id")

if not service_account or not key_json:
    raise RuntimeError("Earth Engine secrets missing. Check [earthengine] in .streamlit/secrets.toml.")

print("Using service account:", service_account)

# Create credentials object
credentials = ee.ServiceAccountCredentials(
    email=service_account,
    key_data=key_json,
)

# IMPORTANT: set your GCP project ID here (same project where you created the service account)
PROJECT_ID = project_id or "YOUR-PROJECT-ID"  # <-- change this

# Initialize EE
if PROJECT_ID:
    ee.Initialize(credentials=credentials, project=PROJECT_ID)
else:
    ee.Initialize(credentials=credentials)
print("Earth Engine initialized OK.")

# Tiny computation to prove it works
value = ee.Number(1).add(1).getInfo()
print("Test computation 1 + 1 =", value)

# Optional: test image collection access (Sentinel-2 count for Jan 2024)
s2_count = (
    ee.ImageCollection("COPERNICUS/S2_SR")
    .filterDate("2024-01-01", "2024-02-01")
    .size()
    .getInfo()
)
print("Number of Sentinel-2 images in Jan 2024:", s2_count)
