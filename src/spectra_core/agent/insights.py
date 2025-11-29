import json
import os
import numpy as np
import xarray as xr
from scipy import ndimage
from skimage import measure
from .llm_providers import get_provider_from_env, get_status
from .cache import qa_cache_get, qa_cache_put, make_cache_key
from .usage_limits import can_call, record_call, reset_session, get_usage

def summarize_ndvi(da_red: xr.DataArray, da_nir: xr.DataArray) -> dict:
    """Compute NDVI summary statistics and region analysis."""
    # Compute NDVI safely
    ndvi = (da_nir - da_red) / (da_nir + da_red + 1e-6)
    ndvi_vals = ndvi.values
    
    # Mask valid pixels
    valid_mask = ~np.isnan(ndvi_vals)
    valid_ndvi = ndvi_vals[valid_mask]
    
    if len(valid_ndvi) == 0:
        return {"stats": {}, "area": {}, "regions": {}}
    
    # Basic stats
    stats = {
        "min": float(np.min(valid_ndvi)),
        "max": float(np.max(valid_ndvi)),
        "mean": float(np.mean(valid_ndvi)),
        "std": float(np.std(valid_ndvi)),
        "p5": float(np.percentile(valid_ndvi, 5)),
        "p25": float(np.percentile(valid_ndvi, 25)),
        "p50": float(np.percentile(valid_ndvi, 50)),
        "p75": float(np.percentile(valid_ndvi, 75)),
        "p95": float(np.percentile(valid_ndvi, 95))
    }
    
    # Area fractions
    total_pixels = len(valid_ndvi)
    area = {
        "below_01": float(np.sum(valid_ndvi < 0.1) / total_pixels),
        "below_02": float(np.sum(valid_ndvi < 0.2) / total_pixels),
        "below_03": float(np.sum(valid_ndvi < 0.3) / total_pixels)
    }
    
    # Region analysis for stressed areas (NDVI < 0.2)
    stress_mask = (ndvi_vals < 0.2) & valid_mask
    labeled_regions, num_regions = ndimage.label(stress_mask)
    
    regions = {"count": int(num_regions), "largest_px": 0}
    if num_regions > 0:
        region_sizes = np.bincount(labeled_regions.flat)[1:]  # Skip background
        regions["largest_px"] = int(np.max(region_sizes))
    
    return {"stats": stats, "area": area, "regions": regions}

def insight_bullets(summary: dict) -> list[str]:
    """Generate concise insight bullets from NDVI summary."""
    if not summary.get("stats"):
        return ["No valid NDVI data available"]
    
    stats = summary["stats"]
    area = summary["area"]
    regions = summary["regions"]
    
    bullets = [
        f"Mean NDVI {stats['mean']:.2f} (P25–P75: {stats['p25']:.2f}–{stats['p75']:.2f})",
        f"{area['below_02']*100:.1f}% area < 0.2 (potential stress)",
        f"{area['below_03']*100:.1f}% area < 0.3 (low vegetation)",
        f"NDVI range: {stats['min']:.2f} to {stats['max']:.2f}",
        f"{regions['count']} stressed regions identified",
        f"Largest stressed cluster: ~{regions['largest_px']} pixels"
    ]
    
    if stats['mean'] > 0.6:
        bullets.append("Overall healthy vegetation detected")
    elif stats['mean'] < 0.3:
        bullets.append("Widespread vegetation stress indicated")
    
    return bullets

def narrative(summary: dict, bbox4326: tuple = None) -> str:
    """Generate narrative description of vegetation condition."""
    if not summary.get("stats"):
        return "No valid NDVI data available for analysis."
    
    stats = summary["stats"]
    area = summary["area"]
    
    condition = "healthy" if stats['mean'] > 0.5 else "stressed" if stats['mean'] < 0.3 else "moderate"
    
    text = f"The vegetation analysis reveals {condition} conditions with mean NDVI of {stats['mean']:.2f}. "
    
    if area['below_02'] > 0.1:
        text += f"Approximately {area['below_02']*100:.1f}% of the area shows potential stress (NDVI < 0.2), "
        text += "which may indicate bare soil, water, or severely stressed vegetation. "
    
    if stats['std'] > 0.2:
        text += "High variability suggests mixed land cover types or patchy vegetation health. "
    
    text += "These results should be interpreted considering cloud cover, seasonal effects, and sensor limitations. "
    text += "Ground validation is recommended for critical applications."
    
    return text

def llm_status() -> tuple[str, str]:
    """Check LLM availability and return (status, detail)."""
    return get_status()

def llm_caps_from_env() -> dict:
    """Get LLM capabilities from environment variables."""
    return {
        "max_tokens": int(os.getenv("LLM_ONLINE_MAX_TOKENS", "256")),
        "temperature": float(os.getenv("LLM_ONLINE_TEMPERATURE", "0.2")),
        "timeout": int(os.getenv("LLM_ONLINE_TIMEOUT", "15")),
        "max_calls_session": int(os.getenv("LLM_MAX_CALLS_SESSION", "10")),
        "max_calls_day": int(os.getenv("LLM_MAX_CALLS_DAY", "100"))
    }

def qa_answer(question: str, context: dict, provider=None, caps=None, enabled=False) -> str:
    """Answer questions about NDVI results using rule-based or LLM approach."""
    question_lower = question.lower()
    summary = context.get("summary", {})
    
    if not enabled or not provider:
        # Rule-based fallback
        return _rule_based_answer(question_lower, summary)
    
    # Get provider ID for caching and usage tracking
    provider_type = type(provider).__name__
    provider_id = f"{provider_type}:{getattr(provider, 'model', 'default')}"
    
    # Use default caps if not provided
    if not caps:
        caps = llm_caps_from_env()
    
    # Check cache first
    cache_key = make_cache_key(summary, question, provider_id, caps["max_tokens"])
    cached = qa_cache_get(cache_key)
    if cached:
        return cached
    
    # Check usage limits
    can_make_call, reason = can_call(provider_id, caps["max_calls_session"], caps["max_calls_day"])
    if not can_make_call:
        return f"⚠️ {reason}. Using rule-based answer: {_rule_based_answer(question_lower, summary)}"
    
    # Build prompt text
    system_prompt = (
        "You are an expert in vegetation analysis. Answer concisely based on the NDVI data provided. "
        "Do not hallucinate or make up information not supported by the data."
    )
    stats = summary.get("stats", {})
    area = summary.get("area", {})
    regions = summary.get("regions", {})
    context_text = (
        f"NDVI Statistics: mean={stats.get('mean', 'N/A'):.3f}, "
        f"range={stats.get('min', 'N/A'):.3f} to {stats.get('max', 'N/A'):.3f}. "
        f"Area fractions: {area.get('below_02', 0)*100:.1f}% below 0.2 (stress), "
        f"{area.get('below_03', 0)*100:.1f}% below 0.3 (low vegetation). "
        f"Regions: {regions.get('count', 0)} stressed areas detected."
    )
    prompt_text = (
        f"{system_prompt}\n\n"
        f"NDVI SUMMARY\n{context_text}\n\n"
        f"USER QUESTION\n{question}"
    )
    
    try:
        answer = provider.answer(prompt_text, caps["max_tokens"], caps["temperature"], caps["timeout"])
        # Record successful call and cache result
        record_call()
        qa_cache_put(cache_key, answer)
        return answer
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}. Using rule-based answer: {_rule_based_answer(question_lower, summary)}"

def _rule_based_answer(question_lower: str, summary: dict) -> str:
    """Generate rule-based answers."""
    if "ndvi" in question_lower and "what" in question_lower:
        return "NDVI (Normalized Difference Vegetation Index) measures vegetation health using (NIR-RED)/(NIR+RED). Values range from -1 to 1, with higher values indicating healthier vegetation."
    
    if "area" in question_lower and ("0.2" in question_lower or "stress" in question_lower):
        if summary.get("area"):
            pct = summary["area"]["below_02"] * 100
            return f"Approximately {pct:.1f}% of the area has NDVI below 0.2, indicating potential vegetation stress."
        return "Area statistics not available."
    
    if "mean" in question_lower or "average" in question_lower:
        if summary.get("stats"):
            return f"The mean NDVI is {summary['stats']['mean']:.2f}."
        return "Statistics not available."
    
    return "I can answer questions about NDVI values, area statistics, and vegetation health. For more detailed analysis, consider enabling LLM mode."

def load_fusion_summary(json_path: str) -> dict:
    """Load fusion summary from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return None
