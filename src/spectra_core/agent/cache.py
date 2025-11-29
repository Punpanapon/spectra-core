import os
import json
import hashlib
from typing import Optional

CACHE_DIR = "outputs/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "insights_cache.json")

def _load_cache() -> dict:
    """Load cache from disk."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def _save_cache(cache: dict) -> None:
    """Save cache to disk."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except:
        pass

def _make_key(summary_json: str, question: str, provider_id: str) -> str:
    """Generate cache key from inputs."""
    content = f"{summary_json}|{question}|{provider_id}"
    return hashlib.sha1(content.encode()).hexdigest()

def qa_cache_get(key: str) -> Optional[str]:
    """Get cached answer by key."""
    cache = _load_cache()
    return cache.get(key)

def qa_cache_put(key: str, value: str) -> None:
    """Cache an answer by key."""
    cache = _load_cache()
    cache[key] = value
    _save_cache(cache)

def make_cache_key(summary: dict, question: str, provider_id: str, max_tokens: int) -> str:
    """Generate cache key from inputs."""
    content = f"{json.dumps(summary, sort_keys=True)}|{question.lower().strip()}|{provider_id}|{max_tokens}"
    return hashlib.sha1(content.encode()).hexdigest()