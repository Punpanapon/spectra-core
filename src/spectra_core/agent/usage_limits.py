import os
import json
from datetime import datetime
from typing import Tuple

USAGE_DIR = "outputs/cache"
USAGE_FILE = os.path.join(USAGE_DIR, "usage.json")

def ensure_store():
    """Create directories and files if missing."""
    os.makedirs(USAGE_DIR, exist_ok=True)
    if not os.path.exists(USAGE_FILE):
        data = {
            "last_reset": _today(),
            "daily": {"calls": 0},
            "session": {"calls": 0}
        }
        with open(USAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

def _today() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")

def _load_usage() -> dict:
    """Load usage data, resetting daily if date changed."""
    ensure_store()
    with open(USAGE_FILE, 'r') as f:
        data = json.load(f)
    
    # Reset daily count if date changed
    if data.get("last_reset") != _today():
        data["last_reset"] = _today()
        data["daily"]["calls"] = 0
        with open(USAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    return data

def can_call(provider_id: str, max_calls_session: int, max_calls_day: int) -> Tuple[bool, str]:
    """Check if we can make another call within limits."""
    data = _load_usage()
    
    session_calls = data["session"]["calls"]
    daily_calls = data["daily"]["calls"]
    
    if session_calls >= max_calls_session:
        return False, f"Session limit reached ({session_calls}/{max_calls_session})"
    
    if daily_calls >= max_calls_day:
        return False, f"Daily limit reached ({daily_calls}/{max_calls_day})"
    
    return True, "OK"

def record_call():
    """Increment call counts and save."""
    data = _load_usage()
    data["session"]["calls"] += 1
    data["daily"]["calls"] += 1
    with open(USAGE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def reset_session():
    """Reset session call count."""
    data = _load_usage()
    data["session"]["calls"] = 0
    with open(USAGE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_usage() -> dict:
    """Get current usage stats."""
    return _load_usage()