from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import requests

from spectra_core.util.config import get_env_or_secret


def _gemini_extract_text(resp: Any) -> str:
    """
    Return text from a google-generativeai response object or dict without assuming a fixed shape.
    """
    txt = getattr(resp, "text", None)
    if txt:
        return str(txt)
    try:
        d = resp.to_dict() if hasattr(resp, "to_dict") else resp
    except Exception:
        d = resp
    if isinstance(d, dict):
        for cand in d.get("candidates", []):
            parts = cand.get("content", {}).get("parts", [])
            if isinstance(parts, list):
                chunks = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
                if chunks:
                    return "".join(chunks)
        if d.get("prompt_feedback") or d.get("promptFeedback"):
            return "Gemini safety filter blocked the answer. Please rephrase the question."
    return ""


class LLMProvider(ABC):
    @abstractmethod
    def answer(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> str:
        pass


class LocalLlamaCpp(LLMProvider):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self.model_path, n_ctx=1024, verbose=False)
        return self._llm
    
    def answer(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> str:
        llm = self._get_llm()
        response = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        return response["choices"][0]["text"].strip()


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    def answer(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAICompatible(LLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
    
    def answer(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or get_env_or_secret("GEMINI_API_KEY", "")
        self.model = model or get_env_or_secret("GEMINI_MODEL", "gemini-2.5-flash")
        self.base_url = (
            base_url
            or get_env_or_secret("GEMINI_BASE_URL", "")
            or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        self.debug = get_env_or_secret("SPECTRA_LLM_DEBUG", "0") in ("1", "true", "True")
        if not self.api_key:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY via env or Streamlit secrets.")

    def answer(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> str:
        """
        Send plain-text prompt; parse defensively.
        """
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError(
                "google-generativeai is not installed. Try: pip install -U google-generativeai"
            ) from e

        genai.configure(api_key=self.api_key)
        model_name = self.model

        def _call(name: str):
            model = genai.GenerativeModel(name)
            return model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "text/plain",
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUAL_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
                ],
                request_options={"timeout": timeout},
            )

        resp = _call(model_name)
        if self.debug:
            try:
                raw = getattr(resp, "text", None) or getattr(resp, "candidates", None) or str(resp)
                print("[GEMINI RAW]", raw)
            except Exception:
                pass

        text = _gemini_extract_text(resp).strip()
        if not text:
            fallback = "gemini-1.5-flash"
            if model_name != fallback:
                try:
                    resp2 = _call(fallback)
                    text = _gemini_extract_text(resp2).strip()
                except Exception:
                    pass
        if not text:
            text = "No text returned by Gemini (after retry)."
        return text


class HuggingFaceProvider(LLMProvider):
    def __init__(self, api_token: str, model: str):
        self.api_token = api_token
        self.model = model
    
    def answer(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").replace(prompt, "").strip()
        return str(result)


def get_provider_from_env(override_mode: str | None = None) -> Optional[LLMProvider]:
    """Get LLM provider based on environment variables."""
    mode = (override_mode or os.getenv("LLM_MODE", "local")).lower()
    
    if mode == "local":
        model_path = os.getenv("LLM_MODEL_PATH")
        if model_path and os.path.exists(model_path):
            try:
                return LocalLlamaCpp(model_path)
            except ImportError:
                return None
        return None
    
    elif mode == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("LLM_MODEL", "llama3.2:3b")
        return OllamaProvider(base_url, model)
    
    elif mode == "openai":
        base_url = os.getenv("LLM_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL")
        if base_url and api_key and model:
            return OpenAICompatible(base_url, api_key, model)
        return None
    
    elif mode == "gemini":
        api_key = get_env_or_secret("GEMINI_API_KEY", "")
        model = get_env_or_secret("GEMINI_MODEL", "gemini-2.5-flash")
        base_url = get_env_or_secret("GEMINI_BASE_URL", "")
        if api_key:
            return GeminiProvider(api_key=api_key, model=model, base_url=base_url or None)
        return None
    
    elif mode == "hf":
        api_token = os.getenv("HF_API_TOKEN")
        model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        if api_token:
            return HuggingFaceProvider(api_token, model)
        return None
    
    return None


def get_status() -> tuple[str, str]:
    """Get LLM status for UI display."""
    use_llm = bool(os.getenv("SPECTRA_USE_LLM", "0") == "1")
    if not use_llm:
        return "off", "Disabled"
    
    mode = os.getenv("LLM_MODE", "local").lower()
    
    if mode == "local":
        model_path = os.getenv("LLM_MODEL_PATH")
        if not model_path:
            return "off", "LLM_MODEL_PATH not set"
        if not os.path.exists(model_path):
            return "off", f"Model file not found"
        try:
            from llama_cpp import Llama
            return "local", "llama-cpp"
        except ImportError:
            return "off", "llama-cpp-python not installed"
    
    elif mode == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
            if response.status_code == 200:
                return "ollama", "Connected"
            return "off", "Ollama not responding"
        except:
            return "off", "Ollama connection failed"
    
    elif mode == "openai":
        base_url = os.getenv("LLM_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL")
        if not all([base_url, api_key, model]):
            missing = [k for k, v in [("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key), ("LLM_MODEL", model)] if not v]
            return "off", f"Missing: {', '.join(missing)}"
        return "online", f"OpenAI-compatible"
    
    elif mode == "gemini":
        api_key = get_env_or_secret("GEMINI_API_KEY", "")
        if not api_key:
            return "off", "Missing: GEMINI_API_KEY"
        return "online", "Gemini"
    
    elif mode == "hf":
        api_token = os.getenv("HF_API_TOKEN")
        if not api_token:
            return "off", "Missing: HF_API_TOKEN"
        return "online", "Hugging Face"
    
    return "off", f"Unknown mode: {mode}"
