from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

# Load .env from the code/ directory explicitly
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "mistralai/mixtral-8x22b-instruct-v0.1")
_client: OpenAI | None = None
_client_api_key: str | None = None


def _resolve_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if key.startswith('"') and key.endswith('"') and len(key) >= 2:
        key = key[1:-1].strip()
    if key.startswith("'") and key.endswith("'") and len(key) >= 2:
        key = key[1:-1].strip()
    return key


def _get_client() -> OpenAI:
    global _client, _client_api_key
    api_key = _resolve_api_key()
    if not api_key:
        raise ValueError("Missing NVIDIA_API_KEY. Set it in code/.env or environment variables.")
    if _client is not None and _client_api_key == api_key:
        return _client
    _client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    _client_api_key = api_key
    return _client


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    _ = temperature
    delay_seconds = 1.0
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            completion = _get_client().chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = completion.choices[0].message.content
            if isinstance(content, list):
                parts = [part.text for part in content if getattr(part, "text", None)]
                return "".join(parts)
            return content or ""
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(delay_seconds)
            delay_seconds *= 2

    detail = repr(last_error) if last_error is not None else "unknown_error"
    raise RuntimeError(f"LLM call failed after retries: {detail}")
