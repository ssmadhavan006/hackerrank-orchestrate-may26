from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()

NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "mistralai/mixtral-8x22b-instruct-v0.1")
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if not NVIDIA_API_KEY:
        raise ValueError("Missing NVIDIA_API_KEY. Set it in code/.env or environment variables.")
    _client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
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

    raise RuntimeError(f"LLM call failed after retries: {last_error}")
