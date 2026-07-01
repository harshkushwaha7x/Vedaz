"""
Small provider-agnostic client for calling a chat-completion AI model.

Supports two providers out of the box:
  - "together"  (Together AI — OpenAI-compatible endpoint; what Vedaz uses)
  - "anthropic" (Claude — native /v1/messages endpoint)

Provider and model are read from environment variables so nothing is
hard-coded and no key ever lives in the repo:

    AI_PROVIDER       "together" (default) or "anthropic"
    TOGETHER_API_KEY  required if AI_PROVIDER=together
    TOGETHER_MODEL    default: "deepseek-ai/DeepSeek-V3-0324"
    ANTHROPIC_API_KEY required if AI_PROVIDER=anthropic
    ANTHROPIC_MODEL   default: "claude-sonnet-4-6"

Model names on hosted providers change often — if a call fails with a
"model not found"-style error, check the provider's current model list
before assuming the code is broken.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional


class AIClientError(RuntimeError):
    pass


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise AIClientError(f"HTTP {e.code} from {url}: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise AIClientError(f"Network error calling {url}: {e}") from e


def call_model(
    system: str,
    user: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    retries: int = 2,
) -> str:
    """Send one system+user turn to the configured model, return the text reply.

    Raises AIClientError if the call fails after retries or if no API key is
    configured for the selected provider.
    """
    provider = (provider or os.environ.get("AI_PROVIDER", "together")).lower()

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            if provider == "together":
                return _call_together(system, user, model, temperature, max_tokens)
            elif provider == "anthropic":
                return _call_anthropic(system, user, model, temperature, max_tokens)
            elif provider == "mock":
                # Offline mode for CI / demo without an API key
                from mock_api import mock_call_model
                return mock_call_model(system, user)
            else:
                raise AIClientError(
                    f"Unknown AI_PROVIDER '{provider}'. Use 'together', 'anthropic', or 'mock'."
                )
        except AIClientError as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_err  # pragma: no cover


def _call_together(system, user, model, temperature, max_tokens) -> str:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise AIClientError(
            "TOGETHER_API_KEY is not set. Export it as an environment variable "
            "(never hard-code it in the repo)."
        )
    model = model or os.environ.get("TOGETHER_MODEL", "deepseek-ai/DeepSeek-V3-0324")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = _post_json("https://api.together.xyz/v1/chat/completions", headers, payload)
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise AIClientError(f"Unexpected Together AI response shape: {resp}") from e


def _call_anthropic(system, user, model, temperature, max_tokens) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIClientError(
            "ANTHROPIC_API_KEY is not set. Export it as an environment variable "
            "(never hard-code it in the repo)."
        )
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    payload = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    resp = _post_json("https://api.anthropic.com/v1/messages", headers, payload)
    try:
        return "".join(block["text"] for block in resp["content"] if block.get("type") == "text")
    except (KeyError, TypeError) as e:
        raise AIClientError(f"Unexpected Anthropic response shape: {resp}") from e


def extract_json(text: str):
    """Best-effort, defensive parse of a model reply that is supposed to be JSON.

    Models routinely wrap JSON in ```json fences, add a sentence of preamble,
    or add trailing commentary. This strips common wrappers and finds the
    outermost {...} or [...] block before parsing, rather than assuming
    json.loads(text) will just work.
    """
    if text is None:
        return None
    cleaned = text.strip()

    # strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # fall back to grabbing the outermost {...} or [...] span
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


def call_model_json(
    system: str,
    user: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1800,
    fix_attempts: int = 1,
):
    """Call the model expecting a JSON reply; parse defensively; on failure,
    ask the model once to repair its own output before giving up.

    Returns (parsed_object_or_None, raw_text_last_seen).
    """
    raw = call_model(system, user, provider, model, temperature, max_tokens)
    # Mock provider returns structured data directly via mock_call_model_json
    if (provider or __import__("os").environ.get("AI_PROVIDER", "together")).lower() == "mock":
        from mock_api import mock_call_model_json
        return mock_call_model_json(system, user)
    parsed = extract_json(raw)
    attempts_left = fix_attempts
    while parsed is None and attempts_left > 0:
        repair_prompt = (
            "Your previous reply could not be parsed as JSON. Reply again with "
            "ONLY valid JSON — no markdown fences, no commentary before or after. "
            f"Here is what you sent last time, fix it:\n\n{raw}"
        )
        raw = call_model(system, repair_prompt, provider, model, temperature, max_tokens)
        parsed = extract_json(raw)
        attempts_left -= 1
    return parsed, raw
