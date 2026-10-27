
from __future__ import annotations

import json
import os
import re
import threading
import time

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI

            _client = OpenAI(
                base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=os.getenv("OPENROUTER_API_KEY") or "missing-key",
                timeout=float(os.getenv("LLM_TIMEOUT", "600")),
                max_retries=0,  # retries are handled (and counted) below
            )
        return _client


# ------------------------------------------------------------------
# Shared generation settings
# ------------------------------------------------------------------

def _env_float(name):
    value = os.getenv(name, "").strip()
    return float(value) if value else None


def _env_int(name):
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def llm_settings() -> dict:
    """The generation settings every method is run with (stored in metadata)."""
    temperature = os.getenv("LLM_TEMPERATURE", "0").strip()
    seed = os.getenv("LLM_SEED", "42").strip()
    provider = [p.strip() for p in os.getenv("LLM_PROVIDER", "").split(",") if p.strip()]
    settings = {
        "model": os.getenv("MODEL_NAME"),
        "temperature": float(temperature) if temperature else None,
        "top_p": _env_float("LLM_TOP_P"),
        "seed": int(seed) if seed else None,
        "reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "").strip() or None,
        "max_output_tokens": _env_int("LLM_MAX_OUTPUT_TOKENS"),
        "provider": provider or None,
        "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
    }
    # only recorded when used, so settings of earlier runs stay comparable
    if os.getenv("LLM_REASONING", "").strip().lower() == "off":
        settings["reasoning"] = "off"
    return settings


# ------------------------------------------------------------------
# Usage accounting (thread-safe; shared by all worker threads)
# ------------------------------------------------------------------

_usage_lock = threading.Lock()


def _empty_usage():
    return {
        "api_requests": 0,          # every request sent, including failed ones
        "successful_calls": 0,      # requests that produced usable text
        "failed_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,     # includes reasoning tokens
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "latency_seconds_sum": 0.0,  # sum over requests = "serial" compute time
    }


_usage = _empty_usage()


def reset_usage():
    global _usage
    with _usage_lock:
        _usage = _empty_usage()


def get_usage() -> dict:
    with _usage_lock:
        u = dict(_usage)
    u["latency_seconds_sum"] = round(u["latency_seconds_sum"], 2)
    return u


def _record(ok, latency, usage_obj, purpose, model, error=None):
    prompt = completion = reasoning = 0
    if usage_obj is not None:
        prompt = getattr(usage_obj, "prompt_tokens", 0) or 0
        completion = getattr(usage_obj, "completion_tokens", 0) or 0
        details = getattr(usage_obj, "completion_tokens_details", None)
        if details is not None:
            reasoning = getattr(details, "reasoning_tokens", 0) or 0
    with _usage_lock:
        _usage["api_requests"] += 1
        _usage["successful_calls" if ok else "failed_requests"] += 1
        _usage["prompt_tokens"] += prompt
        _usage["completion_tokens"] += completion
        _usage["reasoning_tokens"] += reasoning
        _usage["total_tokens"] += prompt + completion
        _usage["latency_seconds_sum"] += latency

    log_path = os.getenv("LLM_USAGE_LOG")
    if log_path:
        line = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": purpose,
            "model": model,
            "ok": ok,
            "latency_s": round(latency, 3),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "error": error,
        }
        with _usage_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")


# ------------------------------------------------------------------
# Main call
# ------------------------------------------------------------------

def generate_summary(
    prompt: str,
    model_name: str = None,
    max_tokens: int = None,
    reasoning_effort: str = None,
    temperature: float = None,
    purpose: str = "generation",
) -> str:
    """
    Send one prompt and return the text answer.

    Retries (with the same policy for every method) on API errors, empty
    answers and finish_reason=length.  Raises RuntimeError when all attempts
    fail, so a truncated or missing output can never be used silently.
    """
    s = llm_settings()
    model = model_name or s["model"]
    if not model:
        raise RuntimeError("MODEL_NAME is not set.")

    limit = max_tokens if max_tokens is not None else s["max_output_tokens"]
    effort = reasoning_effort if reasoning_effort is not None else s["reasoning_effort"]
    temp = temperature if temperature is not None else s["temperature"]

    last_error = None
    for attempt in range(1, s["max_retries"] + 1):
        request_args = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temp is not None:
            request_args["temperature"] = temp
        if s["top_p"] is not None:
            request_args["top_p"] = s["top_p"]
        if s["seed"] is not None:
            request_args["seed"] = s["seed"]
        if limit is not None:
            request_args["max_tokens"] = limit
        if effort is not None:
            request_args["reasoning_effort"] = effort
        # the provider pin belongs to the summarization model only (the judge
        # model may not be served by that provider); JUDGE_PROVIDER pins the judge
        provider = s["provider"] if model_name is None or model_name == s["model"] else \
            [p.strip() for p in os.getenv("JUDGE_PROVIDER", "").split(",") if p.strip()]
        extra = {}
        if provider:
            extra["provider"] = {"order": provider, "allow_fallbacks": False}
        if s.get("reasoning") == "off" and (model_name is None or model_name == s["model"]):
            extra["reasoning"] = {"enabled": False}
        if extra:
            request_args["extra_body"] = extra

        t0 = time.time()
        response = None
        try:
            response = _get_client().chat.completions.create(**request_args)
            latency = time.time() - t0
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason

            print(f"[LLM] model={model} purpose={purpose} finish_reason={finish_reason} "
                  f"attempt={attempt}")

            if finish_reason == "length":
                raise _LengthError(
                    "LLM output reached the token limit. "
                    "The generated text may be incomplete."
                )
            if content is None or not content.strip():
                raise RuntimeError("The model returned no text content.")

            _record(True, latency, getattr(response, "usage", None), purpose, model)
            return content

        except _LengthError as e:
            _record(False, time.time() - t0, getattr(response, "usage", None), purpose,
                    model, str(e))
            last_error = e
            if limit is not None:
                limit = int(limit * 1.5)
        except Exception as e:  # API error, timeout, empty content ...
            _record(False, time.time() - t0, getattr(response, "usage", None), purpose,
                    model, str(e)[:300])
            last_error = e

        print(f"[LLM] attempt {attempt}/{s['max_retries']} failed: {last_error}")
        if attempt < s["max_retries"]:
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"LLM call failed after {s['max_retries']} attempts: {last_error}")


class _LengthError(RuntimeError):
    pass


# ------------------------------------------------------------------
# JSON helper (same parsing tolerance for every method)
# ------------------------------------------------------------------

def extract_json(text):
    if text is None:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def generate_json(prompt: str, attempts: int = 2, purpose: str = "json", **kwargs):
    """Call the model and parse a JSON object; re-ask once if the JSON is invalid."""
    last = None
    for attempt in range(1, attempts + 1):
        raw = generate_summary(prompt, purpose=purpose, **kwargs)
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
        last = raw
        print(f"[LLM] invalid JSON ({purpose}), attempt {attempt}/{attempts}")
    raise RuntimeError(f"Invalid JSON after {attempts} attempts: {str(last)[:200]}")