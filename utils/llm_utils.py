"""
utils/llm_utils.py

Central retry wrapper for all OpenAI LLM calls with DETAILED error reporting.

The OpenAI SDK's APIConnectionError and APIError classes have short __str__
representations ("Connection error.") that hide the real underlying cause.
This module unwraps them so you can actually debug deployment issues.

Common causes of "Connection error" on Railway/Render that this file exposes:
  1. OPENAI_API_KEY is missing or blank → AuthenticationError (was masked)
  2. OPENAI_API_KEY has trailing whitespace from env var copy-paste
  3. Account has no billing / $0 balance → 429 with "insufficient_quota"
  4. Network timeout on cold container → httpx ConnectTimeout
  5. DNS resolution failure → httpx ConnectError
"""
import os
import re
import time
import traceback

import openai


_MAX_ATTEMPTS = 6
_BASE_WAIT_S = 5


def _unwrap_error(err: Exception) -> str:
    """
    Extract the real error detail from an OpenAI SDK exception.

    OpenAI wraps httpx errors and their __str__ is usually just
    "Connection error." which is useless for debugging. This function
    walks the exception chain and body to find the actual cause.
    """
    parts = [type(err).__name__]

    # OpenAI SDK error body often contains JSON with real error info
    body = getattr(err, "body", None)
    if body and isinstance(body, dict):
        msg = body.get("message") or body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else None
        if msg:
            parts.append(f"body.message={msg}")

    # response.status_code reveals if it's auth, rate limit, etc.
    response = getattr(err, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status:
            parts.append(f"status={status}")

    # __cause__ is the underlying httpx error
    cause = err.__cause__ or err.__context__
    if cause:
        parts.append(f"cause={type(cause).__name__}: {cause}")

    # Fallback to the message itself
    msg = str(err)
    if msg and msg != "Connection error.":
        parts.append(f"msg={msg}")

    return " | ".join(parts) if len(parts) > 1 else (msg or type(err).__name__)


def _parse_retry_seconds(err: openai.RateLimitError) -> float | None:
    """Extract the suggested wait time from an OpenAI 429 error message."""
    msg = str(err)
    m = re.search(r"(\d+)m(\d+\.?\d*)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2)) + 1.5
    m = re.search(r"try again in (\d+\.?\d*)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1.5
    m = re.search(r"try again in (\d+)ms", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000 + 1.5
    return None


def check_openai_credentials() -> dict:
    """
    Validate OPENAI_API_KEY at startup by making one tiny test call.

    Returns a dict with 'ok' bool and 'message' string. Used to fail-fast
    if the key is missing/invalid, instead of discovering it only when a
    user triggers an agent workflow.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return {"ok": False, "message": "OPENAI_API_KEY is not set in environment"}
    if not key.startswith("sk-"):
        return {"ok": False, "message": f"OPENAI_API_KEY does not start with 'sk-' (starts with '{key[:3]}')"}
    if key != key.strip():
        return {"ok": False, "message": "OPENAI_API_KEY has leading/trailing whitespace — trim it in Railway Variables"}

    # Make a tiny test call — 1 token, costs fractionally
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=10)
        client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return {"ok": True, "message": "OpenAI credentials valid"}
    except openai.AuthenticationError as e:
        return {"ok": False, "message": f"OPENAI_API_KEY is invalid: {_unwrap_error(e)}"}
    except openai.RateLimitError as e:
        unwrapped = _unwrap_error(e)
        if "insufficient_quota" in unwrapped.lower() or "billing" in unwrapped.lower():
            return {"ok": False, "message": f"OpenAI account has no credits — add billing at platform.openai.com/account/billing. Details: {unwrapped}"}
        return {"ok": True, "message": f"OpenAI reachable but rate-limited: {unwrapped}"}
    except openai.APIConnectionError as e:
        return {"ok": False, "message": f"Cannot reach OpenAI from this container: {_unwrap_error(e)}"}
    except Exception as e:
        return {"ok": False, "message": f"Unexpected error validating key: {type(e).__name__}: {e}"}


def llm_invoke(llm, messages):
    """
    Call llm.invoke(messages) with automatic retry on transient OpenAI errors
    and DETAILED error logging for production debugging.
    """
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return llm.invoke(messages)

        except openai.AuthenticationError as e:
            # Never retry auth errors — key is bad
            detail = _unwrap_error(e)
            raise RuntimeError(
                f"[LLM] Authentication failed — OPENAI_API_KEY is invalid. "
                f"Check Railway Variables tab. Details: {detail}"
            ) from e

        except openai.RateLimitError as e:
            detail = _unwrap_error(e)
            # insufficient_quota means no billing — don't retry
            if "insufficient_quota" in detail.lower():
                raise RuntimeError(
                    f"[LLM] OpenAI account has no credits. "
                    f"Add billing at platform.openai.com/account/billing. "
                    f"Details: {detail}"
                ) from e
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            parsed = _parse_retry_seconds(e)
            wait = max(parsed or 0, 15 * (attempt + 1))
            wait = min(wait, 90)
            print(
                f"  [RateLimit] 429 — waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{_MAX_ATTEMPTS}) — {detail}"
            )
            time.sleep(wait)

        except openai.APIConnectionError as e:
            detail = _unwrap_error(e)
            if attempt == _MAX_ATTEMPTS - 1:
                # Final failure — give users actionable diagnostic info
                raise RuntimeError(
                    f"[LLM] Cannot reach OpenAI after {_MAX_ATTEMPTS} attempts. "
                    f"Diagnostic checklist:\n"
                    f"  1. Is OPENAI_API_KEY set in Railway Variables? (starts with sk-, no whitespace)\n"
                    f"  2. Does the account have credits? platform.openai.com/account/usage\n"
                    f"  3. Is the Railway container able to reach the internet at all?\n"
                    f"Details: {detail}"
                ) from e
            wait = min(3 * (2 ** attempt), 30)
            print(
                f"  [ConnError] {detail} — waiting {wait}s "
                f"(attempt {attempt + 1}/{_MAX_ATTEMPTS})"
            )
            time.sleep(wait)

        except openai.APIError as e:
            # Covers BadRequestError, InternalServerError, etc.
            detail = _unwrap_error(e)
            # Only retry 5xx; 4xx means the request itself is bad
            status = getattr(getattr(e, "response", None), "status_code", 500)
            if status < 500 or attempt == _MAX_ATTEMPTS - 1:
                raise RuntimeError(f"[LLM] OpenAI API error (status={status}): {detail}") from e
            wait = min(3 * (2 ** attempt), 20)
            print(f"  [APIError] status={status} — waiting {wait}s — {detail}")
            time.sleep(wait)

        except Exception as e:
            # Unexpected errors — log full traceback and retry once
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            print(
                f"  [UnexpectedError] {type(e).__name__}: {e} — "
                f"waiting 5s (attempt {attempt + 1}/{_MAX_ATTEMPTS})"
            )
            traceback.print_exc()
            time.sleep(5)