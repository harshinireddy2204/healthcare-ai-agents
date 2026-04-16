"""
utils/llm_utils.py

Central retry wrapper for all OpenAI LLM calls.

Why: gpt-4o-mini has a 200k TPM limit. The care gap agent alone can consume
~150-190k tokens across 13 plan steps, leaving little headroom for subsequent
agents (prior auth, drug safety). When the limit is hit, calls fail with 429.

Solution: parse the "try again in X.Xs" wait time from the error, sleep for
that duration + a buffer, then retry. Falls back to exponential backoff if the
wait time can't be parsed. Tries up to 6 times before re-raising.

Usage:
    from utils.llm_utils import llm_invoke
    response = llm_invoke(llm, messages)
"""
import re
import time
import openai


_MAX_ATTEMPTS = 6
_BASE_WAIT_S  = 5   # seconds for first retry when no header is available


def _parse_retry_seconds(err: openai.RateLimitError) -> float | None:
    """Extract the suggested wait time from an OpenAI 429 error message.

    OpenAI returns one of these formats:
      "Please try again in 1.378s"
      "Please try again in 2m3s"
      "Please try again in 45ms"
    """
    msg = str(err)
    # Minutes + seconds: "2m3s"
    m = re.search(r"(\d+)m(\d+\.?\d*)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2)) + 1.5
    # Seconds only: "1.378s"
    m = re.search(r"try again in (\d+\.?\d*)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1.5
    # Milliseconds: "450ms"
    m = re.search(r"try again in (\d+)ms", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000 + 1.5
    return None


def llm_invoke(llm, messages):
    """
    Call llm.invoke(messages) with automatic retry on transient OpenAI errors.

    Handles two error types:
      - RateLimitError (429): waits the time OpenAI suggests, then retries
      - APIConnectionError: brief backoff then retry (transient network drop)

    Waits the time suggested in the error response when available; otherwise
    uses exponential backoff (5 → 10 → 20 → 40 → 60 s).
    """
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return llm.invoke(messages)
        except openai.AuthenticationError as e:
            # Bad API key — retrying won't help, fail immediately with clear message
            raise RuntimeError(
                f"[LLM] Authentication failed — check OPENAI_API_KEY is set correctly. "
                f"Original error: {e}"
            ) from e
        except openai.RateLimitError as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            wait = _parse_retry_seconds(e) or min(_BASE_WAIT_S * (2 ** attempt), 60)
            print(
                f"  [RateLimit] 429 — waiting {wait:.1f}s "
                f"(attempt {attempt + 1}/{_MAX_ATTEMPTS - 1})"
            )
            time.sleep(wait)
        except openai.APIConnectionError as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise RuntimeError(
                    f"[LLM] Cannot reach OpenAI after {_MAX_ATTEMPTS} attempts. "
                    f"Check: (1) OPENAI_API_KEY is set in Railway Variables, "
                    f"(2) account has credits at platform.openai.com/usage. "
                    f"Original error: {e}"
                ) from e
            wait = min(3 * (2 ** attempt), 30)
            print(
                f"  [ConnError] Connection error — waiting {wait}s "
                f"(attempt {attempt + 1}/{_MAX_ATTEMPTS - 1}): {e}"
            )
            time.sleep(wait)
