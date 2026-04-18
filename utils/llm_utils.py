"""
utils/llm_utils.py

Central LLM wrapper with proactive rate limiting.

Problem this solves:
  OpenAI's gpt-4o-mini has a 200,000 TPM (tokens per minute) limit on free tier.
  A single care-gap agent run consumes ~180k tokens across 13 sequential steps.
  When any agent runs, the TPM bucket gets drained to near zero. The NEXT call
  (even inside the same workflow) then hits 429 because there's no headroom.

  The old approach caught 429s and retried after 15s — but 15s isn't enough
  when the bucket is fully drained. OpenAI's token bucket refills at a
  constant rate (TPM/60 per second), so a fully drained bucket needs ~60s
  of idle time to fully refill. Reactive retry was fighting this losing battle.

Solution: proactive token budget tracking.
  - Before each LLM call, estimate its token cost using tiktoken
  - Track tokens spent in the last 60 seconds (sliding window)
  - If the estimated call would exceed budget, sleep PROACTIVELY until headroom exists
  - On 429 responses, use the actual retry-after hint from OpenAI
  - On bucket-fully-drained 429s, force-record TPM_LIMIT to block subsequent
    calls proactively, then wait 65s for the window to clear

This prevents 429s from happening in the first place, making workflows
reliable instead of lucky.
"""
import os
import re
import threading
import time
from collections import deque
from typing import Optional

import openai


# ── Configuration ─────────────────────────────────────────────────────────────

# OpenAI TPM limits for gpt-4o-mini by tier:
#   Free tier:  200,000 TPM  (default if env var not set)
#   Tier 1:   2,000,000 TPM  (unlocked after $5 spend + 7 days)
#   Tier 2:   2,000,000 TPM  ($50 spend + 7 days)
#   Tier 3:   4,000,000 TPM  ($100 spend + 7 days)
#
# Set OPENAI_TPM_LIMIT on Railway to override. For Tier 1, use 2000000.
# Check your current tier at platform.openai.com → Settings → Limits.
TPM_LIMIT = int(os.getenv("OPENAI_TPM_LIMIT", "200000"))
TPM_SAFETY_MARGIN = 0.90
TPM_USABLE = int(TPM_LIMIT * TPM_SAFETY_MARGIN)

_MAX_ATTEMPTS = 6


# ── Sliding-window token tracker ──────────────────────────────────────────────

class TokenBudget:
    """
    Thread-safe sliding 60-second token budget tracker.
    Mirrors OpenAI's TPM rate limit algorithm.
    """

    def __init__(self, limit: int = TPM_USABLE):
        self.limit = limit
        self._history: deque = deque()
        self._lock = threading.Lock()

    def _prune(self):
        cutoff = time.time() - 60
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def current_usage(self) -> int:
        with self._lock:
            self._prune()
            return sum(tokens for _, tokens in self._history)

    def record(self, tokens: int):
        with self._lock:
            self._history.append((time.time(), tokens))

    def available(self) -> int:
        return max(0, self.limit - self.current_usage())

    def wait_until_available(self, needed: int) -> float:
        """
        Block until there's headroom for `needed` tokens.

        Sticks to OpenAI's actual 200k TPM reality: if the 60-second window
        says we've used too much, wait. The noisy per-call wait messages are
        squelched — only logs if the wait is meaningful (>3s total).

        Returns total seconds slept.
        """
        if needed >= self.limit:
            print(f"  [TokenBudget] Request size {needed} > limit {self.limit}. Waiting 60s...")
            time.sleep(60)
            return 60

        total_slept = 0.0
        logged = False
        while True:
            with self._lock:
                self._prune()
                current = sum(tokens for _, tokens in self._history)
                if current + needed <= self.limit:
                    if logged:
                        print(f"  [TokenBudget] OK after {total_slept:.1f}s wait (usage {current}/{self.limit})")
                    return total_slept

                if not self._history:
                    wait = 1.0
                else:
                    oldest_ts = self._history[0][0]
                    wait = max(0.5, (oldest_ts + 60) - time.time() + 0.2)
                    wait = min(wait, 10.0)

            # Only log if we're going to wait more than 3 seconds total.
            # Prevents the "50 waiting-1-second lines" spam.
            if not logged and total_slept + wait > 3.0:
                print(f"  [TokenBudget] Usage {current}/{self.limit}, need {needed} — waiting up to a few seconds...")
                logged = True

            time.sleep(wait)
            total_slept += wait

            if total_slept > 90:
                print(f"  [TokenBudget] Exceeded max wait (90s). Proceeding anyway.")
                return total_slept


_budget = TokenBudget()


# ── Token estimator ───────────────────────────────────────────────────────────

_tiktoken_enc = None

def _get_tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            _tiktoken_enc = False
    return _tiktoken_enc


def estimate_tokens(messages: list) -> int:
    """
    Estimate token count for a message list.

    Previous version multiplied real token count by 1.5x AND added a 2000-token
    response buffer, which turned a real 2k-token ReAct call into an 11k+
    estimate. That caused livelock: the TPM budget would never have enough
    headroom for the *estimate*, even when actual usage was far below.

    Fixed approach:
      - Count actual message tokens via tiktoken (accurate)
      - Add a fixed 1500 tokens for tool definitions (LangChain injects these
        invisibly — prior auth has ~15 tools, ~100 tokens each)
      - Add 1000 tokens for expected response
      - Apply a modest 1.1x safety margin (not 1.5x)
      - Cap the estimate at 30k — no single call to gpt-4o-mini legitimately
        needs more than this; anything higher indicates the estimator is off
        and would block forever. If the actual call exceeds this, we get a
        429 and fall back to reactive retry, which is fine.
    """
    enc = _get_tiktoken()

    if enc:
        total = 0
        for msg in messages:
            content = ""
            if hasattr(msg, "content"):
                content = msg.content or ""
            elif isinstance(msg, dict):
                content = msg.get("content", "") or ""

            if isinstance(content, str):
                total += len(enc.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total += len(enc.encode(block.get("text", "")))
            total += 4

        # Fixed overhead, not multiplicative
        total += 1500  # tool definition overhead (~15 tools × ~100 tokens)
        total += 1000  # expected response
        estimate = int(total * 1.1)  # modest safety margin

        # Hard cap — prevents livelock if a single message is huge
        return min(estimate, 30000)

    # Fallback heuristic (no tiktoken)
    total_chars = 0
    for msg in messages:
        content = ""
        if hasattr(msg, "content"):
            content = msg.content or ""
        elif isinstance(msg, dict):
            content = msg.get("content", "") or ""
        if isinstance(content, str):
            total_chars += len(content)
    # ~4 chars per token + tool overhead (1500) + response (1000) + 10% margin
    estimate = int((total_chars / 4 + 2500) * 1.1)
    return min(estimate, 30000)


# ── Retry helpers ─────────────────────────────────────────────────────────────

def _parse_retry_seconds(err: openai.RateLimitError) -> Optional[float]:
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


def _is_bucket_fully_drained(err: openai.RateLimitError) -> bool:
    """Detect when OpenAI reports usage >= 97% of TPM_LIMIT (bucket effectively full)."""
    msg = str(err)
    m = re.search(r"Used (\d+)", msg)
    if m:
        used = int(m.group(1))
        return used >= TPM_LIMIT * 0.97
    return False


def _extract_actual_usage(result) -> Optional[int]:
    """
    Pull actual total_tokens from LangChain response.

    LangChain's AIMessage exposes usage in multiple places depending on version:
      - result.usage_metadata (UsageMetadata TypedDict with total_tokens attr)
      - result.response_metadata["token_usage"]["total_tokens"]
      - result.response_metadata["usage"]["total_tokens"]

    We try them all. If nothing works we return None and the caller falls back
    to the estimate (which is conservative).
    """
    # Path 1: usage_metadata (newer LangChain) — it's a TypedDict, not a plain dict
    try:
        um = getattr(result, "usage_metadata", None)
        if um:
            # TypedDicts support both attribute and subscript access
            total = getattr(um, "total_tokens", None)
            if total is None and isinstance(um, dict):
                total = um.get("total_tokens")
            if total:
                return int(total)
    except Exception:
        pass

    # Path 2: response_metadata → token_usage/usage → total_tokens
    try:
        rm = getattr(result, "response_metadata", None)
        if rm and isinstance(rm, dict):
            for key in ("token_usage", "usage"):
                usage = rm.get(key)
                if isinstance(usage, dict):
                    total = usage.get("total_tokens")
                    if total:
                        return int(total)
    except Exception:
        pass

    return None


# ── Main wrapper ──────────────────────────────────────────────────────────────

def llm_invoke(llm, messages):
    """
    Call llm.invoke(messages) with proactive rate limiting and retry.

    Flow:
      1. Estimate tokens needed
      2. Wait until budget has headroom (proactive — prevents 429s)
      3. Make the call
      4. Record actual token usage
      5. On 429: use OpenAI's hint or 65s for full drain
      6. On connection error: exponential backoff
    """
    needed = estimate_tokens(messages)
    _budget.wait_until_available(needed)

    for attempt in range(_MAX_ATTEMPTS):
        try:
            result = llm.invoke(messages)
            actual = _extract_actual_usage(result)
            _budget.record(actual if actual else needed)
            return result

        except openai.AuthenticationError as e:
            raise RuntimeError(
                f"[LLM] Authentication failed — check OPENAI_API_KEY. Error: {e}"
            ) from e

        except openai.RateLimitError as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise

            # Don't record `needed` on 429 — we'd double-count. OpenAI already
            # tracked the usage on their end (that's why it returned 429). Our
            # tracker catches up when the next successful call records its
            # actual usage from response metadata.

            if _is_bucket_fully_drained(e):
                wait = 65
                print(f"  [RateLimit] Bucket drained. Waiting {wait}s for full refill...")
            else:
                parsed = _parse_retry_seconds(e)
                wait = max(parsed or 0, 20 * (attempt + 1))
                wait = min(wait, 90)
                print(
                    f"  [RateLimit] 429 — waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{_MAX_ATTEMPTS - 1})"
                )

            time.sleep(wait)

        except openai.APIConnectionError as e:
            if attempt == _MAX_ATTEMPTS - 1:
                raise RuntimeError(
                    f"[LLM] Cannot reach OpenAI after {_MAX_ATTEMPTS} attempts. "
                    f"Check OPENAI_API_KEY and account credits. Error: {e}"
                ) from e
            wait = min(3 * (2 ** attempt), 30)
            print(
                f"  [ConnError] Connection error — waiting {wait}s "
                f"(attempt {attempt + 1}/{_MAX_ATTEMPTS - 1}): {e}"
            )
            time.sleep(wait)


# ── Startup credential check ──────────────────────────────────────────────────

def check_openai_credentials() -> dict:
    """Validate OPENAI_API_KEY at FastAPI startup."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "message": "OPENAI_API_KEY env var is not set"}
    if not api_key.startswith("sk-"):
        return {"ok": False, "message": "OPENAI_API_KEY doesn't look valid (should start with 'sk-')"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.models.list()
        return {
            "ok": True,
            "message": f"OpenAI credentials valid. TPM: {TPM_LIMIT} (using {TPM_USABLE})",
            "tpm_limit": TPM_LIMIT,
            "tpm_usable": TPM_USABLE,
        }
    except openai.AuthenticationError as e:
        return {"ok": False, "message": f"Invalid OPENAI_API_KEY: {e}"}
    except Exception as e:
        return {"ok": False, "message": f"OpenAI check failed: {e}"}


# ── Debug endpoint helper ─────────────────────────────────────────────────────

def get_budget_status() -> dict:
    """Return current token budget state for /diagnostics endpoint."""
    return {
        "tpm_limit": TPM_LIMIT,
        "tpm_usable": TPM_USABLE,
        "current_usage": _budget.current_usage(),
        "available": _budget.available(),
        "headroom_pct": round(_budget.available() / TPM_USABLE * 100, 1),
    }