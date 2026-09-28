"""
Thin abstraction over LLM providers (OpenAI-compatible API).

Supports:
  - openai / azure / any OpenAI-compatible endpoint
  - Structured JSON output via function-calling / json_mode
"""

from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
from typing import Optional

from openai import OpenAI

from config.settings import LLMConfig, get_config
from src.utils.logging import log

# Known OpenAI-compatible providers: default API base (blank = OpenAI's own
# default) and default chat model to use when .env doesn't set one
# explicitly. Embeddings always default to provider "openai" regardless of
# chat provider (see LLMClient.__init__) — the "embedding_model" entries
# here are only used if TABLEQA_EMBEDDING_PROVIDER explicitly opts into one
# of these providers for embeddings too.
_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "api_base": "",
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "azure": {
        "api_base": "",
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "aqueduct": {
        "api_base": "",
        "chat_model": "",
        "embedding_model": "text-embedding-3-small",
    },
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1",
        "chat_model": "deepseek-chat",
        "embedding_model": "",
    },
    "gemini": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "chat_model": "gemini-2.5-flash",
        "embedding_model": "text-embedding-004",
    },
    # Offline stand-in used to validate the pipeline without credentials; see
    # src/llm/mock_provider.py. Never reaches the network.
    "mock": {
        "api_base": "",
        "chat_model": "mock-model",
        "embedding_model": "mock-embedding",
    },
}


def _normalize_api_base(provider: str, explicit_base: str, provider_default: str = "") -> str:
    """Resolve a provider's OpenAI-compatible base URL.

    Uses ``explicit_base`` (from .env) when given, else the provider's known
    default; then applies provider-specific quirks (aqueduct needs a ``/v1``
    suffix).
    """
    base = (explicit_base or provider_default).strip()
    if provider == "aqueduct" and base:
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
    return base


def _provider_env(provider: str, suffix: str) -> str:
    """Look up ``TABLEQA_<PROVIDER>_<SUFFIX>`` then ``<PROVIDER>_<SUFFIX>``."""
    provider_upper = provider.upper()
    for key in (f"TABLEQA_{provider_upper}_{suffix}", f"{provider_upper}_{suffix}"):
        val = os.environ.get(key, "")
        if val:
            return val
    return ""


# ---------------------------------------------------------------------------
# Transient-failure retry
# ---------------------------------------------------------------------------
# Status codes worth retrying: rate limits and the transient server-side
# failures every hosted provider returns under load (Gemini in particular
# answers 503 UNAVAILABLE during demand spikes). Anything else is a real error
# and is raised immediately.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_RETRY_ATTEMPTS = 6
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 30.0

# Long verbalization/clarification calls generate thousands of tokens in one
# response. Without an explicit timeout the SDK waits 600s on a connection that
# has already died, and the sweep spends its time blocked rather than retrying.
_REQUEST_TIMEOUT = float(os.environ.get("TABLEQA_LLM_TIMEOUT", "180"))

# A 429 can mean "slow down" (worth retrying) or "your quota/credits are gone"
# (retrying just burns time -- an exhausted key stays exhausted). Providers
# distinguish these only in the message.
_EXHAUSTED_MARKERS = (
    "exceeded your current quota",
    "check your plan and billing",
    "insufficient_quota",
    "insufficient balance",
    "billing",
)

# ...except that Gemini's *per-minute* rate limit returns that same
# "exceeded your current quota / check your plan and billing" wording while
# also telling us exactly how long to wait. Treating it as a dead key kills a
# free-tier run that would have succeeded after a short sleep, so an explicit
# retry hint always wins over the markers above.
_RETRY_DELAY_RE = re.compile(
    r"(?:retrydelay['\"]?\s*[:=]\s*['\"]?|please retry in\s*)(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)
# A per-minute quota replenishes within the life of a run; a per-DAY quota does
# not, even though Gemini attaches a (useless) few-second retryDelay to it. Only
# the short-window limits are worth retrying.
_RATE_LIMIT_MARKERS = ("perminute", "per minute", "requests per minute",
                       "rate_limit", "rate limit")
# Quotas whose window outlasts any sensible retry loop: treat as terminal so the
# sweep records the cell and moves on instead of sleeping for hours.
_LONG_WINDOW_MARKERS = ("perday", "per day", "requests per day", "daily limit")


def _retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, when it said so explicitly."""
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return float(match.group(1))
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        for key in ("retry-after", "Retry-After"):
            try:
                return float(headers.get(key))
            except (TypeError, ValueError):
                continue
    return None


def _is_exhausted(exc: Exception) -> bool:
    """Whether an error means the account is out of quota rather than too fast."""
    text = str(exc).lower()
    # A daily cap is terminal for this run regardless of the retry hint the
    # provider attaches: the window is 24h, so no retry loop will outlast it.
    if any(marker in text for marker in _LONG_WINDOW_MARKERS):
        return True
    # An explicit "retry in Ns", or a quota named per-minute, refills within the
    # run. Only a quota with neither is treated as terminal.
    if _retry_after(exc) is not None:
        return False
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return False
    return any(marker in text for marker in _EXHAUSTED_MARKERS)


def _is_transient(exc: Exception) -> bool:
    """Whether an API exception looks worth retrying."""
    if _is_exhausted(exc):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status in _RETRY_STATUS:
        return True
    # Connection resets and timeouts surface without a status code.
    return exc.__class__.__name__ in {
        "APIConnectionError", "APITimeoutError", "InternalServerError",
        "RateLimitError",
    }


def _with_retry(call, what: str):
    """Run an API call, retrying transient failures with exponential backoff.

    A benchmark sweep makes thousands of calls over hours, so a single 503
    should cost a few seconds rather than the whole run.
    """
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == _RETRY_ATTEMPTS or not _is_transient(exc):
                raise
            suggested = _retry_after(exc)
            delay = (suggested + 1.0 if suggested is not None
                     else min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY))
            log.warning(
                "%s failed (%s); retry %d/%d in %.0fs",
                what, type(exc).__name__, attempt, _RETRY_ATTEMPTS - 1, delay,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Client-side rate limiting
# ---------------------------------------------------------------------------
# Gemini's free tier allows 5 requests/minute per model. Firing a pool of
# concurrent verbalization calls at it produces nothing but 429s, so requests
# are paced client-side. Set TABLEQA_LLM_RPM to override (0 = unlimited).
_PROVIDER_RPM: dict[str, float] = {
    "gemini": 5.0,
}


class _RateLimiter:
    """Spaces requests so they do not exceed a requests-per-minute budget."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            time.sleep(wait)


class LLMClient:
    """Wrapper around OpenAI-compatible chat completions."""

    def __init__(self, config: Optional[LLMConfig] = None):
        cfg = config or get_config().llm
        self.provider = (cfg.provider or "openai").strip().lower()
        provider_defaults = _PROVIDER_DEFAULTS.get(self.provider, {})
        self.model = cfg.model or provider_defaults.get("chat_model") or "gpt-4o"
        self.temperature = cfg.temperature
        self.max_completion_tokens = cfg.max_completion_tokens
        self.api_base = _normalize_api_base(
            self.provider, cfg.api_base, provider_defaults.get("api_base", "")
        )

        if self.provider not in _PROVIDER_DEFAULTS and not self.api_base:
            raise ValueError(
                f"Unsupported LLM provider '{cfg.provider}'. "
                f"Use one of {'/'.join(sorted(_PROVIDER_DEFAULTS))} or set "
                "TABLEQA_LLM_API_BASE for an OpenAI-compatible endpoint."
            )
        if self.provider == "aqueduct" and not self.api_base:
            raise ValueError(
                "Aqueduct requires an API base URL. Set TABLEQA_LLM_API_BASE "
                "to your Aqueduct instance URL ending in /v1."
            )

        self.is_mock = self.provider == "mock"
        if self.is_mock:
            # No network, no credentials. Everything above and below this line
            # behaves exactly as it does for a live provider.
            self._client = None
        else:
            kwargs: dict = {"api_key": cfg.api_key, "timeout": _REQUEST_TIMEOUT}
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._client = OpenAI(**kwargs)

        rpm_override = os.environ.get("TABLEQA_LLM_RPM", "")
        rpm = float(rpm_override) if rpm_override else _PROVIDER_RPM.get(self.provider, 0.0)
        self.requests_per_minute = rpm
        self._limiter = _RateLimiter(rpm)

        # ------------------------------------------------------------
        # Embeddings: ALWAYS OpenAI by default, independent of whatever
        # chat provider/model is configured (gpt/deepseek/gemini/...), so
        # switching the chat provider for experiments never changes the
        # retrieval embeddings as a side effect. Only an explicit
        # TABLEQA_EMBEDDING_PROVIDER override changes this.
        # ------------------------------------------------------------
        self._embedding_provider = cfg.embedding_provider or "openai"
        embed_defaults = _PROVIDER_DEFAULTS.get(self._embedding_provider, {})
        self.embedding_model = (
            cfg.embedding_model or embed_defaults.get("embedding_model") or "text-embedding-3-small"
        )

        same_provider = self._embedding_provider == self.provider
        self._embedding_api_key = (
            cfg.embedding_api_key
            or _provider_env(self._embedding_provider, "API_KEY")
            or (cfg.api_key if same_provider else "")
        )
        embed_explicit_base = cfg.embedding_api_base or _provider_env(self._embedding_provider, "API_BASE")
        if not embed_explicit_base and same_provider:
            self._embedding_api_base = self.api_base
        else:
            self._embedding_api_base = _normalize_api_base(
                self._embedding_provider, embed_explicit_base, embed_defaults.get("api_base", "")
            )
        # Built lazily on first embed() call, so chat-only runs never need an
        # embeddings-capable key/provider configured at all.
        self._embedding_client: Optional[OpenAI] = None

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------
    def _complete(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
        max_completion_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat completion request; return the text response.

        ``max_completion_tokens`` overrides the client default for this call
        only. Passing it per call keeps concurrent callers independent, which
        mutating ``self.max_completion_tokens`` would not.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": max_completion_tokens or self.max_completion_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        extra_body = self._extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        log.debug(
            "LLM request: %d messages, provider=%s, model=%s",
            len(messages),
            self.provider,
            self.model,
        )
        if self.is_mock:
            from src.llm import mock_provider
            return mock_provider.generate(messages)
        def call():
            self._limiter.acquire()
            return self._client.chat.completions.create(**kwargs)

        resp = _with_retry(call, what=f"chat({self.model})")
        choice = resp.choices[0]
        message = choice.message
        content = message.content or ""
        if not content:
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            if reasoning:
                raise ValueError(
                    "LLM returned reasoning output but no final answer content "
                    f"(finish_reason={choice.finish_reason}). "
                    "For Qwen reasoning models, disable thinking mode or raise the completion token budget."
                )
            raise ValueError(
                "LLM returned empty content "
                f"(finish_reason={choice.finish_reason})."
            )
        log.debug("LLM response: %d chars", len(content))
        return content

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: Optional[int] = None,
    ) -> str:
        """Simple system + user chat."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._complete(messages, max_completion_tokens=max_completion_tokens)

    def _get_embedding_client(self) -> OpenAI:
        """Build (and cache) the OpenAI client used for embeddings.

        Deferred until first use so that chat-only runs never require an
        embeddings-capable API key/provider to be configured.
        """
        if self._embedding_client is None:
            if not self._embedding_api_key:
                raise ValueError(
                    f"No API key configured for embeddings (provider="
                    f"'{self._embedding_provider}'). Set TABLEQA_EMBEDDING_API_KEY "
                    f"or TABLEQA_{self._embedding_provider.upper()}_API_KEY."
                )
            kwargs: dict = {"api_key": self._embedding_api_key,
                            "timeout": _REQUEST_TIMEOUT}
            if self._embedding_api_base:
                kwargs["base_url"] = self._embedding_api_base
            self._embedding_client = OpenAI(**kwargs)
        return self._embedding_client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per input (same order).

        Uses the configured embedding provider/model via the OpenAI-compatible
        ``embeddings`` endpoint (independent of the chat provider — see
        ``_embedding_provider`` in ``__init__``). Batched into a single request.
        """
        if not texts:
            return []
        log.debug(
            "LLM embedding request: %d texts, provider=%s, model=%s",
            len(texts),
            self._embedding_provider,
            self.embedding_model,
        )
        if self.is_mock or self._embedding_provider == "mock":
            from src.llm import mock_provider
            return [mock_provider.embed(text) for text in texts]
        client = self._get_embedding_client()
        resp = _with_retry(
            lambda: client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            ),
            what=f"embed({self.embedding_model})",
        )
        data = sorted(resp.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]

    # Providers whose OpenAI-compatible endpoint reliably honors
    # response_format={"type": "json_object"}. Left off for providers we
    # don't control the backend of (e.g. aqueduct's gateway to varying
    # self-hosted models), where an unsupported param could error out.
    _JSON_MODE_PROVIDERS = {"openai", "azure", "deepseek", "gemini"}

    # Appended to every chat_json system prompt: many models (Gemini
    # especially) default to pretty-printed JSON with indentation, which
    # burns a lot of the completion-token budget on whitespace and can get
    # the response truncated mid-object for large schemas. Every chat_json
    # call site already asks for JSON, so this is safe to apply generically.
    _COMPACT_JSON_SUFFIX = (
        "\n\nOutput must be a single line of compact, minified JSON: no "
        "indentation, no line breaks, no markdown code fences, no "
        "commentary before or after the JSON."
    )

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Chat for a JSON object and parse it with tolerant fallbacks."""
        messages = [
            {"role": "system", "content": system_prompt + self._COMPACT_JSON_SUFFIX},
            {"role": "user", "content": user_prompt},
        ]
        response_format = (
            {"type": "json_object"} if self.provider in self._JSON_MODE_PROVIDERS else None
        )
        text = self._complete(messages, response_format=response_format)
        return self._extract_json(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> dict:
        """Best-effort JSON extraction from LLM output."""
        candidates = LLMClient._json_candidates(text)
        for candidate in candidates:
            parsed = LLMClient._parse_json_candidate(candidate)
            if isinstance(parsed, dict):
                return parsed
        preview = text.strip().replace("\n", " ")
        raise ValueError(f"Could not parse JSON from: {preview[:200]}...")

    @staticmethod
    def _json_candidates(text: str) -> list[str]:
        """Generate likely JSON substrings from raw model output."""
        stripped = text.strip()
        candidates: list[str] = []
        if stripped:
            candidates.append(stripped)

        if stripped.startswith("```"):
            fenced = re.sub(r"^```(?:json)?\s*", "", stripped, count=1)
            fenced = re.sub(r"\s*```$", "", fenced, count=1)
            if fenced and fenced not in candidates:
                candidates.append(fenced.strip())

        balanced = LLMClient._first_balanced_json_block(stripped)
        if balanced and balanced not in candidates:
            candidates.append(balanced)

        return candidates

    @staticmethod
    def _parse_json_candidate(candidate: str) -> dict | list | None:
        """Parse one likely JSON candidate using tolerant fallbacks."""
        normalized = candidate.strip()
        if not normalized:
            return None

        for attempt in (
            normalized,
            LLMClient._strip_json_prefix(normalized),
            LLMClient._remove_trailing_commas(normalized),
            LLMClient._remove_trailing_commas(
                LLMClient._strip_json_prefix(normalized)
            ),
        ):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

        try:
            parsed = ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            return None
        return parsed if isinstance(parsed, (dict, list)) else None

    @staticmethod
    def _strip_json_prefix(text: str) -> str:
        return re.sub(r"^\s*json\s*[:\-]?\s*", "", text, count=1, flags=re.IGNORECASE)

    @staticmethod
    def _remove_trailing_commas(text: str) -> str:
        return re.sub(r",(\s*[}\]])", r"\1", text)

    @staticmethod
    def _first_balanced_json_block(text: str) -> str:
        """Return the first balanced {...} or [...] block, respecting strings."""
        start = -1
        stack: list[str] = []
        in_string = False
        escaped = False
        for idx, char in enumerate(text):
            if start == -1:
                if char in "{[":
                    start = idx
                    stack.append("}" if char == "{" else "]")
                continue

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append("}" if char == "{" else "]")
            elif stack and char == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start : idx + 1]
        return ""

    def _extra_body(self) -> dict | None:
        """Provider/model-specific request extensions."""
        if "qwen" in self.model.lower():
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None


# Convenience
def get_llm() -> LLMClient:
    return LLMClient()
