"""Google Gemini Developer API integration (official ``google-genai`` SDK).

Security and privacy design
---------------------------
* The API key is read from ``GEMINI_API_KEY`` (optionally from a local ``.env``
  file). It is never returned to the browser or written to logs.
* Uploaded resumes are **never** sent automatically. The caller must pass an
  explicit consent flag, and only a redacted, length-limited excerpt is sent.
* Missing configuration, network failures, rate limits, invalid JSON, blocked
  responses and unavailable models all raise :class:`GeminiError` with a safe
  user-facing message and an HTTP status code.

The service never edits the deterministic NLP scores. Its output is advisory.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import time

from .ner_extractor import _EMAIL_RE, _PHONE_RE, _URL_RE
from .prompt_templates import (
    TASKS,
    ResponseValidationError,
    build_prompt,
    validate_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACKS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)
MAX_RESUME_CONTEXT_CHARS = 7000
MAX_JOB_CONTEXT_CHARS = 3000
REQUEST_TIMEOUT_MS = 45_000


class GeminiError(RuntimeError):
    """A safe, classified Gemini failure."""

    def __init__(self, message: str, status_code: int = 502, code: str = "gemini_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def _sdk_installed() -> bool:
    return importlib.util.find_spec("google.genai") is not None


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - optional dependency
        pass


def _classify_exception(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".lower()
    if any(k in message for k in ("429", "resource_exhausted", "rate limit", "quota")):
        return "rate_limit"
    if any(k in message for k in ("timeout", "deadline exceeded", "timed out")):
        return "timeout"
    if any(k in message for k in ("connection", "network", "dns", "ssl", "proxy")):
        return "network"
    if any(k in message for k in ("api key", "api_key", "permission", "unauthenticated", "401", "403")):
        return "auth"
    if any(
        k in message
        for k in (
            "404",
            "503",
            "not_found",
            "not found",
            "does not exist",
            "unsupported model",
            "no such model",
            "unavailable",
            "high demand",
            "overloaded",
        )
    ):
        return "model_unavailable"
    if any(k in message for k in ("400", "invalid argument", "invalid_argument", "bad request")):
        return "invalid_request"
    return "api_error"


_ERROR_MESSAGES = {
    "rate_limit": "Gemini rate limit reached for this model. Trying a fallback model.",
    "timeout": "Gemini did not respond in time. Check your connection and retry.",
    "network": "Could not reach Gemini. Check your network or proxy settings.",
    "auth": "Gemini rejected the API key. Check GEMINI_API_KEY in your environment.",
    "model_unavailable": "The configured Gemini model is unavailable or overloaded. Trying a fallback model.",
    "invalid_request": "Gemini rejected the request as invalid.",
    "api_error": "Gemini returned an unexpected error.",
}

_ERROR_STATUS = {
    "rate_limit": 429,
    "timeout": 504,
    "network": 502,
    "auth": 502,
    "model_unavailable": 502,
    "invalid_request": 502,
    "api_error": 502,
}


def redact_text(text: str, names: list[str] | None = None) -> tuple[str, dict]:
    """Remove direct identifiers before any text is sent to the cloud.

    Emails, URLs and phone numbers are replaced with placeholders. Known person
    names (from local NER) are replaced when supplied. The summary contains only
    counts, never the removed values.
    """
    summary = {"emails": 0, "urls": 0, "phones": 0, "names": 0}

    def _sub_email(match):
        summary["emails"] += 1
        return "[EMAIL_REDACTED]"

    def _sub_url(match):
        summary["urls"] += 1
        return "[URL_REDACTED]"

    def _sub_phone(match):
        digits = re.sub(r"\D", "", match.group(0))
        if 10 <= len(digits) <= 13:
            summary["phones"] += 1
            return "[PHONE_REDACTED]"
        return match.group(0)

    redacted = _EMAIL_RE.sub(_sub_email, text or "")
    redacted = _URL_RE.sub(_sub_url, redacted)
    redacted = _PHONE_RE.sub(_sub_phone, redacted)
    for name in names or []:
        name = str(name or "").strip()
        if len(name) < 3:
            continue
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.I)
        redacted, count = pattern.subn("[NAME_REDACTED]", redacted)
        summary["names"] += count
    return redacted, summary


class GeminiService:
    """Configured Gemini client with model fallback and validated JSON output."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client=None,
        fallback_models: list[str] | None = None,
    ):
        if api_key is None:
            _load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_key = (api_key or "").strip()
        self.model = (model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL).strip()
        if fallback_models is None:
            raw = os.environ.get("GEMINI_FALLBACK_MODELS", "")
            fallback_models = [m.strip() for m in raw.split(",") if m.strip()] or list(DEFAULT_FALLBACKS)
        self.fallback_models = [m for m in fallback_models if m and m != self.model]
        self._client = client
        self._client_error: str | None = None

    # -- status ------------------------------------------------------------
    @property
    def configured(self) -> bool:
        """True when a key is present, independently of optional SDK availability."""
        return bool(self.api_key)

    def status(self) -> dict:
        installed = _sdk_installed()
        if not self.api_key:
            note = (
                "Gemini is not configured. Add GEMINI_API_KEY to your environment or .env file. "
                "All local NLP features continue to work without it."
            )
        elif not installed:
            note = "The google-genai package is not installed. Run: pip install google-genai"
        else:
            note = "Gemini is configured. Cloud requests require explicit consent for each analysis."
        return {
            "provider": "Google Gemini Developer API",
            "sdk": "google-genai",
            "sdk_installed": installed,
            "api_key_present": bool(self.api_key),
            "configured": self.configured,
            "model": self.model,
            "fallback_models": self.fallback_models,
            "cloud_features_enabled": self.configured and installed,
            "consent_required": True,
            "note": note,
        }

    # -- client ------------------------------------------------------------
    def _get_client(self):
        if not self.api_key:
            raise GeminiError(
                "Gemini is not configured. Set GEMINI_API_KEY and restart the app.",
                503,
                "not_configured",
            )
        if not _sdk_installed():
            raise GeminiError(
                "The google-genai package is not installed. Run: pip install google-genai",
                503,
                "sdk_missing",
            )
        if self._client is None:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
                )
            except Exception as exc:  # noqa: BLE001
                self._client_error = f"{type(exc).__name__}: {exc}"
                raise GeminiError(
                    "Could not initialise the Gemini client. Check the installed google-genai version.",
                    502,
                    "client_init_failed",
                ) from exc
        return self._client

    def _config(self, system_instruction: str):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.25,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

    @staticmethod
    def _response_text(response) -> str:
        try:
            text = getattr(response, "text", None)
        except Exception:  # noqa: BLE001 - some SDK versions raise on blocked output
            text = None
        if not text:
            reason = None
            try:
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    reason = getattr(candidates[0], "finish_reason", None)
            except Exception:  # noqa: BLE001
                reason = None
            suffix = f" (finish reason: {reason})" if reason else ""
            raise GeminiError(
                f"Gemini returned an empty or blocked response{suffix}. Try rephrasing the input.",
                502,
                "empty_response",
            )
        return text

    # -- generation --------------------------------------------------------
    def generate(self, task: str, context: dict) -> dict:
        if task not in TASKS:
            raise GeminiError(f"Unknown AI task '{task}'.", 400, "unknown_task")
        if not self.configured:
            raise GeminiError(
                "Gemini is not configured. Set GEMINI_API_KEY to use cloud AI features.",
                503,
                "not_configured",
            )

        payload = build_prompt(task, context)
        client = self._get_client()
        models = [self.model, *self.fallback_models]
        last_error: GeminiError | None = None
        for model in models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=payload.user,
                    config=self._config(payload.system),
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                logger.warning("Gemini call failed on %s: %s: %s", model, type(exc).__name__, exc)
                code = _classify_exception(exc)
                last_error = GeminiError(
                    _ERROR_MESSAGES[code],
                    _ERROR_STATUS[code],
                    code,
                )
                if code in ("model_unavailable", "timeout", "rate_limit") and model != models[-1]:
                    continue  # try the next documented fallback model
                raise last_error from exc

            text = self._response_text(response)
            try:
                data = validate_response(task, text, allowed_bullets=context.get("bullets"))
            except ResponseValidationError as exc:
                raise GeminiError(
                    f"Gemini returned a response that did not match the expected format: {exc}",
                    502,
                    "invalid_response",
                ) from exc
            return {
                "data": data,
                "model": model,
                "template_id": payload.template_id,
                "advisory_only": True,
                "does_not_change_scores": True,
                "note": (
                    "AI-generated advisory content. Local similarity scores are unchanged and "
                    "the assistant does not make hiring decisions."
                ),
            }
        raise last_error or GeminiError("All configured Gemini models were unavailable.", 502, "model_unavailable")

    def test_connection(self) -> dict:
        """Send a tiny no-resume prompt to verify credentials and model access."""
        if not self.configured:
            raise GeminiError(
                "Gemini is not configured. Set GEMINI_API_KEY to test the connection.",
                503,
                "not_configured",
            )
        client = self._get_client()
        models = [self.model, *self.fallback_models]
        started = time.perf_counter()
        last_error: GeminiError | None = None
        for model in models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents="Reply with the single word OK.",
                    config=self._config("You are a connection test. Reply with OK."),
                )
                text = self._response_text(response)
                return {
                    "ok": True,
                    "model": model,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "response_preview": text[:80],
                }
            except GeminiError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini test failed on %s: %s: %s", model, type(exc).__name__, exc)
                code = _classify_exception(exc)
                last_error = GeminiError(_ERROR_MESSAGES[code], _ERROR_STATUS[code], code)
                if code in ("model_unavailable", "timeout", "rate_limit") and model != models[-1]:
                    continue
                raise last_error from exc
        raise last_error or GeminiError("Connection test failed.", 502, "api_error")


def gemini_status(service: GeminiService | None = None) -> dict:
    return (service or GeminiService()).status()
