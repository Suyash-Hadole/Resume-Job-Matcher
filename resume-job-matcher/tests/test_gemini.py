"""Tests for the Gemini service: status, redaction, errors and fallbacks.

No network calls are made. A fake client implements the small part of the
``google-genai`` client surface used by :class:`GeminiService`.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nlp.gemini_service as gemini_service  # noqa: E402
from nlp.gemini_service import (  # noqa: E402
    DEFAULT_MODEL,
    GeminiError,
    GeminiService,
    redact_text,
)

VALID_MATCH_JSON = json.dumps(
    {
        "summary": "Strong Python overlap, limited SQL evidence.",
        "strengths": ["Python appears in the resume"],
        "gaps": ["SQL is required but not evidenced"],
        "score_interpretation": "The local score reflects partial skill coverage.",
        "next_steps": ["Add a SQL example if available"],
    }
)


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.candidates = []


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def make_service(outcomes, **kwargs):
    client = FakeClient(outcomes)
    fallback_models = kwargs.pop("fallback_models", [])
    service = GeminiService(
        api_key="unit-test-key",
        model="primary-model",
        client=client,
        fallback_models=fallback_models,
        **kwargs,
    )
    return service, client


def test_status_without_api_key():
    service = GeminiService(api_key="")
    status = service.status()
    assert status["api_key_present"] is False
    assert status["configured"] is False
    assert status["consent_required"] is True
    assert "local NLP features continue to work" in status["note"]
    assert status["model"] == DEFAULT_MODEL


def test_status_reports_key_separately_from_sdk(monkeypatch):
    monkeypatch.setattr(gemini_service, "_sdk_installed", lambda: False)
    service = GeminiService(api_key="unit-test-key")
    status = service.status()
    assert status["configured"] is True
    assert status["sdk_installed"] is False
    assert status["cloud_features_enabled"] is False
    assert "pip install google-genai" in status["note"]


def test_status_with_api_key_never_exposes_the_key():
    service = GeminiService(api_key="super-secret-key", model="unit-model")
    status = service.status()
    assert status["api_key_present"] is True
    assert status["configured"] is True
    assert status["model"] == "unit-model"
    assert status["cloud_features_enabled"] is True
    assert "super-secret-key" not in json.dumps(status)


def test_default_model_is_documented_default():
    assert DEFAULT_MODEL == "gemini-3.8-flash"


def test_redaction_removes_direct_identifiers():
    text = (
        "Jane Doe\njane.doe@example.com\nhttps://example.com/profile\n"
        "+91 98765 43210\nJane Doe built dashboards."
    )
    redacted, summary = redact_text(text, names=["Jane Doe"])
    assert "jane.doe@example.com" not in redacted
    assert "https://example.com/profile" not in redacted
    assert "98765" not in redacted
    assert "Jane Doe" not in redacted
    assert redacted.count("[NAME_REDACTED]") == 2
    assert summary == {"emails": 1, "urls": 1, "phones": 1, "names": 2}


def test_redaction_leaves_safe_text_alone():
    redacted, summary = redact_text("Python and SQL experience with pandas.")
    assert redacted == "Python and SQL experience with pandas."
    assert summary == {"emails": 0, "urls": 0, "phones": 0, "names": 0}


def test_generate_returns_validated_data():
    service, client = make_service([VALID_MATCH_JSON])
    result = service.generate("match_explanation", {})
    assert result["model"] == "primary-model"
    assert result["template_id"] == "match_explanation_v1"
    assert result["data"]["strengths"] == ["Python appears in the resume"]
    assert result["advisory_only"] is True
    assert result["does_not_change_scores"] is True
    assert len(client.models.calls) == 1
    assert client.models.calls[0]["model"] == "primary-model"


def test_generate_falls_back_when_model_is_unavailable():
    service, client = make_service(
        [Exception("404 model not found"), VALID_MATCH_JSON],
        fallback_models=["fallback-model"],
    )
    result = service.generate("match_explanation", {})
    assert result["model"] == "fallback-model"
    assert [call["model"] for call in client.models.calls] == ["primary-model", "fallback-model"]


def test_generate_falls_back_when_model_is_overloaded():
    service, client = make_service(
        [Exception("503 UNAVAILABLE high demand"), VALID_MATCH_JSON],
        fallback_models=["fallback-model"],
    )
    result = service.generate("match_explanation", {})
    assert result["model"] == "fallback-model"
    assert [call["model"] for call in client.models.calls] == ["primary-model", "fallback-model"]


def test_generate_falls_back_on_timeout():
    service, client = make_service(
        [Exception("Request timed out after 45000ms"), VALID_MATCH_JSON],
        fallback_models=["fallback-model"],
    )
    result = service.generate("match_explanation", {})
    assert result["model"] == "fallback-model"
    assert [call["model"] for call in client.models.calls] == ["primary-model", "fallback-model"]


def test_generate_falls_back_on_rate_limit():
    service, client = make_service(
        [Exception("429 RESOURCE_EXHAUSTED quota exceeded"), VALID_MATCH_JSON],
        fallback_models=["fallback-model"],
    )
    result = service.generate("match_explanation", {})
    assert result["model"] == "fallback-model"
    assert [call["model"] for call in client.models.calls] == ["primary-model", "fallback-model"]


def test_generate_does_not_fall_back_on_other_errors():
    service, client = make_service([Exception("500 internal api error"), VALID_MATCH_JSON])
    with pytest.raises(GeminiError) as excinfo:
        service.generate("match_explanation", {})
    assert excinfo.value.code == "api_error"
    assert len(client.models.calls) == 1


def test_rate_limit_error_is_classified():
    service, _ = make_service([Exception("429 RESOURCE_EXHAUSTED quota exceeded")])
    with pytest.raises(GeminiError) as excinfo:
        service.generate("match_explanation", {})
    assert excinfo.value.code == "rate_limit"
    assert excinfo.value.status_code == 429


def test_auth_error_is_classified():
    service, _ = make_service([Exception("401 UNAUTHENTICATED bad api key")])
    with pytest.raises(GeminiError) as excinfo:
        service.generate("match_explanation", {})
    assert excinfo.value.code == "auth"


def test_invalid_json_response_is_rejected():
    service, _ = make_service(["this is not json"])
    with pytest.raises(GeminiError) as excinfo:
        service.generate("match_explanation", {})
    assert excinfo.value.code == "invalid_response"


def test_empty_response_is_rejected():
    service, _ = make_service([""])
    with pytest.raises(GeminiError) as excinfo:
        service.generate("match_explanation", {})
    assert excinfo.value.code == "empty_response"


def test_unknown_task_is_rejected_before_any_call():
    service, client = make_service([VALID_MATCH_JSON])
    with pytest.raises(GeminiError) as excinfo:
        service.generate("make_coffee", {})
    assert excinfo.value.code == "unknown_task"
    assert excinfo.value.status_code == 400
    assert client.models.calls == []


def test_generate_without_configuration_raises_503():
    service = GeminiService(api_key="")
    with pytest.raises(GeminiError) as excinfo:
        service.generate("match_explanation", {})
    assert excinfo.value.code == "not_configured"
    assert excinfo.value.status_code == 503


def test_connection_test_uses_the_configured_model():
    service, client = make_service(["OK"])
    result = service.test_connection()
    assert result["ok"] is True
    assert result["model"] == "primary-model"
    assert result["response_preview"] == "OK"
    assert result["latency_ms"] >= 0
    assert client.models.calls[0]["model"] == "primary-model"
    assert "Reply with the single word OK." in client.models.calls[0]["contents"]


def test_connection_test_without_configuration_raises_503():
    service = GeminiService(api_key="")
    with pytest.raises(GeminiError) as excinfo:
        service.test_connection()
    assert excinfo.value.code == "not_configured"
    assert excinfo.value.status_code == 503
