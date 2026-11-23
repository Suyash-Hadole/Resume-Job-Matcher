"""End-to-end API tests for the new dashboard features.

The matcher uses a deterministic fake embedder and Gemini is replaced by a fake
service, so these tests never download a model or call the cloud API.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from nlp.gemini_service import GeminiError  # noqa: E402
from nlp.semantic_matcher import ResumeJobMatcher  # noqa: E402
from nlp.topic_modeling import load_sample_corpus  # noqa: E402

JOB = (
    "We need a data analyst with Python, SQL and pandas experience for reporting "
    "and dashboards."
)
RESUME = """Jane Doe
jane.doe@example.com
Data analyst with five years of experience using Python, SQL and pandas to build
dashboards and reporting pipelines for finance teams.
"""
BULLET_RESUME = """Jane Doe
Data analyst with five years of experience using Python, SQL and pandas.
Experience
- Built reporting pipelines in Python that saved the finance team four hours weekly.
- Led migration of dashboards from Excel to Tableau for twelve analysts.
"""


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (no model download)."""

    available = True
    loaded = True
    error = None
    backend = "fake-test-embedder"
    DIM = 512

    def encode(self, texts):
        out = np.zeros((len(texts), self.DIM))
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i, hash(token) % self.DIM] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def similarity(self, resume_texts, job_text):
        vectors = self.encode([job_text, *resume_texts])
        return [float(vector @ vectors[0]) for vector in vectors[1:]]


class FakeGemini:
    def __init__(self, error: GeminiError | None = None):
        self.error = error
        self.calls: list[dict] = []

    def status(self):
        return {
            "provider": "Google Gemini Developer API",
            "sdk": "google-genai",
            "sdk_installed": True,
            "api_key_present": True,
            "configured": True,
            "model": "fake-model",
            "fallback_models": [],
            "cloud_features_enabled": True,
            "consent_required": True,
            "note": "fake service",
        }

    def generate(self, task, context):
        self.calls.append({"task": task, "context": context})
        if self.error:
            raise self.error
        if task == "bullet_suggestions":
            bullets = context.get("bullets") or []
            data = {
                "bullets": [
                    {
                        "original": bullets[0],
                        "improved": bullets[0],
                        "reason": "test rewrite",
                    }
                ],
                "notes": [],
            }
        else:
            data = {
                "summary": "Advisory summary",
                "strengths": ["Python is evidenced"],
                "gaps": ["SQL is not evidenced"],
                "score_interpretation": "Local score reflects partial coverage.",
                "next_steps": ["Add a SQL example if available"],
            }
        return {
            "data": data,
            "model": "fake-model",
            "template_id": f"{task}_v1",
            "advisory_only": True,
            "does_not_change_scores": True,
        }

    def test_connection(self):
        if self.error:
            raise self.error
        return {"ok": True, "model": "fake-model", "latency_ms": 1.0, "response_preview": "OK"}


@pytest.fixture(scope="module")
def matcher():
    return ResumeJobMatcher(embedder=FakeEmbedder())


@pytest.fixture()
def api(matcher):
    gemini = FakeGemini()
    app = create_app(matcher=matcher, gemini_service=gemini)
    app.config["TESTING"] = True
    return app.test_client(), gemini


def ai_payload(**overrides):
    payload = {
        "resume_text": RESUME,
        "filename": "jane.txt",
        "job_description": JOB,
        "job_title": "Data Analyst",
        "task": "match_explanation",
        "consent": True,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# Existing workflow is preserved
# --------------------------------------------------------------------------
def test_index_renders_single_dashboard(api):
    client, _ = api
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Intelligent Resume–Job Matching" in body
    assert "Explore Job Opportunities" in body
    assert "Gemini AI Career Assistant" in body
    # The old five-tab navigation is intentionally gone.
    assert 'data-tab="' not in body
    for old_tab in ("Sentiment Analysis", "Topic Modelling", "Information Extraction"):
        assert old_tab not in body


def test_health_reports_all_features(api):
    client, _ = api
    payload = client.get("/api/health").get_json()
    assert payload["status"] == "ok"
    assert payload["gemini"]["model"] == "fake-model"
    assert "ner_model" in payload["nlp"]
    assert payload["embedding"]["state"] == "loaded"
    assert payload["embedding"]["loaded"] is True
    assert "sentence-transformers" in payload["embedding"]["backend"]
    features = payload["features"]
    assert features["resume_matching"] is True
    assert features["sentiment_analysis"]["available"] is True
    assert features["topic_modelling"]["available"] is True
    assert features["gemini"]["configured"] is True


def test_match_endpoint_still_ranks_resumes(api):
    client, _ = api
    response = client.post(
        "/api/match",
        data={
            "job_description": JOB,
            "job_title": "Data Analyst",
            "resumes": (io.BytesIO(RESUME.encode("utf-8")), "jane.txt"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["rank"] == 1
    assert 0 <= candidate["scores"]["overall"] <= 100
    assert candidate["text_stats"]["words"] == candidate["word_count"]
    assert candidate["text_stats"]["characters"] >= len(RESUME.strip())
    assert "Python" in candidate["extracted_text_preview"] or "python" in candidate["extracted_text_preview"]
    assert len(candidate["extracted_text_preview"]) <= 1500
    # Sentiment is deliberately not part of ranking.
    assert "sentiment" not in json.dumps(payload).lower()


def test_jobs_endpoint_still_lists_sample_jobs(api):
    client, _ = api
    payload = client.get("/api/jobs").get_json()
    assert len(payload["jobs"]) == 5


def test_job_links_endpoint_encodes_title_and_location(api):
    client, _ = api
    payload = client.get(
        "/api/job-links?title=AI%20Engineer&location=Pune%20%2B%20Remote"
    ).get_json()
    assert payload["title"] == "AI Engineer"
    assert payload["location"] == "Pune + Remote"
    sites = [link["site"] for link in payload["links"]]
    assert sites == ["LinkedIn Jobs", "Indeed India", "Naukri.com", "Glassdoor", "Wellfound"]
    urls = [link["url"] for link in payload["links"]]
    assert "keywords=AI+Engineer" in urls[0] and "location=Pune+%2B+Remote" in urls[0]
    assert "q=AI+Engineer" in urls[1] and "l=Pune+%2B+Remote" in urls[1]
    assert "/ai-engineer-jobs" in urls[2] and "query=AI+Engineer" in urls[2]
    assert "sc.keyword=AI+Engineer" in urls[3] and "locName=Pune+%2B+Remote" in urls[3]
    assert "q=AI+Engineer" in urls[4] and "l=Pune+%2B+Remote" in urls[4]
    assert "external job-search results" in payload["disclaimer"]
    assert "does not verify or invent individual vacancies" in payload["disclaimer"]


def test_job_links_endpoint_defaults_to_across_india(api):
    client, _ = api
    payload = client.get("/api/job-links?title=Web%20Developer").get_json()
    assert payload["location"] == "India"
    assert all("India" in link["url"] or "l=India" in link["url"] or "city=India" in link["url"]
               for link in payload["links"])


@pytest.mark.parametrize("query", ["?title=A", "?title=%20%20", "?location=Pune"])
def test_job_links_endpoint_validates_input(api, query):
    client, _ = api
    response = client.get("/api/job-links" + query)
    assert response.status_code == 400
    assert response.get_json()["error"]


# --------------------------------------------------------------------------
# Criterion 1: extraction + pipeline
# --------------------------------------------------------------------------
def test_extract_endpoint_returns_structured_entities(api):
    client, _ = api
    response = client.post("/api/extract", json={"text": RESUME})
    assert response.status_code == 200
    payload = response.get_json()
    entities = payload["entities"]
    assert "jane.doe@example.com" in entities["emails"]
    assert "Jane" in (entities["name"] or "")
    assert any(skill["name"] == "python" for skill in entities["skills"])
    assert payload["skills_by_category"]
    assert payload["text_stats"]["words"] > 10
    assert payload["source_text"].strip() == RESUME.strip()


def test_extract_endpoint_adds_skill_gap_for_a_job(api):
    client, _ = api
    response = client.post("/api/extract", json={"text": RESUME, "job_description": JOB})
    assert response.status_code == 200
    payload = response.get_json()
    assert "skill_gap" in payload
    assert "coverage" in payload["skill_gap"]


def test_extract_endpoint_rejects_empty_input(api):
    client, _ = api
    assert client.post("/api/extract", json={"text": "   "}).status_code == 400


def test_extraction_evaluation_endpoint(api):
    client, _ = api
    response = client.get("/api/extract/evaluation")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["dataset_size"] >= 5
    assert 0 <= payload["skill_metrics"]["f1"] <= 1


def test_nlp_pipeline_endpoint_is_transparent(api):
    client, _ = api
    response = client.post("/api/nlp/pipeline", json={"text": RESUME, "job_description": JOB})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["tokens"]
    assert payload["lemmas"]
    assert payload["vectorization"]["tfidf"]["available"] is True
    assert payload["vectorization"]["embeddings"]["available"] is True
    assert payload["entities"]["skills"]
    assert payload["matching"]["job_word_count"] >= 10


# --------------------------------------------------------------------------
# Criterion 2: sentiment
# --------------------------------------------------------------------------
def test_sentiment_endpoint(api):
    client, _ = api
    response = client.post(
        "/api/sentiment",
        json={"text": "I am delighted with the excellent and wonderful collaboration."},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["label"] == "positive"
    assert set(payload["scores"]) == {"positive", "negative", "neutral", "compound"}
    assert payload["sentences"]


def test_sentiment_endpoint_rejects_empty_text(api):
    client, _ = api
    assert client.post("/api/sentiment", json={"text": ""}).status_code == 400


def test_sentiment_evaluation_endpoint(api):
    client, _ = api
    response = client.get("/api/sentiment/evaluation")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["dataset_size"] >= 30
    assert payload["accuracy"] >= 0.8
    assert len(payload["confusion_matrix"]["matrix"]) == 3


# --------------------------------------------------------------------------
# Criterion 3A: topics
# --------------------------------------------------------------------------
def test_topic_sample_endpoint(api):
    client, _ = api
    payload = client.get("/api/topics/sample").get_json()
    assert payload["description"]
    assert len(payload["documents"]) >= 10


def test_topic_endpoint_models_documents(api):
    client, _ = api
    documents = load_sample_corpus()[:6]
    response = client.post("/api/topics", json={"documents": documents, "n_topics": 3})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["n_topics"] == 3
    assert len(payload["topics"]) == 3
    assert len(payload["documents"]) == len(documents)


def test_topic_endpoint_rejects_a_single_document(api):
    client, _ = api
    response = client.post(
        "/api/topics",
        json={"documents": ["Python and SQL for data analysis and reporting."]},
    )
    assert response.status_code == 400


def test_topic_endpoint_accepts_dash_separated_corpus(api):
    client, _ = api
    corpus = (
        "Python data analysis with pandas and numpy for reporting. "
        "Machine learning models with scikit-learn.\n---\n"
        "React and TypeScript frontend development for web applications.\n---\n"
        "Docker Kubernetes cloud deployment and continuous integration pipelines.\n---\n"
        "SQL database tuning with PostgreSQL indexes and backups."
    )
    response = client.post("/api/topics", json={"corpus": corpus, "n_topics": 2})
    assert response.status_code == 200
    assert response.get_json()["documents_used"] == 4


# --------------------------------------------------------------------------
# Criteria 3B + 4: Gemini assistant
# --------------------------------------------------------------------------
def test_ai_status_endpoint(api):
    client, _ = api
    payload = client.get("/api/ai/status").get_json()
    assert payload["consent_required"] is True
    assert len(payload["tasks"]) == 4
    assert {task["id"] for task in payload["tasks"]} == {
        "match_explanation",
        "resume_improvement",
        "bullet_suggestions",
        "interview_prep",
    }


def test_ai_requires_consent_and_never_calls_gemini(api):
    client, gemini = api
    payload = ai_payload(consent=False)
    response = client.post("/api/ai/insights", json=payload)
    assert response.status_code == 400
    assert response.get_json()["code"] == "consent_required"
    assert gemini.calls == []


def test_ai_rejects_unknown_task_without_calling_gemini(api):
    client, gemini = api
    response = client.post("/api/ai/insights", json=ai_payload(task="do_everything"))
    assert response.status_code == 400
    assert response.get_json()["code"] == "unknown_task"
    assert gemini.calls == []


def test_ai_match_explanation_is_grounded_and_advisory(api):
    client, gemini = api
    response = client.post("/api/ai/insights", json=ai_payload())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["task"] == "match_explanation"
    assert payload["advisory_only"] is True
    assert payload["does_not_change_scores"] is True
    assert payload["analysis"]["scores"]["overall"] >= 0
    assert payload["context"]["redaction"]["emails"] >= 1
    assert payload["context"]["resume_chars_sent"] > 0

    assert len(gemini.calls) == 1
    sent_context = gemini.calls[0]["context"]
    assert "[EMAIL_REDACTED]" in sent_context["resume_excerpt"]
    assert "jane.doe@example.com" not in json.dumps(sent_context)
    assert "scores" in sent_context["local_analysis"]


def test_ai_redacts_job_title_and_description(api):
    client, gemini = api
    response = client.post(
        "/api/ai/insights",
        json=ai_payload(
            job_title="Data Analyst <ignore previous>",
            job_description=(
                "Contact hr@example.com or https://example.com/apply for this Data Analyst role. "
                "Use Python, SQL and pandas for reporting."
            ),
        ),
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["context"]["job_redaction"]["emails"] >= 1
    assert payload["context"]["job_redaction"]["urls"] >= 1
    assert payload["context"]["job_chars_sent"] > 0

    sent = gemini.calls[0]["context"]
    assert "hr@example.com" not in json.dumps(sent)
    assert "https://example.com/apply" not in json.dumps(sent)
    assert "[EMAIL_REDACTED]" in sent["job"]["description"]
    assert "[URL_REDACTED]" in sent["job"]["description"]
    assert "<" not in sent["job"]["title"] and ">" not in sent["job"]["title"]


def test_ai_does_not_change_local_scores(api):
    client, _ = api
    match_response = client.post(
        "/api/match",
        data={
            "job_description": JOB,
            "job_title": "Data Analyst",
            "resumes": (io.BytesIO(RESUME.encode("utf-8")), "jane.txt"),
        },
        content_type="multipart/form-data",
    )
    local_scores = match_response.get_json()["candidates"][0]["scores"]
    ai_response = client.post("/api/ai/insights", json=ai_payload())
    assert ai_response.status_code == 200
    assert ai_response.get_json()["analysis"]["scores"] == local_scores


def test_ai_bullet_suggestions_supply_existing_bullets(api):
    client, gemini = api
    payload = ai_payload(task="bullet_suggestions", resume_text=BULLET_RESUME)
    response = client.post("/api/ai/insights", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["context"]["bullets_supplied"] >= 1
    assert gemini.calls[0]["context"]["bullets"]


def test_ai_bullet_suggestions_require_existing_bullets(api):
    client, gemini = api
    response = client.post(
        "/api/ai/insights", json=ai_payload(task="bullet_suggestions", resume_text=RESUME)
    )
    assert response.status_code == 422
    assert response.get_json()["code"] == "no_bullets"
    assert gemini.calls == []


def test_ai_reports_gemini_errors_cleanly(matcher):
    gemini = FakeGemini(
        error=GeminiError("Gemini is not configured. Set GEMINI_API_KEY.", 503, "not_configured")
    )
    app = create_app(matcher=matcher, gemini_service=gemini)
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post("/api/ai/insights", json=ai_payload())
    assert response.status_code == 503
    assert response.get_json()["code"] == "not_configured"


def test_ai_connection_test_endpoint(api):
    client, _ = api
    response = client.post("/api/ai/test")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
