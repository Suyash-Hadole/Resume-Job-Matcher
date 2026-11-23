"""Tests for the lightweight Vercel proxy/entrypoint."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import requests

import vercel_proxy


class FakeBackendResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self.content = payload
        self.status_code = status
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(payload)),
            "Connection": "keep-alive",
            "X-Backend": "nlp-service",
        }


@pytest.fixture()
def client():
    vercel_proxy.app.config["TESTING"] = True
    with vercel_proxy.app.test_client() as test_client:
        yield test_client


def test_index_uses_existing_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="fileInput"' in html
    assert 'id="analyzeBtn"' in html
    assert 'id="jobLinksCard"' in html


def test_static_assets_are_served(client):
    css = client.get("/static/css/style.css")
    js = client.get("/static/js/main.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert b"body" in css.data.lower()
    assert b"apiFetch" in js.data


def test_unknown_api_path_is_blocked(client):
    response = client.get("/api/secret")
    assert response.status_code == 404
    assert "Unknown API endpoint" in response.get_json()["error"]


def test_get_request_is_forwarded_with_token_and_query(client, monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeBackendResponse(b'{"status":"ok"}')

    monkeypatch.setattr(vercel_proxy, "_BACKEND_URL", "https://nlp.example.com")
    monkeypatch.setattr(vercel_proxy, "_BACKEND_TOKEN", "top-secret")
    monkeypatch.setattr(vercel_proxy.requests, "request", fake_request)

    response = client.get("/api/health?verbose=1")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert captured["method"] == "GET"
    assert captured["url"] == "https://nlp.example.com/api/health?verbose=1"
    assert captured["headers"]["X-NLP-API-Token"] == "top-secret"
    assert response.headers["X-Backend"] == "nlp-service"
    assert "Connection" not in response.headers


def test_multipart_upload_is_forwarded(client, monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return FakeBackendResponse(b'{"ok":true}')

    monkeypatch.setattr(vercel_proxy.requests, "request", fake_request)

    response = client.post(
        "/api/match",
        data={
            "job_title": "Data Scientist",
            "resumes": (io.BytesIO(b"%PDF-1.4 test"), "cv.pdf", "application/pdf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/match")
    assert captured["data"] == {"job_title": ["Data Scientist"]}
    assert captured["files"][0][0] == "resumes"
    filename, payload, mimetype = captured["files"][0][1]
    assert filename == "cv.pdf"
    assert payload == b"%PDF-1.4 test"
    assert mimetype == "application/pdf"
    assert "Content-Type" not in captured["headers"]


def test_connection_error_returns_clear_503(client, monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(vercel_proxy.requests, "request", raise_connection_error)
    response = client.get("/api/health")
    assert response.status_code == 503
    assert "unreachable" in response.get_json()["error"]


def test_vercel_entrypoint_does_not_import_heavy_nlp_stack():
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys
import vercel_proxy
heavy = [m for m in (
    'torch', 'spacy', 'sentence_transformers', 'sklearn',
    'nlp.embeddings', 'nlp.preprocessing', 'app'
) if m in sys.modules]
print(json.dumps(heavy))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.strip()) == []
