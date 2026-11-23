"""Lightweight Vercel entrypoint for the resume-job matcher.

This module deliberately imports **no** heavy NLP dependency (torch, spaCy,
sentence-transformers, scikit-learn). It serves the existing dashboard and
forwards every ``/api/*`` call to the separate Python NLP service configured by
``NLP_API_URL``.

The browser still talks to the same origin (``/api/...``), so the existing
frontend, templates, CSS and JavaScript do not need to change and no API token
is ever exposed to the browser.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, render_template, request

from nlp.prompt_templates import TASKS, TASK_LABELS

_ROOT = Path(__file__).resolve().parent
_BACKEND_URL = os.environ.get("NLP_API_URL", "http://127.0.0.1:8000").rstrip("/")
_BACKEND_TOKEN = os.environ.get("NLP_API_TOKEN", "").strip()
_TIMEOUT = float(os.environ.get("NLP_REQUEST_TIMEOUT", "280"))

_MAX_RESUMES = 10
_MAX_MB = 5

_ALLOWED_API_PATHS = frozenset(
    {
        "/api/health",
        "/api/jobs",
        "/api/job-links",
        "/api/match",
        "/api/extract",
        "/api/nlp/pipeline",
        "/api/extract/evaluation",
        "/api/sentiment",
        "/api/sentiment/evaluation",
        "/api/topics/sample",
        "/api/topics",
        "/api/ai/status",
        "/api/ai/test",
        "/api/ai/insights",
        "/api/warmup",
    }
)
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "date",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "server",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

app = Flask(
    __name__,
    template_folder=str(_ROOT / "templates"),
    static_folder=str(_ROOT / "static"),
    static_url_path="/static",
)
# Vercel currently caps request bodies at 4.5 MB (20 MB with the beta body-size
# opt-in). This local cap is intentionally higher so local development is not
# artificially restricted; the platform remains the real limit.
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024


@app.get("/")
def index() -> str:
    """Serve the existing dashboard exactly as the full Flask app does."""
    return render_template(
        "index.html",
        max_resumes=_MAX_RESUMES,
        max_mb=_MAX_MB,
        tasks=TASKS,
        task_labels=TASK_LABELS,
    )


@app.route("/api", defaults={"subpath": ""}, methods=_METHODS)
@app.route("/api/<path:subpath>", methods=_METHODS)
def api_proxy(subpath: str):
    """Forward an allowlisted API call to the external NLP backend."""
    path = "/api" + ("/" + subpath if subpath else "")
    if len(path) > 4:
        path = path.rstrip("/")
    if path not in _ALLOWED_API_PATHS:
        return jsonify(error=f"Unknown API endpoint: {path}"), 404

    url = _BACKEND_URL + path
    if request.query_string:
        url += "?" + request.query_string.decode("latin-1")

    headers = {"Accept": request.headers.get("Accept", "application/json")}
    if _BACKEND_TOKEN:
        headers["X-NLP-API-Token"] = _BACKEND_TOKEN

    files = []
    for key in request.files:
        for upload in request.files.getlist(key):
            files.append(
                (
                    key,
                    (
                        upload.filename or "upload",
                        upload.read(),
                        upload.mimetype or "application/octet-stream",
                    ),
                )
            )

    data = None
    json_body = None
    if files:
        # requests sets the multipart boundary itself; never copy the browser's
        # Content-Type header here.
        headers.pop("Content-Type", None)
        data = {key: values for key, values in request.form.lists()}
    elif request.is_json:
        json_body = request.get_json(silent=True)
    elif request.method in {"POST", "PUT", "PATCH"}:
        data = request.get_data()

    try:
        backend_response = requests.request(
            request.method,
            url,
            headers=headers,
            data=data,
            json=json_body,
            files=files or None,
            timeout=_TIMEOUT,
            allow_redirects=False,
        )
    except requests.exceptions.Timeout:
        return (
            jsonify(
                error=(
                    "The NLP backend took too long to respond. "
                    "Check that the service is healthy and the model cache is warm."
                )
            ),
            504,
        )
    except requests.exceptions.ConnectionError:
        return jsonify(error="The NLP backend is unreachable. Check NLP_API_URL."), 503
    except requests.exceptions.RequestException as exc:
        return jsonify(error=f"Could not reach the NLP backend: {exc}"), 502

    response_headers = {
        key: value
        for key, value in backend_response.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }
    return Response(
        backend_response.content,
        status=backend_response.status_code,
        headers=response_headers,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "3000")), debug=False)
