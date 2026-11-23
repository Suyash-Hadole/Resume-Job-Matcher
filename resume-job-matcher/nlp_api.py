"""Production entrypoint for the external NLP inference service.

Run this with gunicorn on Render/Railway/Fly/etc. It reuses the existing
``app.py`` unchanged, sets a writable model cache before the NLP modules are
imported, and optionally requires the shared secret that the Vercel proxy adds
to every API request.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Set cache locations BEFORE importing app.py / sentence-transformers / spaCy.
_CACHE_ROOT = Path(os.environ.get("MODEL_CACHE_ROOT", str(Path(tempfile.gettempdir()) / "resume-job-matcher-models")))
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_CACHE_ROOT / "sentence-transformers"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_ROOT / "transformers"))
os.environ.setdefault("SPACY_CACHE", str(_CACHE_ROOT / "spacy"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Lazy model loading keeps cold starts fast; call /api/warmup after deploy.
os.environ.setdefault("PRELOAD_MODEL", "0")

from flask import jsonify, request  # noqa: E402

from app import app  # noqa: E402  (heavy stack is intentional on this service)

_EXPECTED_TOKEN = os.environ.get("NLP_API_TOKEN", "").strip()
_PUBLIC_PATHS = frozenset({"/api/health", "/api/warmup"})


@app.before_request
def _require_shared_token():
    """Block direct public access to inference endpoints when a token is set."""
    if not _EXPECTED_TOKEN or request.path.rstrip("/") in _PUBLIC_PATHS:
        return None
    if not request.path.startswith("/api/"):
        return None
    if request.headers.get("X-NLP-API-Token", "") != _EXPECTED_TOKEN:
        return jsonify(error="Unauthorized NLP API request."), 401
    return None


@app.get("/api/warmup")
def _warmup():
    """Load both NLP models so the first real user request is fast."""
    from nlp.embeddings import shared_embedder
    from nlp.preprocessing import nlp_status

    embedder = shared_embedder()
    return jsonify(
        status="ok",
        embeddings={
            "available": embedder.available,
            "backend": embedder.backend,
            "error": embedder.error,
        },
        nlp=nlp_status(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
