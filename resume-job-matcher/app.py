"""Flask application for the Intelligent Resume-Job Matching System.

Local NLP features (matching, extraction, sentiment, topic modelling) run
entirely on the server. The optional Gemini career assistant only runs when a
user explicitly consents to sending a redacted excerpt to Google.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import quote_plus

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from nlp.embeddings import MODEL_NAME
from nlp.gemini_service import (
    MAX_JOB_CONTEXT_CHARS,
    MAX_RESUME_CONTEXT_CHARS,
    GeminiError,
    GeminiService,
    redact_text,
)
from nlp.ie_evaluation import IEEvaluationError, evaluate_extraction
from nlp.ner_extractor import EntityExtractor
from nlp.pipeline_demo import PipelineDemoError, run_pipeline_demo
from nlp.preprocessing import nlp_status
from nlp.prompt_templates import (
    TASK_LABELS,
    TASKS,
    ResponseValidationError,
    build_ai_context,
    extract_bullets,
)
from nlp.resume_parser import MAX_FILE_BYTES, ResumeParseError, extract_text
from nlp.semantic_matcher import MatchError, Resume, ResumeJobMatcher, load_sample_jobs
from nlp.sentiment_analyzer import SentimentAnalyzer, SentimentError, sentiment_status
from nlp.topic_modeling import (
    TopicModelError,
    corpus_description,
    load_sample_corpus,
    model_topics,
    topic_status,
)

MAX_RESUMES = 10
MIN_JOB_WORDS = 10
MAX_JOB_CHARS = 20_000
MAX_SENTIMENT_CHARS = 20_000
MAX_EXTRACT_CHARS = 50_000
MAX_TOPIC_DOCS = 50
MAX_TOPIC_DOC_CHARS = 20_000
MAX_TOPIC_TOTAL_CHARS = 300_000
DEFAULT_TOPICS = 5

log = logging.getLogger("resume_matcher")


class InputError(ValueError):
    """A validation failure with an HTTP status code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _error(message: str, status: int, **extra):
    payload = {"error": message}
    payload.update(extra)
    return jsonify(payload), status


def _json_body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _slugify(value: str) -> str:
    """Turn a job title or location into a URL-safe hyphenated slug."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _read_text_payload(max_chars: int = MAX_EXTRACT_CHARS, file_fields=("resume", "file", "resumes")):
    """Read text from an uploaded file or a JSON/form ``text`` field."""
    for field in file_fields:
        storage = request.files.get(field)
        if storage and storage.filename:
            text = extract_text(os.path.basename(storage.filename), storage.read())
            return text, os.path.basename(storage.filename)
        if field == "resumes":
            storages = [f for f in request.files.getlist(field) if f and f.filename]
            if storages:
                storage = storages[0]
                text = extract_text(os.path.basename(storage.filename), storage.read())
                return text, os.path.basename(storage.filename)

    payload = _json_body()
    text = payload.get("text") or request.form.get("text") or ""
    if not str(text).strip():
        raise InputError("Provide a resume file or paste some text.")
    text = str(text).strip()
    if len(text) > max_chars:
        raise InputError(f"Text is too long (maximum {max_chars:,} characters).")
    return text, None


def _resolve_job(job_id: str, custom: str, job_title: str, jobs_by_id: dict) -> tuple[str, str]:
    if custom:
        job_text = custom
        title = job_title.strip() or "Custom job description"
    elif job_id:
        job = jobs_by_id.get(job_id)
        if job is None:
            raise InputError(f"Unknown job_id '{job_id}'.")
        job_text, title = job["description"], job["title"]
    else:
        raise InputError("Select a sample job or paste a job description.")
    if len(job_text) > MAX_JOB_CHARS:
        raise InputError(f"Job description is too long (max {MAX_JOB_CHARS:,} characters).")
    if len(job_text.split()) < MIN_JOB_WORDS:
        raise InputError(f"Job description is too short (need at least {MIN_JOB_WORDS} words).")
    return job_text, title


def create_app(
    matcher: ResumeJobMatcher | None = None,
    gemini_service: GeminiService | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_RESUMES * MAX_FILE_BYTES + 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    app.extensions["matcher_instance"] = matcher
    app.extensions["gemini_service"] = gemini_service or GeminiService()
    jobs = load_sample_jobs()
    jobs_by_id = {j["id"]: j for j in jobs}

    def get_matcher() -> ResumeJobMatcher:
        if app.extensions.get("matcher_instance") is None:
            app.extensions["matcher_instance"] = ResumeJobMatcher()
        return app.extensions["matcher_instance"]

    def get_gemini() -> GeminiService:
        return app.extensions["gemini_service"]

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            max_resumes=MAX_RESUMES,
            max_mb=MAX_FILE_BYTES // 1024 // 1024,
            tasks=TASKS,
            task_labels=TASK_LABELS,
        )

    @app.get("/api/health")
    def health():
        inst = app.extensions.get("matcher_instance")
        embedder = getattr(inst, "embedder", None) if inst is not None else None
        embedder_loaded = bool(embedder and getattr(embedder, "loaded", False))
        embedder_error = getattr(embedder, "error", None) if embedder else None
        model_name = getattr(embedder, "model_name", MODEL_NAME) if embedder else MODEL_NAME

        if embedder_loaded:
            embedding_state = "loaded"
            embedding_backend = f"sentence-transformers / {model_name}"
        elif embedder_error:
            embedding_state = "fallback"
            embedding_backend = "tfidf-fallback (sentence-transformers model unavailable)"
        else:
            embedding_state = "lazy" if inst is not None else "idle"
            embedding_backend = f"{model_name} (loads on first analysis)"

        gem = get_gemini().status()
        nlp = nlp_status()
        return jsonify(
            status="ok",
            embedding_model=MODEL_NAME,
            embedding_model_loaded=embedder_loaded,
            embedding={
                "model": model_name,
                "state": embedding_state,
                "loaded": embedder_loaded,
                "backend": embedding_backend,
                "error": embedder_error,
            },
            nlp=nlp,
            gemini=gem,
            features={
                "resume_matching": True,
                "information_extraction": {"available": nlp["available"]},
                "sentiment_analysis": sentiment_status(),
                "topic_modelling": topic_status(),
                "gemini": {
                    "configured": gem["configured"],
                    "model": gem["model"],
                    "cloud_features_enabled": gem["cloud_features_enabled"],
                },
            },
        )

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify(jobs=jobs)

    @app.get("/api/job-links")
    def job_links():
        """Build external job-search URLs; no vacancies are fetched or claimed."""
        title = (request.args.get("title") or "").strip()
        location = (request.args.get("location") or "").strip() or "India"
        if len(title) < 2 or len(title) > 120:
            return _error("Enter a job title between 2 and 120 characters.", 400)
        if len(location) > 120:
            return _error("Location is too long (maximum 120 characters).", 400)

        q = quote_plus(title)
        loc = quote_plus(location)
        title_slug = _slugify(title) or "jobs"

        links = [
            {
                "site": "LinkedIn Jobs",
                "url": f"https://www.linkedin.com/jobs/search/?keywords={q}&location={loc}",
                "hint": "Professional network search",
            },
            {
                "site": "Indeed India",
                "url": f"https://in.indeed.com/jobs?q={q}&l={loc}",
                "hint": "India job board search",
            },
            {
                "site": "Naukri.com",
                "url": f"https://www.naukri.com/{title_slug}-jobs?query={q}&city={loc}",
                "hint": "India-focused role search",
            },
            {
                "site": "Glassdoor",
                "url": f"https://www.glassdoor.co.in/Job/index.htm?sc.keyword={q}&locT=C&locName={loc}",
                "hint": "Jobs with company reviews",
            },
            {
                "site": "Wellfound",
                "url": f"https://wellfound.com/jobs?q={q}&l={loc}",
                "hint": "Startup and remote roles",
            },
        ]
        return jsonify(
            title=title,
            location=location,
            links=links,
            disclaimer=(
                "These are external job-search results. Each link opens the platform's "
                "search page for this title and location; this app does not verify or "
                "invent individual vacancies."
            ),
        )

    # ------------------------------------------------------------------
    # Existing resume-matching workflow (preserved)
    # ------------------------------------------------------------------
    @app.post("/api/match")
    def match():
        job_id = (request.form.get("job_id") or "").strip()
        custom = (request.form.get("job_description") or "").strip()
        files = [f for f in request.files.getlist("resumes") if f and f.filename]

        try:
            job_text, job_title = _resolve_job(
                job_id, custom, request.form.get("job_title", ""), jobs_by_id
            )
        except InputError as exc:
            return _error(exc.message, exc.status)

        if not files:
            return _error("Upload at least one resume (PDF, DOCX or TXT).", 400)
        if len(files) > MAX_RESUMES:
            return _error(f"Too many files. Maximum is {MAX_RESUMES} resumes per request.", 400)

        resumes, errors, seen = [], [], set()
        for storage in files:
            name = os.path.basename(storage.filename)
            if name.lower() in seen:
                errors.append({"filename": name, "error": "Duplicate file name in this upload."})
                continue
            seen.add(name.lower())
            try:
                text = extract_text(name, storage.read())
                resumes.append(Resume(name=name, text=text))
            except ResumeParseError as exc:
                errors.append({"filename": name, "error": str(exc)})

        if not resumes:
            return jsonify(error="None of the uploaded resumes could be processed.", file_errors=errors), 422

        try:
            result = get_matcher().rank(resumes, job_text, job_title=job_title)
        except MatchError as exc:
            return _error(str(exc), 400)
        except Exception:  # noqa: BLE001
            log.exception("Matching failed")
            return _error("Unexpected error while matching resumes. Check the server log.", 500)

        result["file_errors"] = errors
        return jsonify(result)

    # ------------------------------------------------------------------
    # Criterion 1: information extraction + transparent pipeline demo
    # ------------------------------------------------------------------
    @app.post("/api/extract")
    def extract():
        try:
            text, filename = _read_text_payload(MAX_EXTRACT_CHARS)
            extractor = EntityExtractor()
            entities = extractor.extract(text)
            payload = _json_body()
            job_text = (payload.get("job_description") or request.form.get("job_description") or "").strip()
            response = {
                "source": filename or "pasted text",
                "source_text": text,
                "text_stats": {
                    "characters": len(text),
                    "words": len(text.split()),
                },
                "entities": entities,
                "skills_by_category": entities["skills_by_category"],
                "ner_backend": entities["ner_backend"],
                "ner_model_loaded": entities["ner_model_loaded"],
            }
            if job_text and len(job_text.split()) >= MIN_JOB_WORDS:
                from nlp.skill_gap import analyze_skill_gap

                job_entities = extractor.extract(job_text)
                response["skill_gap"] = analyze_skill_gap(
                    {s["name"]: s for s in entities["skills"]},
                    {s["name"]: s for s in job_entities["skills"]},
                )
            return jsonify(response)
        except InputError as exc:
            return _error(exc.message, exc.status)
        except ResumeParseError as exc:
            return _error(str(exc), 422)
        except Exception:  # noqa: BLE001
            log.exception("Information extraction failed")
            return _error("Unexpected error during information extraction.", 500)

    @app.post("/api/nlp/pipeline")
    def pipeline():
        try:
            text, filename = _read_text_payload(MAX_EXTRACT_CHARS)
            payload = _json_body()
            job_text = (payload.get("job_description") or request.form.get("job_description") or "").strip()
            inst = get_matcher()
            result = run_pipeline_demo(
                text,
                job_text=job_text,
                embedder=inst.embedder,
                extractor=inst.entities,
            )
            result["source"] = filename or "pasted text"
            return jsonify(result)
        except InputError as exc:
            return _error(exc.message, exc.status)
        except (PipelineDemoError, ResumeParseError) as exc:
            return _error(str(exc), 422 if isinstance(exc, ResumeParseError) else 400)
        except Exception:  # noqa: BLE001
            log.exception("NLP pipeline demo failed")
            return _error("Unexpected error while running the NLP demonstration.", 500)

    @app.get("/api/extract/evaluation")
    def extraction_evaluation():
        try:
            return jsonify(evaluate_extraction())
        except IEEvaluationError as exc:
            return _error(str(exc), 500)
        except Exception:  # noqa: BLE001
            log.exception("Extraction evaluation failed")
            return _error("Unexpected error while evaluating information extraction.", 500)

    # ------------------------------------------------------------------
    # Criterion 2: sentiment analysis
    # ------------------------------------------------------------------
    @app.post("/api/sentiment")
    def sentiment():
        payload = _json_body()
        text = payload.get("text") or request.form.get("text") or ""
        try:
            if not str(text).strip():
                raise InputError("Enter some text to analyse.")
            if len(str(text)) > MAX_SENTIMENT_CHARS:
                raise InputError(f"Text is too long (maximum {MAX_SENTIMENT_CHARS:,} characters).")
            result = SentimentAnalyzer().analyze(str(text))
            result["text_preview"] = str(text).strip()[:500]
            return jsonify(result)
        except (InputError, SentimentError) as exc:
            status = exc.status if isinstance(exc, InputError) else 400
            return _error(str(exc), status)
        except Exception:  # noqa: BLE001
            log.exception("Sentiment analysis failed")
            return _error("Unexpected error during sentiment analysis.", 500)

    @app.get("/api/sentiment/evaluation")
    def sentiment_evaluation():
        try:
            return jsonify(SentimentAnalyzer().evaluate())
        except SentimentError as exc:
            return _error(str(exc), 500)
        except Exception:  # noqa: BLE001
            log.exception("Sentiment evaluation failed")
            return _error("Unexpected error while evaluating sentiment analysis.", 500)

    # ------------------------------------------------------------------
    # Criterion 3A: topic modelling
    # ------------------------------------------------------------------
    @app.get("/api/topics/sample")
    def topic_sample():
        try:
            return jsonify(
                description=corpus_description(),
                documents=load_sample_corpus(),
            )
        except TopicModelError as exc:
            return _error(str(exc), 500)

    @app.post("/api/topics")
    def topics():
        payload = _json_body()
        documents = payload.get("documents")
        if documents is None:
            corpus = payload.get("corpus") or request.form.get("corpus") or ""
            if not str(corpus).strip():
                return _error("Provide a list of documents or a corpus separated by '---' lines.", 400)
            documents = [part.strip() for part in re.split(r"(?m)^\s*-{3,}\s*$", str(corpus)) if part.strip()]
        if isinstance(documents, str):
            documents = [documents]
        if not isinstance(documents, list):
            return _error("'documents' must be a list of texts or {title, text} objects.", 400)
        if len(documents) > MAX_TOPIC_DOCS:
            return _error(f"Too many documents (maximum {MAX_TOPIC_DOCS}).", 400)

        for index, doc in enumerate(documents):
            text = doc if isinstance(doc, str) else (doc.get("text", "") if isinstance(doc, dict) else "")
            if len(str(text)) > MAX_TOPIC_DOC_CHARS:
                return _error(
                    f"Document {index + 1} is too long (maximum {MAX_TOPIC_DOC_CHARS:,} characters).",
                    400,
                )
        total_chars = sum(
            len(doc if isinstance(doc, str) else str(doc.get("text", "")))
            for doc in documents
            if isinstance(doc, (str, dict))
        )
        if total_chars > MAX_TOPIC_TOTAL_CHARS:
            return _error(f"Corpus is too large (maximum {MAX_TOPIC_TOTAL_CHARS:,} characters).", 400)

        try:
            n_topics = int(payload.get("n_topics") or request.form.get("n_topics") or DEFAULT_TOPICS)
        except (TypeError, ValueError):
            return _error("'n_topics' must be an integer.", 400)

        try:
            return jsonify(model_topics(documents, n_topics=n_topics))
        except TopicModelError as exc:
            return _error(str(exc), 400)
        except Exception:  # noqa: BLE001
            log.exception("Topic modelling failed")
            return _error("Unexpected error during topic modelling.", 500)

    # ------------------------------------------------------------------
    # Criteria 3B + 4: Gemini career assistant
    # ------------------------------------------------------------------
    @app.get("/api/ai/status")
    def ai_status():
        return jsonify(
            gemini=get_gemini().status(),
            tasks=[{"id": task, "label": TASK_LABELS[task]} for task in TASKS],
            consent_required=True,
            disclaimer=(
                "AI output is advisory only. It never changes the local similarity scores "
                "and must not be used as an automated hiring decision."
            ),
        )

    @app.post("/api/ai/test")
    def ai_test():
        try:
            return jsonify(get_gemini().test_connection())
        except GeminiError as exc:
            return _error(exc.message, exc.status_code, code=exc.code)

    @app.post("/api/ai/insights")
    def ai_insights():
        payload = _json_body()
        consent_raw = payload.get("consent", request.form.get("consent"))
        consent = str(consent_raw).strip().lower() in {"true", "1", "yes", "on"}
        if not consent:
            return _error(
                "Explicit consent is required before any text is sent to Gemini. "
                "Tick the consent checkbox and try again.",
                400,
                code="consent_required",
            )

        task = str(payload.get("task") or request.form.get("task") or "").strip()
        if task not in TASKS:
            return _error(
                f"Unknown AI task. Choose one of: {', '.join(TASKS)}.",
                400,
                code="unknown_task",
            )

        job_id = str(payload.get("job_id") or request.form.get("job_id") or "").strip()
        custom = str(payload.get("job_description") or request.form.get("job_description") or "").strip()
        job_title_raw = str(payload.get("job_title") or request.form.get("job_title") or "")
        try:
            job_text, job_title = _resolve_job(job_id, custom, job_title_raw, jobs_by_id)
        except InputError as exc:
            return _error(exc.message, exc.status)

        # Resume text can come from a file upload or, for API clients/tests, JSON.
        filename = "resume"
        try:
            upload = None
            for field in ("resume", "file"):
                candidate = request.files.get(field)
                if candidate and candidate.filename:
                    upload = candidate
                    break
            if upload is not None:
                filename = os.path.basename(upload.filename)
                resume_text = extract_text(filename, upload.read())
            else:
                resume_text = str(payload.get("resume_text") or "").strip()
                filename = str(payload.get("filename") or "pasted resume")
                if not resume_text:
                    raise InputError("Provide a resume file (or resume_text) for AI analysis.")
                if len(resume_text) > MAX_EXTRACT_CHARS:
                    raise InputError(f"Resume text is too long (maximum {MAX_EXTRACT_CHARS:,} characters).")
                if len(resume_text.split()) < 20:
                    raise InputError("Resume text is too short for a meaningful AI analysis.")
        except ResumeParseError as exc:
            return _error(str(exc), 422)
        except InputError as exc:
            return _error(exc.message, exc.status)

        try:
            analysis = get_matcher().rank([Resume(filename, resume_text)], job_text, job_title=job_title)
        except MatchError as exc:
            return _error(str(exc), 400)
        except Exception:  # noqa: BLE001
            log.exception("AI local analysis failed")
            return _error("Unexpected error while analysing the resume locally.", 500)

        candidate = analysis["candidates"][0]
        entities = candidate.get("entities") or {}
        names = [entities.get("name"), *(entities.get("persons") or [])]
        redacted, redaction = redact_text(resume_text, names=[n for n in names if n])
        truncated = len(redacted) > MAX_RESUME_CONTEXT_CHARS
        redacted = redacted[:MAX_RESUME_CONTEXT_CHARS]

        # The job title and description are untrusted user text. Redact direct
        # identifiers from both before they become part of any cloud prompt.
        redacted_title, title_redaction = redact_text(job_title, names=[])
        redacted_job, job_redaction = redact_text(job_text, names=[])
        job_redaction = {
            key: job_redaction.get(key, 0) + title_redaction.get(key, 0)
            for key in ("emails", "urls", "phones", "names")
        }
        job_for_prompt = {
            "title": redacted_title,
            "description": redacted_job[:MAX_JOB_CONTEXT_CHARS],
            "skills": analysis["job"]["skills"],
        }
        context = build_ai_context(job_for_prompt, candidate, redacted, redaction)
        context["job_redaction"] = job_redaction
        if task == "bullet_suggestions":
            bullets = extract_bullets(redacted)
            if not bullets:
                return _error(
                    "No existing bullet points were found in the resume. "
                    "Add bullet-point achievements and try again.",
                    422,
                    code="no_bullets",
                )
            context["bullets"] = bullets

        try:
            generated = get_gemini().generate(task, context)
        except GeminiError as exc:
            return _error(exc.message, exc.status_code, code=exc.code)
        except ResponseValidationError as exc:
            return _error(f"Invalid AI response: {exc}", 502, code="invalid_response")
        except Exception:  # noqa: BLE001
            log.exception("Gemini request failed")
            return _error("Unexpected error while contacting Gemini.", 500)

        return jsonify(
            {
                "task": task,
                "label": TASK_LABELS[task],
                "model": generated.get("model"),
                "template_id": generated.get("template_id"),
                "result": generated.get("data"),
                "analysis": {
                    "scores": candidate["scores"],
                    "verdict": candidate["verdict"],
                    "skill_coverage": candidate["skill_gap"]["coverage"],
                    "matching_skills": [m["display"] for m in candidate["skill_gap"]["matching"]],
                    "missing_skills": [m["display"] for m in candidate["skill_gap"]["missing"]],
                },
                "context": {
                    "redaction": redaction,
                    "job_redaction": job_redaction,
                    "resume_chars_sent": len(redacted),
                    "job_chars_sent": len(job_for_prompt["description"]),
                    "resume_excerpt_truncated": truncated,
                    "job_title": redacted_title,
                    "bullets_supplied": len(context.get("bullets", [])),
                },
                "consent": True,
                "advisory_only": True,
                "does_not_change_scores": True,
                "disclaimer": (
                    "Gemini output is advisory and does not modify the local NLP scores. "
                    "It must not be used as an automated hiring decision."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return _error("Upload too large. Reduce the number or size of resumes.", 413)

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            return _error(exc.description or exc.name, exc.code or 500)
        return exc

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        log.exception("Unhandled error")
        if request.path.startswith("/api/"):
            return _error("Internal server error.", 500)
        return "Internal server error", 500

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("PRELOAD_MODEL", "1") == "1":
        log.info("Preloading embedding model (first run downloads ~90 MB)...")
        inst = ResumeJobMatcher()
        app.extensions["matcher_instance"] = inst
        log.info("Embedding backend: %s", inst.embedder.backend)
        if inst.embedder.error:
            log.warning("Embedding model unavailable: %s", inst.embedder.error)
    gem_status = app.extensions["gemini_service"].status()
    log.info(
        "Gemini configured: %s (model: %s)",
        gem_status["configured"],
        gem_status["model"],
    )
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
