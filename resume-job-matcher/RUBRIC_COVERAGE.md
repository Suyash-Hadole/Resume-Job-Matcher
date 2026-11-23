# Rubric Coverage — Intelligent Resume–Job Matching System

This map is based on the **actual executed application, notebooks and tests**, not on file names or UI labels.

Final verification date: 2026-11-23

| Assessment | Marks | Implemented modules | Demonstration evidence | Status |
|---|---:|---|---|---|
| **CO1 — Text extraction, preprocessing, cleaning and vectorization** | 7 | `app.py` (`/api/match`, `/api/extract`, `/api/nlp/pipeline`), `nlp/preprocessing.py`, `nlp/embeddings.py`, `nlp/semantic_matcher.py`, `nlp/ner_extractor.py` | PDF, DOCX and TXT extraction verified by pytest. `/api/nlp/pipeline` returns original text, cleaned text, cleaning steps, tokens with POS/lemma/stop flags, lemmas, stopwords, embeddings, TF-IDF and cosine. `/api/match` returned semantic 74.0, TF-IDF 46.2 and skill coverage 83.3 for the live sample. `/api/health` reports `all-MiniLM-L6-v2` loaded. Intermediate results are displayed in the dashboard. | **Completed** |
| **CO2 — Information extraction and sentiment analysis** | 8 | `nlp/ner_extractor.py`, `nlp/preprocessing.py`, `/api/extract`, `/api/extract/evaluation`; `notebooks/sentiment_analysis.ipynb` | NER returned name, email, phone, URL, education, organization, dates and 11 technical skills from the uploaded sample, with a structured table in the report. Sentiment notebook executed with **0 errors**, 23 saved outputs and 2 images; accuracy **0.917**, macro-F1 **0.916**, per-class precision/recall/F1, confusion matrix and interpretation. Sentiment is **not** a dashboard tab. | **Completed** |
| **CO3 — Topic modelling and cloud-based AI services** | 7 | `notebooks/topic_modelling.ipynb`, `nlp/topic_modeling.py`, `nlp/gemini_service.py`, `.env`, `/api/ai/status`, `/api/ai/test` | LDA notebook executed with **0 errors**, 16 saved outputs and 2 images; CountVectorizer matrix, **4 topics**, perplexity **1063.19**, keywords, document-topic heatmap and interpretation. Gemini uses official `google-genai` SDK, key from environment/`.env`, connection test endpoint, secure error classification, 45-second timeout and fallback chain. Live `/api/ai/test` returned HTTP 200 via fallback after the primary model was overloaded/quota-limited. Local NLP works without credentials. | **Completed** |
| **CO4 — Generative AI, prompt engineering and integration** | 8 | `nlp/prompt_templates.py`, `nlp/gemini_service.py`, `app.py` (`/api/ai/insights`), `templates/index.html`, `static/js/main.js` | Four tasks implemented: match explanation, resume improvement, bullet suggestions and interview preparation. Prompt templates are versioned and structured (`*_v1`). Every request requires explicit consent, redacts PII from **both the resume and the job title/description**, neutralises untrusted text, limits context, strictly validates JSON against a schema and returns advisory-only output. Live consented tasks returned HTTP 200 with schema-valid results. Browser test confirmed unticked consent blocks the request. Gemini output never changes the local score payload. | **Completed** |

---

## Gap analysis and resolution

| Area | Initial risk / gap observed | Resolution | Status |
|---|---|---|---|
| Long Gemini call | SDK could hang and leave the browser waiting until client timeout | Added `HttpOptions(timeout=45_000)` and fallback on timeout | Fixed |
| `503 UNAVAILABLE` | High-demand model errors were classified as generic API errors | Classified as `model_unavailable` and added documented fallback models | Fixed |
| `429 RESOURCE_EXHAUSTED` | Free-tier quota is **per model**; stopping on 429 broke the primary model | 429 now falls back to another model; final error is clear if all are exhausted | Fixed |
| Documentation counts | Older docs reported 155/156 tests | Updated to the verified **164 passed** | Fixed |
| Rubric mapping file | `RUBRIC_MAPPING.md` existed but no dedicated coverage document | Added this `RUBRIC_COVERAGE.md` | Fixed |
| Sentiment / LDA as tabs | Potential mismatch with the “not a separate tab” requirement | Browser and HTML checks confirm only one matching dashboard; notebooks are separate | Verified |
| API key exposure | `.env.example` contained a live-looking key identical to `.env` | Replaced with a placeholder; repo scan found no key outside ignored `.env` | Fixed |
| Resume-only redaction | Job title and job description were sent without PII redaction | Both are redacted before prompting; `job_redaction` and `job_chars_sent` are returned in the response context | Fixed |
| Prompt injection via job title | Job title was inserted un-neutralised into prompts | `_neutralise()` now applies to the title as well as the description | Fixed |
| Soft response validation | Unknown top-level fields accepted, invalid list items silently dropped, `answer_hints` optional, invented bullet facts not fully checked | Strict validation rejects unknown fields, invalid items, missing hints and new numbers/non-trivial tokens in rewritten bullets | Fixed |
| Configured vs SDK status | `configured` required the optional SDK, so a key-only environment looked unconfigured | `configured` now means key present; `cloud_features_enabled` additionally requires the SDK | Fixed |
| IE benchmark wording | Perfect 1.0 on 10 self-consistent cases could be mistaken for general accuracy | Documented as a smoke/wiring benchmark, not an accuracy claim | Fixed |

No assessment criterion is currently partially implemented or missing.

---

## Verification commands and observed output

```powershell
# Full automated suite
.\.venv\Scripts\python.exe -m pytest -q
# 164 passed

# Syntax checks
.\.venv\Scripts\python.exe -m compileall -q app.py nlp tests
node --check static/js/main.js

# Notebook execution
.\.venv\Scripts\python.exe scripts/verify_notebooks.py
# sentiment_analysis.ipynb: executed, 8 code cells, 0 errors
# topic_modelling.ipynb: executed, 9 code cells, 0 errors

# Live endpoint audit (with the Flask app running on port 5055)
.\.venv\Scripts\python.exe scripts/verify_endpoints.py
# 31/31 passed
```

### Live endpoint evidence

- `GET /` → 200, single dashboard, no old five-tab navigation or sentiment/topic tabs.
- `GET /api/health` → spaCy NER ready, MiniLM loaded, Gemini configured.
- `GET /api/jobs` → 5 curated sample jobs.
- `GET /api/job-links` → 5 encoded external searches for LinkedIn, Indeed, Naukri, Glassdoor and Wellfound.
- `POST /api/match` → ranked candidate with semantic, TF-IDF, skill coverage, NER, skill gap and extracted-text preview.
- `POST /api/extract` → structured entities and skill categories.
- `POST /api/nlp/pipeline` → tokens, lemmas, cleaned text, embeddings, TF-IDF and cosine.
- `GET /api/sentiment/evaluation` → real saved evaluation metrics.
- `POST /api/topics` → LDA topics with keywords.
- `GET /api/ai/status` → Gemini configured and four tasks exposed.

### Browser evidence

- Loaded sample resume, entered job title and location, clicked **Analyze & rank**.
- Report rendered with score cards, NER table, skill categories, suggestions, external job links and the optional Gemini card.
- Five genuine external search URLs opened with encoded title/location.
- Clicking **Generate** without consent displayed: “Explicit consent is required before sending any text to Gemini.”

---

## Score integrity

The local score is computed only from:

- Semantic similarity — 50%
- Skill coverage — 35%
- TF-IDF baseline — 15%

Sentiment and Gemini are explicitly excluded. Responses from Gemini include `advisory_only=true` and `does_not_change_scores=true`, and the frontend displays the same disclaimer.

---

## Final status

**All four criteria (CO1–CO4) are fully implemented, executed and demonstrable — 30/30 target coverage.**
