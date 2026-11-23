# Implementation and Verification Report

Project: `D:\resume-job-matcher`
Date: 2026-11-22

## Scope

The existing Flask application was **restructured in place**. The blue/white design, NLP modules, scoring algorithm and API routes were preserved. The five-tab navigation was removed from the UI and replaced with one focused Resume-Job Matching dashboard. Sentiment analysis and topic modelling were moved into separate, executed Jupyter notebooks. Gemini remains optional and now lives inside the matching results page.

No project rebuild was performed.

---

## Fully implemented

| Area | Status | Evidence |
|---|---|---|
| Single-dashboard UI | Fully implemented | `templates/index.html` contains one workspace with role fields, upload, report, job links and Gemini card. No `data-tab` navigation remains. |
| Role inputs and location | Fully implemented | Free-text job title and description, sample role selector, any location plus quick chips for Pune/Mumbai/Bengaluru/Remote/Across India. |
| Resume parsing | Fully implemented | PDF, DOCX and TXT extraction with validation; multi-file drag-and-drop; real sample resume button. |
| NER and extraction in results | Fully implemented | Every candidate card shows name/contact/education/experience/organisation/location/date table plus extracted-text preview and normalised skills. |
| Semantic matching | Fully implemented | MiniLM sentence embeddings with chunking, cosine similarity, TF-IDF baseline, skill coverage and transparent raw/calibrated scores. |
| Skill-gap analysis | Fully implemented | Matched, missing (priority flagged) and extra skills, category coverage bars and grounded recommendations. |
| Candidate ranking | Fully implemented | Deterministic sort by overall score with ranks, verdicts and visible score components. |
| External job-search links | Fully implemented | `GET /api/job-links` builds five encoded search URLs (LinkedIn, Indeed India, Naukri, Glassdoor, Wellfound), opens new tabs and shows an external-search disclaimer. No invented vacancies. |
| Sentiment notebook | Fully implemented | Executed `notebooks/sentiment_analysis.ipynb` with embedded 36-case dataset, preprocessing, VADER, charts, accuracy 0.917, macro-F1 0.916, per-class metrics and confusion matrix. |
| Topic notebook | Fully implemented | Executed `notebooks/topic_modelling.ipynb` with embedded 15-document corpus, preprocessing, CountVectorizer, four-topic LDA, keyword charts, document-topic heatmap and perplexity 1063.19. |
| Gemini integration | Fully implemented in code | Official google-genai SDK, lazy import, env-var config, fallbacks, status/connection endpoints, classified errors and redaction. |
| Gemini assistant tasks | Fully implemented | Four task cards in the results page: match explanation, resume improvement, bullet suggestions and interview preparation. |
| Consent handling | Fully implemented | UI checkbox and server-side consent check; without consent the API returns 400 and never calls Gemini. |
| Model-status indicator | Fully implemented | spaCy, embeddings and Gemini chips driven by `/api/health`. |
| Preserved backend | Fully implemented | `/api/extract`, `/api/nlp/pipeline`, `/api/extract/evaluation`, `/api/sentiment*`, `/api/topics*` and AI routes remain available. |
| Tests and documentation | Fully implemented | README, RUBRIC_MAPPING, DEMO_GUIDE, BUILD_STATUS and this report updated. Full suite in the documented `.venv`: **164 passed**. |

---

## Partially implemented / partially verified

| Item | Reason | Current mitigation |
|---|---|---|
| Visual review at every breakpoint | Desktop rendering was checked; not every screen size was manually inspected. | Responsive CSS rules exist at 1000/900/520 px and JS syntax check passes. |
| Real MiniLM inside the automated suite | Unit tests use a deterministic fake embedder for speed; real-model tests skip when weights are unavailable. | The app loads real MiniLM when available and falls back to TF-IDF with an explicit warning. |
| Score calibration | Rescaling ranges and weights are documented heuristics, not trained parameters. | Raw cosines are always displayed beside calibrated values. |
| Benchmark size | Extraction (10 cases) and sentiment (36 cases) are classroom-sized datasets. | Presented as evaluation demonstrations, not production accuracy claims. |

---

## Still missing

**None of the required assessment features are missing from the implementation.**

Live Gemini responses were verified using the user-provided API key in the local `.env` file, which is ignored by Git. `POST /api/ai/test` and all four consented tasks returned schema-valid HTTP 200 results with `advisory_only=true` and `does_not_change_scores=true`. A 45-second per-model timeout was added after a hung SDK call was observed; timeouts and per-model rate limits now use the same fallback chain as overloaded models. The key is never logged or returned to the browser.

Known intentional limitations (not missing requirements):

- Scanned/image-only PDFs are rejected because there is no OCR.
- Skill extraction is dictionary-based; unseen tools need an alias in `data/skills.json`.
- LDA is exploratory and flagged unreliable on very small corpora.
- VADER can miss sarcasm and domain-specific tone.
- Job links are external searches only; the app does not verify individual vacancies.
- Gemini output is advisory and must be reviewed by a human.

---

## Verification performed

- `python -m compileall -q app.py nlp tests` - passed.
- `node --check static/js/main.js` - passed.
- `python -m pytest -q` - **164 passed**.
- Both notebooks executed with nbclient and saved with outputs/images; zero error outputs.
- Sentiment notebook saved accuracy 0.917, macro-F1 0.916 and two visualisations.
- Topic notebook saved four LDA topics, a document-topic heatmap, keyword bars and perplexity 1063.19.
- API tests specifically cover `/api/job-links` encoding/validation and `/api/match` preview fields.
- Index test asserts a single dashboard and absence of the old five-tab navigation.
- Fake Gemini tests confirm consent, redaction, four task schemas, fallbacks, classified errors and unchanged local scores.
- A live browser session confirmed the single-dashboard frontend end to end: sample resume load, custom title/location inputs, a successful `/api/match` (HTTP 200) with rendered scores, NER table and extracted-text preview, five external job links that refresh when location changes, and consent gating that blocks any `/api/ai/insights` call when the checkbox is unticked.
- A live Gemini test with a user-provided key returned HTTP 200 for `/api/ai/test` and consented, schema-valid responses for all four tasks (`match_explanation`, `resume_improvement`, `bullet_suggestions`, `interview_prep`); redaction removed 1 email, 1 phone, 1 URL and 1 name before sending 484 characters on each call. Bullet suggestions rewrote only existing bullets without adding metrics.
- An overloaded-model bug was fixed: Gemini 503/high-demand responses are now classified as `model_unavailable` and fall back to `gemini-3.6-flash`, `gemini-3.5-flash` or `gemini-3.1-flash-lite`. A regression test was added.
