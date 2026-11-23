# Build status

Last updated: 2026-11-23

## Architecture change completed

- The UI is now **one Resume-Job Matching dashboard**; the five navigation tabs were removed from `templates/index.html` and `static/js/main.js`.
- All backend/NLP modules and API routes are preserved. Extraction, sentiment and topic endpoints still work for tests/API use.
- Sentiment analysis and topic modelling now live in **executed Jupyter notebooks** with saved outputs and charts.
- New `GET /api/job-links` generates five URL-encoded external search URLs for any title/location.
- Matching results now include `text_stats` and `extracted_text_preview`, rendered inside each candidate card along with NER/profile data.
- Gemini is an optional card inside the matching results section with explicit consent, redaction and no score changes.

## Completed capabilities

- **Criterion 1:** resume parsing, cleaning, spaCy preprocessing, NER + regex extraction, 116-skill normalisation, TF-IDF and MiniLM embeddings, cosine similarity, skill-gap analysis, ranking, transparent pipeline and a 10-case extraction smoke benchmark.
- **Criterion 2:** VADER sentiment in `notebooks/sentiment_analysis.ipynb` with a 36-sentence balanced labelled set, accuracy 0.917, macro-F1 0.916, per-class metrics, 3x3 confusion matrix; structured extraction in the dashboard.
- **Criterion 3A:** `notebooks/topic_modelling.ipynb` with CountVectorizer and scikit-learn LDA over 15 documents, four topics, keyword weights, document-topic heatmap and perplexity 1063.19; API edge cases preserved.
- **Criterion 3B:** official google-genai integration with secure env vars, fallbacks, status/connection test, classified errors, redaction, consent gating.
- **Criterion 4:** four Gemini tasks with separate prompt templates, schema validation, prompt-injection defence, bullet anti-fabrication checks and advisory-only output.
- Blue/white responsive dashboard, loading/error/empty states, local-vs-cloud labels and three model status chips.
- Documentation: README, RUBRIC_MAPPING, DEMO_GUIDE, BUILD_STATUS, IMPLEMENTATION_REPORT.

## Verified (actually executed)

- `python -m compileall -q app.py nlp tests` - passed.
- `node --check static/js/main.js` - passed.
- `python -m pytest -q` - **164 passed in the documented `.venv`** (was 149 before the architecture change; new tests cover the single dashboard, job links, preview fields, Gemini fallback, redaction and strict response validation). A bare system Python is unsupported because it lacks `vaderSentiment` and `google-genai`.
- Sentiment notebook executed with nbclient: 16 cells, 23 outputs, 2 saved images, no errors.
- Topic notebook executed with nbclient: 19 cells, 16 outputs, 2 saved images, no errors.
- Notebook results: sentiment accuracy 0.917 / macro-F1 0.916; LDA 4 topics, 15x800 matrix, perplexity 1063.19.
- Test suite makes no Gemini API calls and needs no key; fake services are injected through `create_app`.
- `/api/job-links` tests confirm URL encoding for `AI Engineer` and `Pune + Remote`, default location India, all five platforms, validation errors and external-search disclaimer.

## Live smoke-test checklist for the new dashboard

| Route / action | Expected |
|---|---|
| GET / | 200, single dashboard, no data-tab navigation, Explore Job Opportunities and Gemini card present |
| GET /api/health | ok; embedding/nlp/gemini status objects |
| GET /api/jobs | 5 sample roles |
| GET /api/job-links?title=AI Engineer&location=Pune | 5 encoded external URLs with disclaimer |
| POST /api/match | ranked candidate with scores, NER, text preview/stats |
| POST /api/extract | structured entities and skills |
| POST /api/nlp/pipeline | tokens, lemmas, TF-IDF and embeddings |
| GET /api/extract/evaluation | labelled benchmark metrics |
| POST /api/sentiment | VADER scores (API preserved) |
| GET /api/topics/sample | 15 documents (API preserved) |
| GET /api/ai/status | 4 tasks, consent required |
| POST /api/ai/test | 503 not_configured without key (expected) |
| POST /api/ai/insights without consent | 400 consent_required |

### Observed browser run (2026-11-22)

A live browser session on `http://127.0.0.1:5055` confirmed the full frontend workflow:

- Header chips read **spaCy: NER ready**, **Embeddings: all-MiniLM-L6-v2 loaded** and **Gemini: optional (not configured)**.
- **Load sample resume** added `sample_priya.txt`; typing `AI Engineer` and selecting the `Remote` location chip worked.
- Clicking **Analyze & rank** produced one live `/api/match` request (HTTP 200) and rendered the matching report: rank #1 Priya Sharma, overall 27, semantic 43 (raw cosine 38.7%), TF-IDF baseline 9.2 (raw cosine 4.6%), skill coverage 11.8% (2 of 17 required skills), NER/extracted-information table, and the extracted-text disclosure. These values depend on the exact job text; a later run with a Data Scientist description produced 73.1 / 74.0 / 46.2 / 83.3.
- **Explore Job Opportunities** rendered five real external search links (LinkedIn, Indeed India, Naukri, Glassdoor, Wellfound) with the external-search disclaimer. Changing location from `Remote` to `Pune` refreshed all five hrefs without rerunning matching.
- Clicking a Gemini **Generate** button with consent unticked showed the consent error banner and made **zero** `/api/ai/insights` requests.

Earlier `browser.wait` timeouts were caused by stale element refs after DOM re-renders, not by application JavaScript. Re-querying refs and clicking the current button worked on the first attempt.

## Observed live Gemini run (2026-11-22)

- With the user-provided key in the project `.env` (ignored by Git), `POST /api/ai/test` returned HTTP 200. It used `gemini-3.6-flash` after the primary was overloaded, and an earlier run used `gemini-3.5-flash` after two models were overloaded, demonstrating the fallback chain.
- A hung SDK call could leave the browser waiting with no server error. A 45-second per-model `HttpOptions.timeout` was added, and timeouts now continue down the fallback chain. After the fix, the connection test returned in 22 seconds via `gemini-3.6-flash`.
- All four consented `/api/ai/insights` tasks returned HTTP 200 with schema-valid results: `match_explanation`, `resume_improvement`, `bullet_suggestions` and `interview_prep`. Each response had `consent=true`, `advisory_only=true` and `does_not_change_scores=true`.
- Bullet suggestions rewrote only the two existing bullets and explicitly reported that no metrics were introduced.
- Redaction summary on every live call: 1 email, 1 phone, 1 URL and 1 name removed; 484 redacted characters sent.
- Repeated live tests triggered HTTP 429 from the free tier. The Google error showed the quota is **per model** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), so `rate_limit` now falls back to another model too. If every model is exhausted, the user gets a clear "wait and retry" message.
- Final verification with `gemini-3.8-flash` quota exhausted: `/api/ai/test` returned HTTP 200 using `gemini-3.6-flash` in 37.9 seconds, and a consented `/api/ai/insights` call also returned HTTP 200 using `gemini-3.6-flash`.
- Earlier `503 UNAVAILABLE` responses from an overloaded `gemini-3.8-flash` exposed a fallback bug: 503 was classified as a generic API error instead of a model-unavailable error. This was fixed by classifying 503/high-demand as `model_unavailable` and updating fallbacks to `gemini-3.6-flash`, `gemini-3.5-flash` and `gemini-3.1-flash-lite`.

## Final 30-mark audit (2026-11-23)

- Re-inspected the actual implementation against every CO1–CO4 rubric requirement. No missing or partial criterion was found.
- Created `RUBRIC_COVERAGE.md` with per-criterion files, evidence and status.
- Expanded `DEMO_GUIDE.md` with explicit CO1–CO4 demonstration steps, verified evidence and an `/api/nlp/pipeline` command for showing intermediate vectorization results.
- Re-ran `python -m pytest -q` → **164 passed**.
- Re-ran `python -m compileall -q app.py nlp tests` and `node --check static/js/main.js` → passed.
- Re-executed both notebooks with nbclient: sentiment 8 code cells / 0 errors, topic modelling 9 code cells / 0 errors. Saved outputs remain: sentiment 23 outputs + 2 images, LDA 16 outputs + 2 images.
- Ran a live 31-check endpoint audit (`scripts/verify_endpoints.py`) → **31/31 passed**, covering single dashboard, health, jobs, job links, match, extraction, NLP pipeline, sentiment evaluation, LDA topics and Gemini status.
- Browser re-verification: loaded sample resume, produced a ranked report with NER/score cards/extracted text/preview, rendered five genuine external search URLs, and confirmed an unticked consent checkbox blocks `/api/ai/insights`.
- Security/safety fixes completed in this audit: `.env.example` is placeholder-only (repo scan found no key outside `.env`), both job title and job description are redacted before cloud prompts, job/title text is bracket-neutralised, unknown response fields/invalid list items/missing `answer_hints`/invented bullet facts are rejected, and `configured` now means key-present while `cloud_features_enabled` also requires the SDK.
- Documentation corrected: test count qualified to the documented venv, IE benchmark labelled a smoke benchmark, sentiment dataset scope described honestly, sentiment preprocessing marked display-only, and stale browser numbers replaced by reproducible values.
- Final status: **all four criteria completed and demonstrable; 30/30 target coverage**.

## Vercel deployment hardening (2026-11-23)

- Diagnosed the reported **5876.36 MB** bundle: a Linux Vercel install of the original `requirements.txt` pulls default CUDA/nvidia PyTorch wheels through `sentence-transformers`; local Windows `.venv` is only 1.43 GiB because it has CPU-only torch.
- Standard Vercel Python functions allow **500 MB uncompressed**; Large Functions are a 5 GB beta, so the original bundle exceeded both.
- Kept the full Flask/NLP app unchanged in `app.py` and moved Vercel to a lightweight same-origin proxy: `vercel_proxy.py` + `pyproject.toml` entrypoint + slim `requirements.txt`.
- Added `nlp_api.py`, `Procfile`, `requirements-backend.txt` (CPU-only torch + pinned spaCy model wheel), `/api/warmup`, shared-token auth and writable model-cache paths for the external inference service.
- Added `.vercelignore`, `functions.vercel_proxy.py.excludeFiles`, `.env.example` variables and split deployment instructions in `README.md`.
- Added `scripts/prepare_vercel_public.py` + `vercel.json` build command so the existing `static/**` files are copied to Vercel's required `public/static/**`; the Flask static route remains for local development.
- Added `tests/test_vercel_proxy.py` and `tests/test_nlp_api.py`: dashboard/static routes, allowlist, token forwarding, multipart upload, connection errors, backend shared-token auth and a subprocess check that the Vercel import pulls **no** torch/spacy/sentence-transformers/sklearn/app modules.
- Ran `.venv\\Scripts\\python.exe scripts\\measure_vercel_bundle.py` → source **0.27 MB** + minimal runtime packages **7.55 MB** = **7.82 MB estimated uncompressed**, far below the 500 MB limit. `vercel build --yes` could not be run here because the CLI has no valid login; the user should run it with `VERCEL_ANALYZE_BUILD_OUTPUT=1` after setting project env vars.
- Local split-stack smoke test: full NLP app on `:5055` + Vercel proxy on `:3000` → dashboard 200, CSS/JS 200, health reported `sentence-transformers / all-MiniLM-L6-v2` and spaCy NER ready, real `/api/match` returned a ranked candidate with overall score 88.6, `/api/jobs` returned 5 roles.
- Ran `scripts/verify_endpoints.py` against the unchanged full app → **31/31 passed**, including MiniLM, spaCy NER, TF-IDF, hybrid scores, extraction, sentiment, LDA and Gemini status.
- Full pytest suite after the deployment changes: **172 passed** (164 original + 8 deployment tests).

## Not verified in this environment

1. **Every responsive breakpoint visually.** Responsive rules exist at 1000/900/520 px; desktop rendering was checked.
2. **Real MiniLM inside the automated suite.** Unit tests use a deterministic fake embedder for speed; real-model tests skip automatically if weights are unavailable. The app itself loads MiniLM when available and reports fallback otherwise.
3. **Score calibration.** Rescaling ranges and weights remain documented heuristics; raw cosines are shown.

## Known limitations

- Scanned/image-only PDFs are rejected (no OCR).
- Skill extraction is dictionary-based; unseen tools need an alias in data/skills.json.
- Extraction and sentiment benchmarks are small classroom datasets.
- LDA is exploratory and flagged unreliable on very small corpora.
- VADER can miss sarcasm and domain-specific tone.
- Job links are external searches only; this app does not verify individual vacancies.
- Gemini output is advisory only and must be reviewed by a human.
