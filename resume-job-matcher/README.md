# Intelligent Resume-Job Matching System

A focused Flask dashboard that reads PDF / DOCX / TXT resumes, extracts structured information with NER, ranks candidates with sentence embeddings + TF-IDF cosine similarity, analyses skill gaps, and generates **external job-search links** for any job title and location.

Sentiment analysis and LDA topic modelling are **separate executed Jupyter notebooks** (`notebooks/`). Gemini is an **optional card inside the matching results page** - local NLP works fully without an API key, and nothing is sent to Google without explicit consent.

**Stack:** Python 3.10+ · Flask · spaCy · Sentence Transformers (`all-MiniLM-L6-v2`) · scikit-learn · VADER · pandas / matplotlib / seaborn · google-genai · pypdf · python-docx · vanilla HTML/CSS/JS

---

## Architecture at a glance

| Surface | Purpose | Key files |
|---|---|---|
| **One dashboard** | Job inputs → resume upload → extraction/NER → embeddings + TF-IDF → skill gap → ranking → report → job links → optional Gemini | `templates/index.html`, `static/js/main.js`, `static/css/style.css`, `app.py` |
| **Reusable NLP backend** | Parsing, preprocessing, NER, skill taxonomy, embeddings, matching, skill gap, pipeline demo, IE benchmark | `nlp/*.py`, `data/*.json` |
| **Sentiment notebook** | Independently executed recruitment-feedback sentiment exercise with metrics and charts | `notebooks/sentiment_analysis.ipynb` |
| **Topic notebook** | Independently executed LDA exercise on resume/job corpus with keywords and charts | `notebooks/topic_modelling.ipynb` |
| **Optional cloud AI** | Four consent-gated Gemini tasks inside the results page; never changes scores | `nlp/gemini_service.py`, `nlp/prompt_templates.py` |
| **Vercel entrypoint** | Serves the existing dashboard and proxies `/api/*` to the NLP service | `vercel_proxy.py`, `pyproject.toml`, `vercel.json`, `requirements.txt` |
| **External NLP service** | Runs the unchanged full Flask app plus token auth and warmup | `nlp_api.py`, `Procfile`, `requirements-backend.txt` |

The previous five-tab navigation was removed from the UI. The backend modules and routes for extraction, sentiment and topics are **preserved** for tests and API use. The notebooks embed the same small classroom datasets inline so they execute independently.

---

## Main dashboard features

### 1. Role and resume inputs
- Type **any job title** (AI Engineer, Python Developer, Data Scientist, Web Developer, etc.) or start from 5 sample roles.
- Paste/edit a custom job description (at least 10 words) or use the sample text.
- Enter **any preferred location**. Quick chips: Pune, Mumbai, Bengaluru, Remote and Across India.
- Drag-and-drop or browse for up to 10 PDF/DOCX/TXT resumes (5 MB each).
- **Load sample resume** adds a real in-memory TXT file for an instant demo.

### 2. Local NLP pipeline (runs on this machine)
- Text extraction and Unicode cleaning (`nlp/resume_parser.py`, `nlp/preprocessing.py`).
- spaCy `en_core_web_sm` NER plus regex extraction for names, emails, phones, URLs, education, dates and years of experience (`nlp/ner_extractor.py`).
- 116-skill taxonomy with aliases and categories (`data/skills.json`) so `ReactJS`/`react.js`/`React` normalise to **React**.
- `all-MiniLM-L6-v2` sentence embeddings with chunking and averaging, cosine similarity, and a TF-IDF (1-2-gram) baseline (`nlp/embeddings.py`, `nlp/semantic_matcher.py`).
- Skill-gap analysis: matched, missing and extra skills with priority and category coverage (`nlp/skill_gap.py`).
- Transparent scoring: raw cosines are shown beside calibrated 0-100 scores; semantic backend and NER availability are stated explicitly.

### 3. Matching report
For every resume the report shows:
- rank, verdict and overall score;
- MiniLM semantic score and TF-IDF baseline with raw cosines;
- matched, missing (high-priority flagged) and extra skills;
- category coverage bars and grounded suggestions;
- **NER and extracted candidate information** (name, contacts, education, experience, organisations, locations, dates) directly inside the result card;
- a preview of the actual extracted resume text;
- score weights, job requirements and clear fallback warnings.

The extraction and pipeline APIs remain available at `POST /api/extract` and `POST /api/nlp/pipeline`, and the labelled extraction benchmark remains at `GET /api/extract/evaluation`.

### 4. Explore Job Opportunities
After the report, `GET /api/job-links` builds **real external search URLs** from the entered title and location:

- LinkedIn Jobs - `jobs/search/?keywords=...&location=...`
- Indeed India - `in.indeed.com/jobs?q=...&l=...`
- Naukri.com - slug plus `query`/`city` parameters
- Glassdoor India - `sc.keyword` plus `locName`
- Wellfound - `jobs?q=...&l=...`

Links open in a new tab (`target="_blank" rel="noopener noreferrer"`). The UI labels them **External job search** and states that the app does **not verify or invent individual vacancies**. Changing the title or location refreshes the links without rerunning matching.

### 5. Optional Gemini Career Assistant (inside results)
Four tasks:
1. **Match explanation** - explains local strengths, gaps and score interpretation.
2. **Resume improvement** - priority actions based on the computed gap.
3. **Bullet suggestions** - rewrites *existing* bullets without inventing numbers or experience.
4. **Interview preparation** - technical and HR questions with answer hints.

Safety: consent checkbox per request, redaction of emails/phones/URLs/names, length-limited excerpt, untrusted text wrapping, strict JSON schemas, task-specific prompts, classified errors, and `advisory_only: true` / `does_not_change_scores: true`. Local matching completes before any cloud call and works with no key.

---

## Project layout

```
resume-job-matcher/
├── app.py                      Full Flask app (15 endpoints), used locally and on the NLP service
├── vercel_proxy.py             Lightweight Vercel entrypoint (no heavy NLP imports)
├── nlp_api.py                  External backend entrypoint (auth + warmup + model cache)
├── pyproject.toml              Vercel entrypoint = vercel_proxy:app
├── vercel.json                 Function duration / bundle exclusions / install command
├── requirements.txt            Vercel-only dependencies (Flask + requests)
├── requirements-backend.txt    Full NLP stack with CPU-only PyTorch
├── requirements-dev.txt        Tests + notebook dependencies
├── Procfile / .python-version  Backend process and Python version
├── .vercelignore / .env.example / .gitignore
├── README.md
├── RUBRIC_MAPPING.md           Four-criterion evidence map
├── DEMO_GUIDE.md               Presentation walkthrough
├── BUILD_STATUS.md             Verification log
├── IMPLEMENTATION_REPORT.md    Fully / partially / missing status
├── nlp/
│   ├── resume_parser.py        PDF / DOCX / TXT extraction
│   ├── preprocessing.py        Cleaning, spaCy pipeline, lemmatisation
│   ├── ner_extractor.py        NER + regex + 116-skill normalisation
│   ├── embeddings.py           MiniLM wrapper (lazy, chunked, shared)
│   ├── semantic_matcher.py     Semantic/TF-IDF scores, ranking, report payload
│   ├── skill_gap.py            Matched / missing / extra skills
│   ├── pipeline_demo.py        Transparent NLP pipeline
│   ├── sentiment_analyzer.py   VADER + labelled evaluation (API/notebook support)
│   ├── topic_modeling.py       LDA pipeline (API/notebook support)
│   ├── ie_evaluation.py        Extraction benchmark
│   ├── prompt_templates.py     Four Gemini prompts + JSON validation
│   └── gemini_service.py       SDK client, redaction, fallbacks, errors
├── data/
│   ├── skills.json             116-skill taxonomy with aliases
│   ├── sample_jobs.json        5 sample roles
│   ├── sentiment_dataset.json  36 balanced labelled sentences
│   ├── topic_corpus.json       15-document anonymised corpus
│   └── ie_eval_cases.json      10 annotated extraction cases
├── notebooks/
│   ├── sentiment_analysis.ipynb   Executed VADER notebook (outputs + charts saved)
│   └── topic_modelling.ipynb      Executed LDA notebook (outputs + charts saved)
├── templates/index.html        One dashboard, no navigation tabs
├── static/css/style.css        Blue/white responsive theme
├── static/js/main.js           Single-dashboard workflow + AI + job links
└── tests/                      9 pytest modules (172 tests)
```

---

## Installation (Windows, PowerShell)

```powershell
cd resume-job-matcher

python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

python -m pip install --upgrade pip

# Full local / NLP backend runtime (CPU-only PyTorch)
pip install -r requirements-backend.txt

# Optional: notebooks, pytest and PDF-generation tests
pip install -r requirements-dev.txt
```

`requirements.txt` is intentionally **Vercel-only** (Flask + requests). The heavy NLP stack, including CPU-only PyTorch and the spaCy model wheel, lives in `requirements-backend.txt`. This separation is what prevents the multi-GB Vercel bundle.

> First run needs internet to download `all-MiniLM-L6-v2` (~90 MB), cached in `%USERPROFILE%\.cache\huggingface`. Afterwards local NLP works offline. If the download fails, the app explicitly falls back to TF-IDF.

### Optional Gemini key

```powershell
Copy-Item .env.example .env
notepad .env        # set GEMINI_API_KEY=your-key-here
```

Or for one session: `$env:GEMINI_API_KEY = "your-key-here"`. The key is read server-side only and is never logged or returned to the browser. `.env.example` contains only a placeholder; never commit a real key. Default model is `gemini-3.8-flash`; if it is overloaded, rate-limited, unavailable or times out, the app automatically tries `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-flash` and `gemini-2.0-flash`. Before any cloud call, both the resume excerpt and the job title/description are redacted and untrusted text is neutralised in prompts.

---

## Run and use

```powershell
python app.py
# open http://127.0.0.1:5000
```

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `5000` | Port |
| `HOST` | `127.0.0.1` | Bind address |
| `PRELOAD_MODEL` | `1` | `0` loads MiniLM on first analysis |
| `GEMINI_API_KEY` | empty | Enables optional cloud AI |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Primary model |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.6-flash,gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash,gemini-2.0-flash` | Tried in order when the primary model is unavailable or overloaded |

### Dashboard workflow
1. Enter a job title, edit/paste a description and set a location (or click a sample role/location chip).
2. Add one or more resumes (or click **Load sample resume**).
3. Click **Analyze & rank**.
4. Read ranking, scores, NER/profile, skill gap and extracted-text preview.
5. Use **Explore Job Opportunities** to open real external searches for the title/location.
6. Optionally tick the consent checkbox and run a Gemini task in the **Gemini AI Career Assistant** card.

Header chips report spaCy, embedding and Gemini status from `/api/health`.

---

## Vercel deployment (split architecture)

### Why the split is required

The original Vercel build installed `sentence-transformers` with the default Linux PyTorch wheels. Those wheels pull several `nvidia-*-cu12` / CUDA packages; combined with `torch`, `transformers`, `spacy`, `scipy`, `pandas` and model caches this produced the reported **5876.36 MB** function bundle. Vercel's standard Python function limit is **500 MB uncompressed**; Large Functions are a 5 GB beta and the original bundle was still larger than that. Even CPU-only PyTorch plus spaCy and scikit-learn exceeds 500 MB, so the full NLP pipeline cannot honestly run inside one standard Vercel function.

This project therefore uses the smallest practical split:

```text
Browser ──same-origin /api/*──▶ Vercel Flask proxy ──HTTPS + shared token──▶ External NLP service
   │                                  │                                      │
   ├─ templates/index.html            ├─ no torch / spaCy / transformers     ├─ unchanged app.py
   ├─ static/css, static/js           ├─ serves dashboard + static assets    ├─ MiniLM, spaCy, TF-IDF
   └─ existing API payloads           └─ allowlisted proxy only              ├─ Gemini (optional)
                                                                              └─ job links, all 15 endpoints
```

The browser code is unchanged: it still calls `/api/...` on the Vercel domain. The Vercel function adds `X-NLP-API-Token` server-side; the browser never sees it.

### 1. Deploy the external NLP service first

Any Python host that supports a `Procfile` works (Render, Railway, Fly.io). Example for Render:

1. Create a **Web Service** from `https://github.com/Suyash-Hadole/NLP-Resume-Job-Matcher` and set **Root Directory** to `resume-job-matcher`.
2. Runtime: **Python 3.12**.
3. Build command: `pip install -r requirements-backend.txt`
4. Start command: `gunicorn nlp_api:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --graceful-timeout 60`
5. Use at least **2 GB RAM** (4 GB recommended). Free/sleeping tiers work but the first request after idle is slow.
6. Set service environment variables:

| Variable | Required | Example | Purpose |
|---|---|---|---|
| `NLP_API_TOKEN` | Yes | `openssl rand -hex 32` output | Shared secret; Vercel proxy sends it as `X-NLP-API-Token` |
| `GEMINI_API_KEY` | No | your key | Enables optional Gemini on the **backend only** |
| `GEMINI_MODEL` | No | `gemini-3.8-flash` | Primary Gemini model |
| `GEMINI_FALLBACK_MODELS` | No | see `.env.example` | Fallback model order |
| `MODEL_CACHE_ROOT` | No | `/tmp/resume-job-matcher-models` | Writable model cache |
| `PRELOAD_MODEL` | No | `0` | Lazy loading; use `/api/warmup` after deploy |

7. After the first deploy, call `GET https://<your-backend>/api/warmup`. It downloads `all-MiniLM-L6-v2` (~90 MB), loads spaCy `en_core_web_sm`, and reports the real embedding/NER state. `GET /api/health` is intentionally public for monitoring; every other `/api/*` route requires the token when `NLP_API_TOKEN` is set.

The spaCy model wheel is pinned in `requirements-backend.txt`, so no separate `python -m spacy download` step is needed on the host.

### 2. Deploy Vercel

1. Import the same repository and set **Root Directory** to `resume-job-matcher`.
2. Framework Preset: **Flask** (auto-detected from `requirements.txt`).
3. Install command: `pip install -r requirements.txt` (already declared in `vercel.json`; it installs only Flask, requests and python-dotenv). Build command: `python scripts/prepare_vercel_public.py`, which copies the existing `static/` assets into `public/static/**` because Vercel serves static files from `public/**` and ignores Flask's `static_folder`.
4. `pyproject.toml` points Vercel to `vercel_proxy:app`, so `app.py` is never imported in the Vercel function. The proxy keeps an explicit Flask static route for local development; on Vercel the CDN serves the same files from `public/static/**`.
5. Add Vercel project environment variables:

| Variable | Required | Example | Purpose |
|---|---|---|---|
| `NLP_API_URL` | Yes | `https://resume-nlp.onrender.com` | Backend base URL (no trailing `/api`) |
| `NLP_API_TOKEN` | Yes | same value as backend | Server-side shared secret |
| `NLP_REQUEST_TIMEOUT` | No | `280` | Proxy timeout in seconds |

Do **not** add `GEMINI_API_KEY` to Vercel. Do not put any token in `static/js/main.js`, `templates/index.html` or GitHub.

6. Deploy. `.vercelignore` and `functions.vercel_proxy.py.excludeFiles` keep `.venv`, caches, notebooks, tests, model weights and backend-only files out of the function bundle. A local estimate with `.venv\Scripts\python.exe scripts\measure_vercel_bundle.py` is about **7.8 MB uncompressed** (0.27 MB source + 7.55 MB minimal packages), versus the 500 MB standard limit. For the definitive report, run `VERCEL_ANALYZE_BUILD_OUTPUT=1 npx vercel build --yes` in the project.

### 3. Verify the deployed split

- `GET /` renders the original dashboard; `/static/css/style.css` and `/static/js/main.js` load with 200.
- `GET /api/health` returns the backend's real MiniLM/spaCy/Gemini status through the proxy.
- Upload a PDF/DOCX/TXT resume and run **Analyze & rank**; scores, NER, skill gap and preview come from the real backend.
- **Explore Job Opportunities** still produces the five encoded external search URLs.
- With `GEMINI_API_KEY` set on the backend, the consent-gated AI card works; without it the card honestly reports "not configured".

### Remaining platform limitations

- Vercel request bodies are limited to **4.5 MB by default** (20 MB with the beta opt-in), while the UI allows 10 files × 5 MB. On Vercel, unusually large batches should be sent directly to the backend with the token, or the beta body limit should be enabled. Normal-sized resumes work unchanged.
- The proxy uses `"maxDuration": "max"` for the current plan; very slow cold backend starts can still hit that ceiling. The `/api/warmup` endpoint reduces this risk.
- Vercel has no persistent writable filesystem for model caches. No model is downloaded or stored there by design.
- All real calculations (NER, embeddings, TF-IDF, hybrid score, ranking, skill gap, Gemini) happen on the external service. Vercel holds no dummy or hardcoded results.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Single matching dashboard |
| `GET /api/health` | GET | Model/backend status and feature availability |
| `GET /api/jobs` | GET | 5 sample roles |
| `GET /api/job-links` | GET | `title`, `location` to 5 encoded external search URLs plus disclaimer |
| `POST /api/match` | POST | `job_title`, `job_description`, 1-10 `resumes` to ranked report with text preview/stats |
| `POST /api/extract` | POST | Structured entities, skills, optional skill gap |
| `POST /api/nlp/pipeline` | POST | Tokens, lemmas, vectors, TF-IDF, embeddings, cosine |
| `GET /api/extract/evaluation` | GET | Labelled extraction benchmark |
| `POST /api/sentiment` / `GET /api/sentiment/evaluation` | POST / GET | VADER analysis / metrics (notebook support) |
| `GET /api/topics/sample` / `POST /api/topics` | GET / POST | LDA corpus / modelling (notebook support) |
| `GET /api/ai/status` / `POST /api/ai/test` | GET / POST | Gemini status / connection test |
| `POST /api/ai/insights` | POST | Consent-gated task; resume file or `resume_text`, job context |

```powershell
curl.exe "http://127.0.0.1:5000/api/job-links?title=AI%20Engineer&location=Pune"
curl.exe -X POST http://127.0.0.1:5000/api/match -F job_title="AI Engineer" -F "job_description=..." -F "resumes=@C:\path\cv.pdf"
```

`/api/ai/insights` returns `400 consent_required` without consent; `/api/ai/test` returns `503 not_configured` without a key. Neither calls Gemini in those cases.

---

## How scoring works

All scores are 0-100; raw cosines are shown beside them.

| Component | Weight | Definition |
|---|---|---|
| Semantic | 50% | MiniLM embedding cosine, rescaled (0.15 to 0, 0.70 to 100) |
| Skill coverage | 35% | % of recognised job skills found in the resume |
| TF-IDF baseline | 15% | Lemmatised 1-2-gram TF-IDF cosine, rescaled (0.50 to 100) |

If the job has no recognised skills, the skill weight is removed and the rest renormalised. Verdicts: 75+ Excellent, 60+ Good, 40+ Moderate, else Weak. Constants live in `nlp/semantic_matcher.py`. Sentiment and Gemini **never** affect these scores.

---

## Jupyter notebooks

Both notebooks are saved **with executed outputs and charts**. Run them from the project root:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\sentiment_analysis.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\topic_modelling.ipynb
```

If the `jupyter` CLI is not installed, `nbclient` is in `requirements.txt` and the notebooks can also be run with *Kernel -> Restart & Run All* in JupyterLab/VS Code.

| Notebook | Dataset | Calculated results saved in outputs |
|---|---|---|
| `sentiment_analysis.ipynb` | 36 hand-labelled recruitment/job-review texts embedded inline (12 per class) | VADER scores, class visualisations, accuracy **0.917**, macro-F1 **0.916**, per-class precision/recall/F1, 3x3 confusion matrix, misclassified examples, interpretation |
| `topic_modelling.ipynb` | 15 anonymised resume/job/project documents embedded inline | CountVectorizer matrix (15x800), 4-topic LDA, perplexity **1063.19**, top keywords/weights, document-topic heatmap, dominant-topic counts, interpretation |

The notebooks are teaching exercises; sentiment is not candidate personality scoring and LDA topics are exploratory, not ground truth.

---

## Testing and verification

```powershell
python -m compileall -q app.py nlp tests
node --check static/js/main.js
python -m pytest -q
```

Latest full suite in the documented `.venv`: **164 passed**. Tests use a deterministic fake embedder and a fake Gemini service; no key or network call is required. A bare system Python is unsupported because it lacks `vaderSentiment` and `google-genai`. A separate live Gemini check was performed with the user-provided key in the local `.env` file (ignored by Git). Requests use a 45-second per-model timeout and automatically fall back on timeouts, overloaded models or per-model rate limits. See `BUILD_STATUS.md` and `IMPLEMENTATION_REPORT.md` for the executed verification log.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| spaCy chip says blank pipeline | `python -m spacy download en_core_web_sm`, restart |
| Embeddings chip says TF-IDF fallback | No MiniLM download; check internet/proxy and restart. Matching still works |
| Gemini card says not configured | Expected without key. Local matching is unaffected; set `GEMINI_API_KEY` to enable it |
| Gemini task returns consent error | Tick the consent checkbox in the results card and retry |
| Job links look wrong | Check the title/location fields; links are generated from those exact values and are external searches, not verified vacancies |
| `ModuleNotFoundError: nlp` in tests | Run `python -m pytest` from `D:\resume-job-matcher` |
| PDF rejected | Scanned/image-only PDFs are unsupported (no OCR); export a text-based PDF or use DOCX/TXT |

---

## Notes and limitations

- Built-in Flask server is for local use; use a WSGI server for deployment.
- Extraction and sentiment benchmarks are small classroom datasets, not production accuracy claims.
- Skill extraction is dictionary-based; add aliases in `data/skills.json`.
- Experience is taken from explicit "N years" statements or estimated from date ranges and labelled as an estimate.
- Gemini output is advisory and must be reviewed by a human.
