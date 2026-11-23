# Rubric Mapping (30 marks)

This document maps each assessment criterion to the implementation, user-visible evidence, API surface and tests. Results are computed at request time or by executing the notebooks; there are no hard-coded scores or placeholder buttons.

**Architecture note:** the web app is now one Resume-Job Matching dashboard. Sentiment analysis and topic modelling are separate executed Jupyter notebooks. All original backend modules and routes are preserved.

## Quick verification

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python app.py
# open http://127.0.0.1:5000
```

`GET /api/health` reports the embedding backend, spaCy NER status, sentiment/topic availability and Gemini configuration. The dashboard shows three chips: spaCy, Embeddings and Gemini.

---

## Criterion 1 - Information extraction and vectorisation

| Requirement | Implementation | Evidence in the app |
|---|---|---|
| Resume parsing (PDF / DOCX / TXT) | nlp/resume_parser.py | Multi-file drag-and-drop on the one dashboard; Load sample resume adds a real TXT file |
| Text cleaning / preprocessing | nlp/preprocessing.py | Extracted-text preview in each candidate card; /api/nlp/pipeline shows tokens/lemmas |
| NER: names, organisations, locations, dates | nlp/ner_extractor.py (spaCy + regex) | NER and extracted candidate information table inside every result card |
| Contact details and education/experience | nlp/ner_extractor.py | Same result card: email, phone, links, education, years, organisations |
| Skills with normalisation and categories | nlp/ner_extractor.py + data/skills.json | Matched/missing/extra skill chips and category coverage bars |
| TF-IDF vectorisation | nlp/semantic_matcher.py | TF-IDF baseline score with raw cosine |
| Sentence embeddings and cosine similarity | nlp/embeddings.py (all-MiniLM-L6-v2) | Semantic score on every ranked candidate; fallback stated explicitly |
| Matching and skill-gap analysis | nlp/semantic_matcher.py, nlp/skill_gap.py | Ranking, score breakdown, matched/missing skills, recommendations |
| Extraction smoke benchmark | nlp/ie_evaluation.py + data/ie_eval_cases.json | GET /api/extract/evaluation reports precision/recall/F1 on 10 small self-consistent cases; a wiring check, not a general accuracy claim |
| Detailed understandable report | templates/index.html, static/js/main.js | Ranking, per-candidate scores, NER/profile, text preview, job requirements, weights |

**Key routes:** POST /api/match, POST /api/extract, POST /api/nlp/pipeline, GET /api/extract/evaluation.

**Tests:** tests/test_matcher.py, tests/test_information_extraction.py, tests/test_api_features.py.

**Why it is genuine:** the benchmark calls the live extractor on 10 annotated cases, so it verifies wiring, but those cases are short and self-consistent and currently score 1.0; it is a smoke benchmark rather than evidence of general NER accuracy. Matching payloads include text_stats and extracted_text_preview so the report displays real extracted content.

---

## Criterion 2 - Sentiment analysis and information extraction

### Sentiment (separate notebook)

| Requirement | Implementation | Evidence |
|---|---|---|
| Sentiment model | VADER in nlp/sentiment_analyzer.py | notebooks/sentiment_analysis.ipynb |
| Positive / negative / neutral / compound scores | SentimentIntensityAnalyzer.polarity_scores | Score columns, mean sub-score chart, compound strip plot |
| Manually labelled dataset | data/sentiment_dataset.json embedded inline | 36 short labelled texts, 12 per class: recruitment feedback, job reviews, job descriptions, cover letters and company blurbs |
| Text preprocessing | URL/email/punctuation normalisation | Display-only `clean_text` column and printed examples; VADER scores the original text so sentiment cues are preserved |
| Accuracy / precision / recall / macro-F1 | sklearn.metrics | Accuracy 0.917, macro-F1 0.916 in saved outputs |
| Confusion matrix | sklearn.metrics.confusion_matrix | 3x3 heatmap with saved image output |
| Actual calculated results and interpretation | Executed cells | Misclassified examples, per-class table, dynamic conclusion |
| No personality scoring | Scope statement in notebook and README | Uses job reviews/recruitment feedback only; never ranks candidates |

**Notebook:** notebooks/sentiment_analysis.ipynb (executed, charts and metrics saved).

### Information extraction

See Criterion 1. The dashboard result card itself shows persons, organisations, locations, dates, emails, phones, links, education, experience and categorised skills. The 10-case labelled benchmark remains available via GET /api/extract/evaluation.

**API preserved:** POST /api/sentiment, GET /api/sentiment/evaluation, POST /api/extract.

**Tests:** tests/test_sentiment.py, tests/test_information_extraction.py, tests/test_api_features.py.

---

## Criterion 3A - Topic modelling with LDA

| Requirement | Implementation | Evidence |
|---|---|---|
| LDA algorithm | sklearn LatentDirichletAllocation | notebooks/topic_modelling.ipynb and nlp/topic_modeling.py |
| Vectorisation | CountVectorizer with 1-2 grams and stop words | Matrix printed as 15 documents x 800 terms |
| Sample resume/job corpus | data/topic_corpus.json embedded inline | 15 anonymised docs across data, web, cloud, product/design |
| Text preprocessing | Lowercase, punctuation removal, stop words | Printed processed examples |
| Topic extraction and keywords | lda.components_ ranked weights | Four topic keyword tables and bar-chart subplots |
| Document-topic distributions | lda.fit_transform | Full distribution table, heatmap, dominant-topic counts |
| Diagnostics / interpretation | Perplexity 1063.19, top documents per topic | Saved text output and two image outputs |
| Robust API edge cases | nlp/topic_modeling.py | 400 on too few docs; reliable false on unstable corpora |

**Notebook:** notebooks/topic_modelling.ipynb (executed, charts and metrics saved).

**API preserved:** GET /api/topics/sample, POST /api/topics.

**Tests:** tests/test_topics.py, tests/test_api_features.py.

**Honest limitation:** LDA is exploratory; topic labels are descriptive, not ground truth.

---

## Criterion 3B - Google Gemini integration

| Requirement | Implementation | Evidence |
|---|---|---|
| Official SDK | google-genai, imported lazily in nlp/gemini_service.py | /api/health, /api/ai/status |
| API key configuration | GEMINI_API_KEY from environment / .env | .env.example, README |
| Model configuration | GEMINI_MODEL, default gemini-3.8-flash, with overload/404 fallback to gemini-3.6-flash, gemini-3.5-flash and gemini-3.1-flash-lite | Gemini chip and AI status card |
| No exposed keys | status() returns booleans/counts only | test_status_with_api_key_never_exposes_the_key |
| Connection status | /api/health, /api/ai/status, /api/ai/test | Gemini chip, AI status card, Test connection button |
| Robust errors | Classified not_configured, auth, rate_limit, timeout, network, model_unavailable, invalid response | Safe error banner in the results card |
| Local NLP without Gemini | Optional/lazy cloud service | Full suite passes with fake service; app starts with no key |
| Never auto-send resumes | Consent checked server-side before processing | test_ai_requires_consent_and_never_calls_gemini |
| Redaction | redact_text removes emails, URLs, phones, names | Response context.redaction counts |
| UI placement | Optional Gemini AI Career Assistant card inside matching results | No separate AI tab |

**Tests:** tests/test_gemini.py, tests/test_api_features.py.

---

## Criterion 4 - Gemini career assistant

| Feature | Task ID | Output schema |
|---|---|---|
| Match explanation | match_explanation | summary, strengths, gaps, score_interpretation, next_steps |
| Resume improvement | resume_improvement | summary, priority_actions, missing_skill_advice, project_ideas |
| Professional bullet suggestions | bullet_suggestions | bullets with original/improved/reason, notes |
| Interview preparation | interview_prep | technical_questions, hr_questions, preparation_tips |

| Safety / quality requirement | Implementation | Evidence |
|---|---|---|
| Separate prompt templates | nlp/prompt_templates.py | test_templates_are_distinct |
| No fabricated experience | Prompts forbid new employers/degrees/dates/metrics; bullet rewrites must reference existing bullets | test_bullet_rewrite_rejects_new_numbers |
| Untrusted text treated as data | Wrapped as untrusted resume/job with bracket neutralisation | test_untrusted_text_cannot_close_its_tag |
| Validated JSON | Task-specific schemas and type/length checks | tests/test_prompts.py |
| Scores unchanged | Local analysis runs before Gemini; response has advisory_only and does_not_change_scores | test_ai_does_not_change_local_scores |
| Explicit consent | Checkbox in results card plus server-side consent check | test_ai_requires_consent_and_never_calls_gemini |
| Frontend integration | Four task cards inside the matching results section | templates/index.html, static/js/main.js |

**API:** GET /api/ai/status, POST /api/ai/test, POST /api/ai/insights.

**Tests:** tests/test_prompts.py, tests/test_gemini.py, tests/test_api_features.py.

---

## Cross-cutting and new requirements

| Requirement | Implementation |
|---|---|
| Preserve existing backend/NLP modules | All original routes and modules retained; only the multi-tab UI was removed |
| One professional blue/white responsive dashboard | templates/index.html, static/css/style.css, static/js/main.js |
| Any job title, description and location | Free-text fields plus sample selector and location chips |
| External job-search links | GET /api/job-links; five encoded links, new tabs, external-search disclaimer, no fake vacancies |
| Extracted info and NER inside matching results | Candidate card tables and extracted-text preview |
| Sentiment / topic notebooks executed and saved | notebooks/*.ipynb with outputs and images |
| Gemini inside results, not a tab | Optional card after the report |
| Gemini reliability | 45-second per-model timeout; fallback on 503/high-demand, timeouts and per-model rate limits; classified auth and network errors |
| Loading / error / empty states | Real busy states, alert boxes, meaningful empty states |
| Documentation | README, this rubric map, DEMO_GUIDE, BUILD_STATUS, IMPLEMENTATION_REPORT |
| Tests | 164 passing tests across seven modules |

## Suggested evidence to show an examiner

1. Run python -m pytest -q and show 164 passed.
2. Open the single dashboard; show the role fields, location chips, upload area and three status chips.
3. Load the sample resume, choose Data Scientist and location Pune, then click Analyze and rank.
4. Point at semantic score, TF-IDF baseline, skill gap, NER/profile table and extracted-text preview.
5. Show the Explore Job Opportunities links for AI Engineer in Pune, then change location to Remote and watch them refresh.
6. Open both notebooks and show saved charts, confusion matrix, LDA topics and interpretation.
7. In the results card, show Gemini not configured (or, with a key, run one consented task) and point out the redaction counts and does_not_change_scores.
