# Demo Guide

A practical walkthrough for presenting the project and answering viva questions. The demo works with **no Gemini key**; the AI card simply explains that cloud features are disabled while every local feature continues to work.

## Assessment criterion mapping

The full evidence table is in **`RUBRIC_COVERAGE.md`**. Use this quick map during the live demo.

| Assessment | Marks | Where to demonstrate it | Evidence to show |
|---|---:|---|---|
| **CO1 — Text extraction, preprocessing and vectorization** | 7 | Single matching dashboard + `/api/nlp/pipeline` | PDF/DOCX/TXT upload, cleaned text, token/lemma table, embeddings, TF-IDF, cosine, semantic + skill scores, skill gap |
| **CO2A — Information extraction** | 4 | NER table inside the report + `/api/extract` | Name, email, phone, URL, education, organization, dates and technical skills extracted from the uploaded document |
| **CO2B — Sentiment analysis notebook** | 4 | `notebooks/sentiment_analysis.ipynb` | VADER classification, 36 labelled cases, accuracy 0.917, macro-F1 0.916, per-class table, confusion matrix, interpretation |
| **CO3A — Topic modelling** | 3 | `notebooks/topic_modelling.ipynb` | CountVectorizer matrix, 4 LDA topics, keywords, document-topic heatmap, perplexity 1063.19 |
| **CO3B — Cloud AI service** | 4 | Header chip + Gemini card + `/api/ai/status` | Official `google-genai` SDK, key in environment/`.env`, connection test, classified errors, 45-second timeout, fallback chain, local NLP still works without key |
| **CO4 — Generative AI and prompt engineering** | 8 | Four Generate buttons inside the results page | Match explanation, resume improvement, bullet suggestions, interview questions; versioned prompts, consent, redaction, schema validation, advisory-only output and unchanged scores |

---

## 1. Pre-demo checklist

```powershell
cd D:\resume-job-matcher
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python app.py
```

Open http://127.0.0.1:5000.

- [ ] Status pill says backend ready and three chips are visible.
- [ ] Chips read spaCy NER ready, Embeddings loaded/lazy/fallback, Gemini optional or configured.
- [ ] There is **one** dashboard - no navigation tabs.
- [ ] `/api/health` returns JSON with `status: "ok"`.
- [ ] Both notebooks are open with saved outputs.

If the port is busy: `$env:PORT=5055; python app.py`.

### Verified evidence to quote

- `python -m pytest -q` → **164 passed**
- `python -m compileall -q app.py nlp tests` → passed
- `node --check static/js/main.js` → passed
- Both notebooks re-executed with nbclient → **0 errors**
- Live endpoint audit → **31/31 checks passed**
- Browser run → sample resume analyzed, report rendered, five external job links produced, consent gating blocked an unconsented Gemini call

---

## 2. Recommended 10-minute run sheet

| Time | Area | What to show | What to say |
|---|---|---|---|
| 1 min | Header | Status pill, three chips, blue/white layout | "One focused matching app; local NLP is default, Gemini is optional." |
| 3 min | Role + upload + report | Load sample resume, choose Data Scientist and Pune, click Analyze and rank | "NER, MiniLM embeddings, TF-IDF baseline and skill gap all produce this report." |
| 2 min | Report details | NER table, extracted-text preview, matched/missing skills, raw cosines | "Every number comes from the uploaded resume and the entered job." |
| 1 min | Job links | Show five Explore Job Opportunities links; change location to Remote | "These are real external searches, not invented vacancies." |
| 1 min | Notebooks | Show sentiment confusion matrix and LDA topic charts | "Sentiment and topics are separate executed subject notebooks." |
| 1 min | Gemini card | Without a key show 'not configured'; with a key run one consented task | "Consent is mandatory, PII is redacted, output is advisory only." |
| 1 min | Wrap-up | `pytest` result and RUBRIC_MAPPING.md | "164 tests; no live API needed." |

---

## 3. Copy-paste demo inputs

### Resume (save as `priya.txt` or use Load sample resume)

```text
Priya Sharma
priya.sharma@example.com | +91 98765 43210 | github.com/priyasharma
Data Scientist with 5 years of experience building machine learning models.
Skills: Python, SQL, pandas, NumPy, scikit-learn, XGBoost, Tableau, Apache Spark, statistics, A/B testing
Experience
- Built predictive models with scikit-learn and XGBoost and ran A/B testing experiments.
- Created Tableau dashboards that tracked model performance for business teams.
Education
B.Tech in Computer Science, Pune University
```

### Job description

```text
We are hiring a Data Scientist to use Python, pandas, NumPy, scikit-learn, SQL and Tableau for data analysis, statistics, machine learning and A/B testing. Experience with Apache Spark and AWS is a plus.
```

### Role and location demo

- Type job title: `AI Engineer`; location: `Pune` - show the five links.
- Change location to `Remote` or `India` - links refresh with the new encoded query.

---

## 4. Step-by-step script

### A. Matching dashboard

1. Select the **Data Scientist** sample role (or type your own title and paste a description).
2. Set location to **Pune**.
3. Click **Load sample resume** (or upload 1-3 PDF/DOCX/TXT files).
4. Click **Analyze and rank**.
5. Point out:
   - ranking with overall, semantic and TF-IDF scores plus raw cosines;
   - matched and missing skills with high-priority flags;
   - NER and extracted candidate information inside the result card;
   - extracted resume text preview;
   - fallback warning if MiniLM or spaCy NER is unavailable.
6. For a deeper CO1 demonstration, show that the same resume is converted to numerical form:

   ```powershell
   curl http://127.0.0.1:5000/api/nlp/pipeline -H "Content-Type: application/json" -d "{\"text\":\"Priya Sharma Python SQL pandas Data Scientist 5 years B.Tech Pune University\",\"job_description\":\"Data Scientist Python SQL pandas machine learning\"}"
   ```

   Point to `text.cleaned`, `tokens`, `lemmas`, `entities`, `vectorization.embeddings`, `vectorization.tfidf` and `matching.semantic_cosine` / `matching.tfidf_cosine`.

**Talking point:** sentiment and Gemini are absent from this calculation by design. Ranking is deterministic.

### B. Explore Job Opportunities

1. Show the section beneath the report.
2. Explain each link is an external search URL containing the entered title and location.
3. Click one link (optional) to show it opens a real job-search page in a new tab.
4. Emphasise the disclaimer: this app does not claim any specific vacancy exists.

### C. Notebooks

1. Open `notebooks/sentiment_analysis.ipynb` and show the saved confusion matrix and metrics.
2. Open `notebooks/topic_modelling.ipynb` and show the four topics, keyword bars and document-topic heatmap.
3. Mention both datasets are embedded so the notebooks run independently.

### D. Gemini card (inside results)

**Without a key:** show the status card and confirm local matching still works.

**With a key:**

1. Select a resume (defaults to top-ranked).
2. Tick the consent checkbox.
3. Run **Match explanation**; show summary, strengths, gaps and both redaction summaries (`redaction` for the resume, `job_redaction` for the title/description) plus `resume_chars_sent` and `job_chars_sent`.
4. Run **Bullet suggestions** on a resume with existing bullets; show that only existing bullets are rewritten.
5. Show `advisory_only` and `does_not_change_scores` in the response.

**Verified live:** a real key in the local `.env` file (ignored by Git) was used; `/api/ai/test` and all four consented tasks returned HTTP 200 with schema-valid output. In the earlier run, 1 email, 1 phone, 1 URL and 1 name were redacted from the resume; the current build also redacts the job title/description and neutralises untrusted text. Requests use a 45-second per-model timeout. If `gemini-3.8-flash` is overloaded, unavailable, rate-limited or times out, the app automatically falls back to `gemini-3.6-flash`, then `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-flash` or `gemini-2.0-flash`.

**Talking point:** consent is checked before any text is prepared or sent; without it the API returns 400 and Gemini is never called.

---

## 5. Viva questions and model answers

**Q: Why VADER instead of a trained classifier?** A: It is transparent, needs no training data and gives interpretable scores. The notebook still evaluates it on a balanced labelled set, so limitations are measured.

**Q: Why report macro-F1 as well as accuracy?** A: Macro-F1 averages per-class F1 equally, so a majority class cannot hide weak classes. The confusion matrix shows exact errors.

**Q: Why LDA with CountVectorizer?** A: LDA is a generative model for word counts; CountVectorizer supplies counts. TF-IDF is used for similarity, not as LDA input.

**Q: How do job links avoid fake vacancies?** A: The API only builds encoded search URLs for five known platforms. It never scrapes or stores listings, and the UI labels them external searches.

**Q: How do you stop Gemini inventing experience?** A: Prompts forbid new facts; context is limited to local analysis and a redacted excerpt; strict schema validation rejects unknown fields, invalid list items and missing answer hints; bullet rewrites must match supplied bullets and cannot add numbers or new non-trivial tokens (proper nouns/technologies).

**Q: How do you defend against prompt injection?** A: Resume, job title and job description are all bracket-neutralised and wrapped as untrusted data, system prompts explicitly treat them as data, and response validation is a second defence.

**Q: Does Gemini change ranking?** A: No. Scores are computed first and `does_not_change_scores` is true.

**Q: What leaves the machine?** A: Nothing unless consent is given. Even then, emails, URLs, phones and names are redacted from the resume **and** from the job title/description, only length-limited excerpts are sent, and the response shows `resume_chars_sent`, `job_chars_sent`, `redaction` and `job_redaction`.

---

## 6. Troubleshooting

| Symptom | Quick fix |
|---|---|
| Status pill red | Check terminal traceback; confirm server running |
| spaCy chip blank pipeline | `python -m spacy download en_core_web_sm`, restart |
| Embeddings chip fallback | No MiniLM download; check internet/proxy. Matching still works |
| Gemini card not configured | Expected without key; local features work |
| Job links look wrong | Verify title/location fields; links reflect exact input |
| Tests fail | Run from `D:\resume-job-matcher` with the venv active |

---

## 7. Closing summary (30 seconds)

"The system keeps the original deterministic matching pipeline and presents it in one focused dashboard: NER, MiniLM embeddings, TF-IDF, skill gap, ranking, extracted information and external job-search links. Sentiment and LDA are separate executed notebooks with real metrics and charts. Gemini is an optional, consent-gated card whose advisory output never changes local scores. The full suite passes with 164 tests and needs no API key. `RUBRIC_COVERAGE.md` maps every mark to executed evidence."
