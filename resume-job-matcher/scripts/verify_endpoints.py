import json
from urllib.parse import quote_plus

import requests

BASE = "http://127.0.0.1:5055"
RESUME = """Priya Sharma
priya.sharma@example.com | +91 98765 43210 | github.com/priyasharma
Data Scientist with 5 years of experience building machine learning models.
Skills: Python, SQL, pandas, NumPy, scikit-learn, XGBoost, Tableau, Apache Spark, statistics, A/B testing
Experience
- Built predictive models with scikit-learn and XGBoost and ran A/B testing experiments.
- Created Tableau dashboards that tracked model performance for business teams.
Education
B.Tech in Computer Science, Pune University
"""
JOB = """We are hiring a Data Scientist to use Python, pandas, NumPy, scikit-learn, SQL and Tableau for data analysis, statistics, machine learning and A/B testing. Experience with Apache Spark and AWS is a plus."""

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))


def get(path, **kwargs):
    return requests.get(BASE + path, timeout=60, **kwargs)


def post_json(path, payload):
    return requests.post(BASE + path, json=payload, timeout=120)


index = get("/")
check("GET / 200", index.status_code == 200)
check("single dashboard", "Resumen" not in index.text and "tab-navigation" not in index.text)
check("no old five-tab nav", "Sentiment Analysis" not in index.text and "Topic Modelling" not in index.text)

health = get("/api/health").json()
check("health embedding model", health.get("embedding_model") == "all-MiniLM-L6-v2", str(health.get("embedding_model")))
check("health NER available", health.get("nlp", {}).get("ner_model") is True, json.dumps(health.get("nlp"))[:200])

jobs = get("/api/jobs").json()["jobs"]
check("5 curated jobs", len(jobs) == 5, str(len(jobs)))

links = get("/api/job-links", params={"title": "Data Scientist", "location": "Pune"}).json()
search_links = links.get("links") or links.get("job_links") or []
check("5 external job links", len(search_links) == 5, str(len(search_links)))
check("job links encoded", "Data%20Scientist" in json.dumps(links) or "Data+Scientist" in json.dumps(links))
check("job links genuine domains", all(any(d in link.get("url", "") for d in ("linkedin", "indeed", "naukri", "glassdoor", "wellfound")) for link in search_links))

match = requests.post(
    BASE + "/api/match",
    data={"job_description": JOB, "job_title": "Data Scientist", "location": "Pune"},
    files={"resumes": ("priya.txt", RESUME.encode(), "text/plain")},
    timeout=180,
)
match_body = match.json()
check("POST /api/match 200", match.status_code == 200, str(match_body.get("error")))
check("candidate returned", len(match_body.get("candidates", [])) == 1)
if match_body.get("candidates"):
    candidate = match_body["candidates"][0]
    check("scores present", all(k in candidate.get("scores", {}) for k in ("overall", "semantic", "tfidf", "skill_coverage")), json.dumps(candidate.get("scores"))[:200])
    check("entities present", bool(candidate.get("entities")))
    check("skills extracted", bool(candidate.get("matching_skills") or candidate.get("skills")))
    check("missing skills present", "missing_skills" in candidate or "skill_gap" in candidate)
    check("preview present", bool(candidate.get("extracted_text_preview") or candidate.get("preview")))

extract = post_json("/api/extract", {"text": RESUME, "job_description": JOB}).json()
check("POST /api/extract entities", bool(extract.get("entities")), json.dumps(extract)[:200])
check("extract skills", bool(extract.get("skills_by_category")) or extract.get("entities", {}).get("entity_counts", {}).get("skills", 0) > 0)

pipeline = post_json("/api/nlp/pipeline", {"text": RESUME, "job_description": JOB}).json()
check("pipeline tokens", bool(pipeline.get("tokens")))
check("pipeline lemmas", bool(pipeline.get("lemmas")))
check("pipeline vectors", bool(pipeline.get("vectorization", {}).get("embeddings")))
check("pipeline tfidf", bool(pipeline.get("vectorization", {}).get("tfidf")))
check("pipeline cosine", "cosine" in json.dumps(pipeline).lower())

extract_eval = get("/api/extract/evaluation").json()
check("extract evaluation endpoint", bool(extract_eval))

sentiment = post_json("/api/sentiment", {"text": "This is an excellent, well-structured resume with strong technical skills."}).json()
check("sentiment label", sentiment.get("label") in {"positive", "negative", "neutral"}, json.dumps(sentiment)[:200])
sent_eval = get("/api/sentiment/evaluation").json()
check("sentiment evaluation metrics", any(k in json.dumps(sent_eval).lower() for k in ("accuracy", "f1", "precision", "recall")))

sample = get("/api/topics/sample").json()
check("topic sample corpus", bool(sample.get("documents") or sample.get("corpus")))
docs = sample.get("documents") or sample.get("corpus") or []
topics = post_json("/api/topics", {"documents": docs[:10], "n_topics": 3}).json()
check("LDA topics generated", bool(topics.get("topics")), json.dumps(topics)[:200])
check("LDA keywords", bool(topics.get("topics", [{}])[0].get("keywords") or topics.get("topics", [{}])[0].get("words")))

ai_status = get("/api/ai/status").json()
check("Gemini configured", ai_status.get("gemini", {}).get("configured") is True, json.dumps(ai_status.get("gemini", {}))[:200])
check("Gemini tasks listed", len(ai_status.get("tasks", [])) == 4)

failed = [name for name, ok, _ in results if not ok]
for name, ok, detail in results:
    print(("PASS" if ok else "FAIL"), name, detail)
print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("FAILED:", ", ".join(failed))
