"""Automated tests for the resume-job matching system.

Fast tests use a deterministic bag-of-words ``FakeEmbedder`` so they do not
need the 90 MB MiniLM model. ``TestRealEmbedder`` exercises the real model and
is skipped automatically when it cannot be loaded.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from nlp.embeddings import EmbeddingModel, chunk_words  # noqa: E402
from nlp.ner_extractor import EntityExtractor, SkillExtractor  # noqa: E402
from nlp.preprocessing import clean_text, preprocess  # noqa: E402
from nlp.resume_parser import ResumeParseError, extract_text, validate_file  # noqa: E402
from nlp.semantic_matcher import MatchError, Resume, ResumeJobMatcher, load_sample_jobs  # noqa: E402
from nlp.skill_gap import analyze_skill_gap  # noqa: E402

DS_RESUME = """Priya Sharma
priya.sharma@example.com | +91 98765 43210 | github.com/priyasharma
Data Scientist with 5 years of experience building machine learning models.
Skills: Python, SQL, pandas, NumPy, scikit-learn, XGBoost, Tableau, Jupyter, Apache Spark, statistics, A/B testing
Experience
Data Scientist, Acme Analytics Pvt Ltd, Mumbai (2020 - Present)
- Built predictive models with scikit-learn and XGBoost and ran A/B testing experiments.
Education
B.Tech in Computer Science, Pune University
"""

WEB_RESUME = """Rahul Verma
rahul.verma@example.com
Full-stack developer. Skills: JavaScript, TypeScript, React, Node.js, Express, MongoDB, HTML, CSS, Git, Docker, REST APIs
Built responsive web apps with ReactJS and Node.js backed by MongoDB.
"""

DS_JOB = (
    "Data Scientist needed. You will use Python, pandas, numpy, scikit-learn, SQL and Tableau "
    "for data analysis, statistics, machine learning and A/B testing. Experience with Apache Spark "
    "and AWS is a plus."
)


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (no model download)."""

    available = True
    loaded = True
    error = None
    backend = "fake-test-embedder"
    DIM = 512

    def encode(self, texts):
        out = np.zeros((len(texts), self.DIM))
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                out[i, hash(tok) % self.DIM] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def similarity(self, resume_texts, job_text):
        vecs = self.encode([job_text, *resume_texts])
        return [float(v @ vecs[0]) for v in vecs[1:]]


class UnavailableEmbedder:
    available = False
    loaded = False
    error = "simulated failure"
    backend = "tfidf-fallback (sentence-transformers model unavailable)"


@pytest.fixture(scope="module")
def matcher():
    return ResumeJobMatcher(embedder=FakeEmbedder())


@pytest.fixture()
def client(matcher):
    app = create_app(matcher=matcher)
    app.config["TESTING"] = True
    return app.test_client()


def make_docx(text: str) -> bytes:
    import docx

    document = docx.Document()
    for line in text.split("\n"):
        document.add_paragraph(line)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def make_pdf(text: str) -> bytes:
    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.split("\n"):
        pdf.drawString(40, y, line[:110])
        y -= 16
        if y < 40:
            pdf.showPage()
            y = 800
    pdf.save()
    return buf.getvalue()


# ---------------------------------------------------------------- parsing ---
class TestResumeParser:
    def test_docx_roundtrip(self):
        text = extract_text("cv.docx", make_docx(DS_RESUME))
        assert "Priya Sharma" in text and "scikit-learn" in text

    def test_pdf_roundtrip(self):
        text = extract_text("cv.pdf", make_pdf(DS_RESUME))
        assert "Priya Sharma" in text and "XGBoost" in text

    def test_txt(self):
        assert "Rahul Verma" in extract_text("cv.txt", WEB_RESUME.encode())

    @pytest.mark.parametrize(
        "name,data,fragment",
        [
            ("cv.exe", b"MZ....", "Unsupported file type"),
            ("cv.pdf", b"", "empty"),
            ("cv.pdf", b"not a pdf at all", "valid PDF"),
            ("cv.docx", b"not a zip", "valid DOCX"),
            ("", b"data", "no name"),
        ],
    )
    def test_validation_errors(self, name, data, fragment):
        with pytest.raises(ResumeParseError) as exc:
            validate_file(name, data)
        assert fragment in str(exc.value)

    def test_too_large(self):
        with pytest.raises(ResumeParseError, match="too large"):
            validate_file("cv.txt", b"a" * (5 * 1024 * 1024 + 1))

    def test_nearly_empty_text(self):
        with pytest.raises(ResumeParseError, match="enough text"):
            extract_text("cv.txt", b"hi")

    def test_corrupt_pdf(self):
        with pytest.raises(ResumeParseError):
            extract_text("cv.pdf", b"%PDF-1.4 this is broken")


# ---------------------------------------------------------- preprocessing ---
class TestPreprocessing:
    def test_clean_text(self):
        cleaned = clean_text("• Python  developer\r\n\r\n\r\n\r\n● SQL")
        assert "\r" not in cleaned and "\n\n\n" not in cleaned
        assert "Python developer" in cleaned

    def test_preprocess_removes_stopwords(self):
        out = preprocess("The developers are building the applications")
        assert "the" not in out.split() and "are" not in out.split()
        assert out

    def test_empty(self):
        assert clean_text("") == "" and preprocess("") == ""


# ----------------------------------------------------------------- skills ---
class TestSkills:
    def setup_method(self):
        self.ex = SkillExtractor()

    def names(self, text):
        return set(self.ex.extract(text))

    def test_normalisation_of_aliases(self):
        found = self.names("Used ReactJS, node js, Postgres, sklearn and k8s daily")
        assert {"react", "node.js", "postgresql", "scikit-learn", "kubernetes"} <= found

    def test_special_characters(self):
        assert {"c++", "c#"} <= self.names("Languages: C++, C# and Python.")

    def test_no_false_positives_for_common_words(self):
        found = self.names("We go to the rest of the meeting. Our CV is attached. Spring is nice.")
        assert not ({"go", "r", "c", "rest api", "spring boot"} & found)

    def test_single_letter_languages_need_exact_case(self):
        assert {"r", "c"} <= self.names("Programming: Python, R, C, SQL")

    def test_counts_mentions(self):
        info = self.ex.extract("Python Python python")
        assert info["python"]["count"] == 3

    def test_display_names(self):
        info = self.ex.extract("aws and pytorch and nlp")
        assert info["aws"]["display"] == "AWS"
        assert info["pytorch"]["display"] == "PyTorch"
        assert info["natural language processing"]["display"] == "Natural Language Processing"


class TestEntities:
    def test_contact_and_education(self):
        ent = EntityExtractor().extract(DS_RESUME)
        assert ent["emails"] == ["priya.sharma@example.com"]
        assert any("98765" in p for p in ent["phones"])
        assert any("github.com/priyasharma" in link for link in ent["links"])
        assert any("tech" in d.lower() for d in ent["education"])
        assert ent["name"] == "Priya Sharma"
        assert ent["experience"]["years"] == 5

    def test_experience_from_date_ranges(self):
        text = "Engineer\nAcme (2018 - 2021)\nBeta Corp (2021 - 2023)\nEducation\nB.Tech 2014 - 2018"
        years = EntityExtractor().extract(text)["experience"]
        assert years["years"] == 5 and "estimated" in years["source"]

    def test_spacy_ner_when_model_available(self):
        from nlp.preprocessing import nlp_status

        if not nlp_status()["ner_model"]:
            pytest.skip("en_core_web_sm not installed")
        ent = EntityExtractor().extract(
            "Worked at Google and Microsoft in London and Bangalore.\n" + DS_RESUME
        )
        assert {"Google", "Microsoft"} & set(ent["organizations"])
        assert {"London", "Bangalore"} & set(ent["locations"])


# --------------------------------------------------------------- skill gap ---
class TestSkillGap:
    def test_gap_math_and_transferable(self):
        ex = SkillExtractor()
        job = ex.extract("Python PyTorch Docker AWS AWS SQL")
        resume = ex.extract("Python TensorFlow Docker Azure Git")
        gap = analyze_skill_gap(resume, job)
        assert {m["skill"] for m in gap["matching"]} == {"python", "docker"}
        assert {m["skill"] for m in gap["missing"]} == {"pytorch", "aws", "sql"}
        assert {e["skill"] for e in gap["extra"]} == {"tensorflow", "azure", "git"}
        assert gap["coverage"] == pytest.approx(40.0)
        pytorch = next(m for m in gap["missing"] if m["skill"] == "pytorch")
        assert pytorch["transferable_from"] == ["TensorFlow"]
        aws = next(m for m in gap["missing"] if m["skill"] == "aws")
        assert aws["priority"] == "high"
        assert gap["recommendations"]

    def test_full_coverage(self):
        ex = SkillExtractor()
        skills = ex.extract("Python SQL")
        gap = analyze_skill_gap(skills, skills)
        assert gap["coverage"] == 100.0 and not gap["missing"]

    def test_no_job_skills(self):
        gap = analyze_skill_gap({}, {})
        assert gap["coverage"] is None and gap["recommendations"]


# ----------------------------------------------------------------- matcher ---
class TestMatcher:
    def test_relevant_resume_ranks_first(self, matcher):
        result = matcher.rank([Resume("web.txt", WEB_RESUME), Resume("ds.txt", DS_RESUME)], DS_JOB)
        names = [c["filename"] for c in result["candidates"]]
        assert names == ["ds.txt", "web.txt"]
        assert [c["rank"] for c in result["candidates"]] == [1, 2]
        top, bottom = result["candidates"]
        assert top["scores"]["overall"] > bottom["scores"]["overall"]
        assert top["scores"]["skill_coverage"] > bottom["scores"]["skill_coverage"]

    def test_scores_in_range_and_shape(self, matcher):
        cand = matcher.match(DS_RESUME, DS_JOB)
        for key in ("overall", "semantic", "tfidf"):
            assert 0 <= cand["scores"][key] <= 100
        assert cand["skill_gap"]["matching"] and "entities" in cand
        assert cand["verdict"]

    def test_identical_text_scores_high_on_tfidf(self, matcher):
        cand = matcher.match(DS_JOB, DS_JOB)
        assert cand["scores"]["tfidf_raw"] > 80 and cand["scores"]["skill_coverage"] == 100.0

    def test_weights_sum_to_one(self, matcher):
        result = matcher.rank([Resume("a", DS_RESUME)], DS_JOB)
        assert sum(result["scoring"]["weights"].values()) == pytest.approx(1.0, abs=0.01)

    def test_job_without_skills_drops_skill_weight(self, matcher):
        job = "We are looking for a friendly person to greet customers and manage the front desk every day."
        result = matcher.rank([Resume("a", DS_RESUME)], job)
        assert "skills" not in result["scoring"]["weights"]
        assert result["candidates"][0]["skill_gap"]["coverage"] is None

    def test_validation(self, matcher):
        with pytest.raises(MatchError):
            matcher.rank([Resume("a", DS_RESUME)], "too short")
        with pytest.raises(MatchError):
            matcher.rank([], DS_JOB)

    def test_fallback_is_reported_honestly(self):
        m = ResumeJobMatcher(embedder=UnavailableEmbedder())
        result = m.rank([Resume("a", DS_RESUME)], DS_JOB)
        assert result["scoring"]["semantic_is_fallback"] is True
        assert "fallback" in result["scoring"]["semantic_backend"]
        assert 0 <= result["candidates"][0]["scores"]["overall"] <= 100


class TestRegressions:
    def test_fallback_uses_tfidf_scale(self):
        """Fallback semantic score must not be squashed by the MiniLM scale."""
        m = ResumeJobMatcher(embedder=UnavailableEmbedder())
        cand = m.rank([Resume("a", DS_RESUME)], DS_JOB)["candidates"][0]
        expected = min(100.0, cand["scores"]["semantic_raw"] / 50 * 100)
        assert cand["scores"]["semantic"] == pytest.approx(expected, abs=0.2)

    def test_skill_lists_and_fields_are_not_organizations(self):
        text = "Jane Roe\nTools: Git, Docker\nB.Tech in Computer Science\nWorked at Acme Analytics Pvt Ltd for years."
        orgs = EntityExtractor().extract(text)["organizations"]
        assert "Git, Docker" not in orgs and "Computer Science" not in orgs
        assert any("Acme Analytics" in o for o in orgs)


class TestSampleJobs:
    def test_required_roles_present_and_have_skills(self):
        jobs = load_sample_jobs()
        titles = {j["title"] for j in jobs}
        for expected in ("Python Developer", "Data Scientist", "NLP Engineer", "Full-Stack Developer"):
            assert expected in titles
        assert any("Machine Learning Engineer" in t or "ML Engineer" in t for t in titles)
        ex = SkillExtractor()
        for job in jobs:
            assert len(job["description"].split()) >= 50
            assert len(ex.extract(job["description"])) >= 8, job["title"]


class TestChunking:
    def test_short_text_single_chunk(self):
        assert chunk_words("one two three") == ["one two three"]

    def test_long_text_overlaps_and_covers(self):
        words = [f"w{i}" for i in range(500)]
        chunks = chunk_words(" ".join(words), max_words=100, overlap=20)
        assert len(chunks) > 4
        assert chunks[0].split()[0] == "w0" and chunks[-1].split()[-1] == "w499"
        assert chunks[1].split()[0] == "w80"


# --------------------------------------------------------------------- API ---
class TestAPI:
    def post(self, client, files=None, **form):
        data = dict(form)
        if files is not None:
            data["resumes"] = [(io.BytesIO(content), name) for name, content in files]
        return client.post("/api/match", data=data, content_type="multipart/form-data")

    def test_index_and_health_and_jobs(self, client):
        assert client.get("/").status_code == 200
        health = client.get("/api/health").get_json()
        assert health["status"] == "ok" and health["embedding_model"] == "all-MiniLM-L6-v2"
        assert len(client.get("/api/jobs").get_json()["jobs"]) == 5

    def test_match_multiple_files_with_sample_job(self, client):
        resp = self.post(
            client,
            files=[("web.docx", make_docx(WEB_RESUME)), ("ds.pdf", make_pdf(DS_RESUME)), ("ds.txt", DS_RESUME.encode())],
            job_id="data-scientist",
        )
        body = resp.get_json()
        assert resp.status_code == 200, body
        assert body["job"]["title"] == "Data Scientist"
        assert len(body["candidates"]) == 3
        assert body["candidates"][-1]["filename"] == "web.docx"
        assert body["file_errors"] == []

    def test_custom_job_description(self, client):
        resp = self.post(client, files=[("ds.txt", DS_RESUME.encode())], job_description=DS_JOB)
        assert resp.status_code == 200
        assert resp.get_json()["job"]["title"] == "Custom job description"

    def test_partial_failure_reports_file_errors(self, client):
        resp = self.post(
            client,
            files=[("ds.txt", DS_RESUME.encode()), ("bad.pdf", b"nope"), ("x.exe", b"MZ")],
            job_id="python-developer",
        )
        body = resp.get_json()
        assert resp.status_code == 200
        assert {e["filename"] for e in body["file_errors"]} == {"bad.pdf", "x.exe"}
        assert len(body["candidates"]) == 1

    def test_all_files_invalid(self, client):
        resp = self.post(client, files=[("bad.pdf", b"nope")], job_id="python-developer")
        assert resp.status_code == 422 and resp.get_json()["file_errors"]

    @pytest.mark.parametrize(
        "kwargs,status",
        [
            ({"job_id": "python-developer"}, 400),                       # no files
            ({"files": [("a.txt", DS_RESUME.encode())]}, 400),           # no job
            ({"files": [("a.txt", DS_RESUME.encode())], "job_id": "nope"}, 400),
            ({"files": [("a.txt", DS_RESUME.encode())], "job_description": "too short"}, 400),
        ],
    )
    def test_validation_errors(self, client, kwargs, status):
        resp = self.post(client, **kwargs)
        assert resp.status_code == status and "error" in resp.get_json()

    def test_too_many_files(self, client):
        files = [(f"r{i}.txt", DS_RESUME.encode()) for i in range(11)]
        resp = self.post(client, files=files, job_id="python-developer")
        assert resp.status_code == 400 and "Too many" in resp.get_json()["error"]

    def test_duplicate_filenames(self, client):
        files = [("a.txt", DS_RESUME.encode()), ("a.txt", WEB_RESUME.encode())]
        body = self.post(client, files=files, job_id="python-developer").get_json()
        assert len(body["candidates"]) == 1 and "Duplicate" in body["file_errors"][0]["error"]

    def test_unknown_route_returns_json(self, client):
        resp = client.get("/api/nope")
        assert resp.status_code == 404 and "error" in resp.get_json()


# ---------------------------------------------------- real embedding model ---
@pytest.fixture(scope="module")
def model():
    m = EmbeddingModel()
    if not m.available:
        pytest.skip(f"MiniLM model unavailable: {m.error}")
    return m


class TestRealEmbedder:

    def test_embedding_shape_and_norm(self, model):
        vecs = model.encode(["python developer", "chef who cooks pasta"])
        assert vecs.shape == (2, 384)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)

    def test_semantic_ordering(self, model):
        sims = model.similarity(
            ["Experienced Python backend engineer building REST APIs", "Professional pastry chef baking bread and cakes"],
            "Looking for a Python developer to build web services",
        )
        assert sims[0] > sims[1]

    def test_end_to_end_with_real_model(self, model):
        m = ResumeJobMatcher(embedder=model)
        result = m.rank([Resume("web.txt", WEB_RESUME), Resume("ds.txt", DS_RESUME)], DS_JOB)
        assert result["candidates"][0]["filename"] == "ds.txt"
        assert result["scoring"]["semantic_is_fallback"] is False
