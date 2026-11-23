"""Resume-to-job matching: semantic similarity, TF-IDF baseline, skill coverage.

Scores (all reported on a 0-100 scale):

* ``semantic_raw``  - raw cosine similarity of MiniLM embeddings x 100.
* ``semantic``      - the same value rescaled so typical resume/job cosine
                      values (0.15 .. 0.70) span 0 .. 100 (in TF-IDF fallback
                      mode the TF-IDF scale is used instead).
* ``tfidf_raw``     - raw TF-IDF cosine x 100 (lexical baseline).
* ``tfidf``         - rescaled so a raw cosine of 0.50 maps to 100.
* ``skill_coverage``- share of the job's recognised skills found in the resume.
* ``overall``       - weighted blend of the calibrated scores.

If the embedding model cannot be loaded the semantic component falls back to
TF-IDF cosine and ``scoring.semantic_backend`` says so explicitly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingModel, shared_embedder
from .ner_extractor import DATA_DIR, EntityExtractor, SkillExtractor
from .preprocessing import clean_text, nlp_status, preprocess
from .skill_gap import analyze_skill_gap

SEMANTIC_LO, SEMANTIC_HI = 0.15, 0.70
TFIDF_HI = 0.50

DEFAULT_WEIGHTS = {"semantic": 0.50, "skills": 0.35, "tfidf": 0.15}


class MatchError(ValueError):
    """Raised for invalid matching input."""


@dataclass
class Resume:
    name: str
    text: str


def _rescale(value: float, lo: float, hi: float) -> float:
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


def verdict(score: float) -> str:
    if score >= 75:
        return "Excellent match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Moderate match"
    return "Weak match"


def load_sample_jobs(path: str | None = None) -> list[dict]:
    path = path or os.path.join(DATA_DIR, "sample_jobs.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class ResumeJobMatcher:
    def __init__(
        self,
        embedder: EmbeddingModel | None = None,
        entity_extractor: EntityExtractor | None = None,
        weights: dict | None = None,
        background_corpus: list[str] | None = None,
    ):
        self.embedder = embedder or shared_embedder()
        self.skills = SkillExtractor() if entity_extractor is None else entity_extractor.skills
        self.entities = entity_extractor or EntityExtractor(self.skills)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        if background_corpus is None:
            try:
                background_corpus = [j["description"] for j in load_sample_jobs()]
            except (OSError, ValueError, KeyError):
                background_corpus = []
        self.background = [preprocess(t) for t in background_corpus]

    # -- baselines ----------------------------------------------------------
    def tfidf_scores(self, resume_texts: list[str], job_text: str) -> list[float]:
        docs = [preprocess(job_text), *[preprocess(t) for t in resume_texts]]
        corpus = docs + self.background
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        try:
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:  # empty vocabulary
            return [0.0] * len(resume_texts)
        sims = cosine_similarity(matrix[0], matrix[1 : 1 + len(resume_texts)])[0]
        return [float(s) for s in sims]

    def semantic_scores(self, resume_texts: list[str], job_text: str) -> tuple[list[float], str]:
        if self.embedder.available:
            return self.embedder.similarity(resume_texts, job_text), self.embedder.backend
        return self.tfidf_scores(resume_texts, job_text), self.embedder.backend

    # -- public API ---------------------------------------------------------
    def analyze_job(self, job_text: str) -> dict:
        job_text = clean_text(job_text)
        skills = self.skills.extract(job_text)
        return {
            "skills": sorted(skills.values(), key=lambda s: (-s["count"], s["display"])),
            "skill_map": skills,
            "word_count": len(job_text.split()),
        }

    def rank(self, resumes: list[Resume], job_text: str, job_title: str | None = None) -> dict:
        """Score and rank every resume against ``job_text``."""
        job_text = clean_text(job_text)
        if len(job_text.split()) < 10:
            raise MatchError("Job description is too short (need at least 10 words).")
        if not resumes:
            raise MatchError("At least one resume is required.")

        texts = [clean_text(r.text) for r in resumes]
        job = self.analyze_job(job_text)
        sem_raw, sem_backend = self.semantic_scores(texts, job_text)
        tfidf_raw = self.tfidf_scores(texts, job_text)

        weights = dict(self.weights)
        if not job["skill_map"]:
            weights.pop("skills", None)
        total_w = sum(weights.values())

        candidates = []
        for resume, text, s_raw, t_raw in zip(resumes, texts, sem_raw, tfidf_raw):
            entities = self.entities.extract(text)
            resume_skills = {s["name"]: s for s in entities["skills"]}
            gap = analyze_skill_gap(resume_skills, job["skill_map"])

            if self.embedder.available:
                semantic = _rescale(s_raw, SEMANTIC_LO, SEMANTIC_HI) * 100
            else:  # fallback values are TF-IDF cosines, so use the TF-IDF scale
                semantic = _rescale(s_raw, 0.0, TFIDF_HI) * 100
            tfidf = _rescale(t_raw, 0.0, TFIDF_HI) * 100
            components = {"semantic": semantic, "tfidf": tfidf}
            if "skills" in weights:
                components["skills"] = gap["coverage"] or 0.0
            overall = sum(weights[k] * components[k] for k in weights) / total_w

            candidates.append(
                {
                    "filename": resume.name,
                    "scores": {
                        "overall": round(overall, 1),
                        "semantic": round(semantic, 1),
                        "semantic_raw": round(max(s_raw, 0.0) * 100, 1),
                        "tfidf": round(tfidf, 1),
                        "tfidf_raw": round(max(t_raw, 0.0) * 100, 1),
                        "skill_coverage": gap["coverage"],
                    },
                    "verdict": verdict(overall),
                    "entities": {k: v for k, v in entities.items() if k != "skills"},
                    "skills": [
                        {k: v for k, v in s.items() if k != "name"} | {"name": s["name"]}
                        for s in entities["skills"]
                    ],
                    "skill_gap": gap,
                    "word_count": len(text.split()),
                    "text_stats": {"characters": len(text), "words": len(text.split())},
                    "extracted_text_preview": text[:1500],
                }
            )

        candidates.sort(key=lambda c: (-c["scores"]["overall"], c["filename"].lower()))
        for position, cand in enumerate(candidates, start=1):
            cand["rank"] = position

        return {
            "job": {
                "title": job_title,
                "word_count": job["word_count"],
                "skills": job["skills"],
            },
            "scoring": {
                "weights": {k: round(v / total_w, 3) for k, v in weights.items()},
                "semantic_backend": sem_backend,
                "semantic_is_fallback": not self.embedder.available,
                "nlp_backend": nlp_status()["backend"],
                "ner_model_loaded": nlp_status()["ner_model"],
            },
            "candidates": candidates,
        }

    def match(self, resume_text: str, job_text: str, name: str = "resume") -> dict:
        """Convenience wrapper for a single resume."""
        return self.rank([Resume(name, resume_text)], job_text)["candidates"][0]
