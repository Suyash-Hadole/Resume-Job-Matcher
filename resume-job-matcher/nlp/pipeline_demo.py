"""A transparent, step-by-step demonstration of the local NLP pipeline.

The output is intentionally verbose so it can be shown in class:

1. raw extracted text and cleaned/normalised text,
2. tokens with lemma, POS tag and stop-word flags,
3. stop words removed and lemmas retained,
4. spaCy + regex named entities and technical skills,
5. TF-IDF document-term matrix details and top weighted terms,
6. MiniLM sentence-embedding dimension and cosine similarity (with an explicit
   TF-IDF fallback when the embedding model is unavailable),
7. skill-gap / matching results when a job description is supplied.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingModel, shared_embedder
from .ner_extractor import EntityExtractor
from .preprocessing import clean_text, get_nlp, lemmatize, nlp_status
from .skill_gap import analyze_skill_gap

MAX_DEMO_CHARS = 20_000
MAX_PREVIEW_CHARS = 3500
MAX_TOKENS = 90
MAX_TERMS = 15


class PipelineDemoError(ValueError):
    """Raised for invalid demonstration input."""


def _tfidf_block(resume_text: str, job_text: str) -> tuple[dict, float | None]:
    documents = [resume_text]
    if job_text:
        documents.append(job_text)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {
            "available": False,
            "note": "No usable vocabulary after stop-word removal.",
            "top_terms": [],
        }, None

    features = vectorizer.get_feature_names_out()
    row = np.asarray(matrix[0].todense()).ravel()
    top_indices = row.argsort()[::-1][:MAX_TERMS]
    top_terms = [
        {"term": str(features[i]), "weight": round(float(row[i]), 4)}
        for i in top_indices
        if row[i] > 0
    ]
    similarity = None
    if len(documents) == 2:
        similarity = float(cosine_similarity(matrix[0], matrix[1])[0][0])
    return {
        "available": True,
        "ngram_range": "1-2",
        "stop_words": "scikit-learn English stop list",
        "document_term_matrix": list(matrix.shape),
        "vocabulary_size": int(len(features)),
        "top_terms": top_terms,
        "description": (
            "Each row is a document and each column a 1-2 gram. TF-IDF rewards terms "
            "that are frequent in this document but rare across the supplied documents."
        ),
    }, similarity


def _embedding_block(resume_text: str, job_text: str, embedder: EmbeddingModel) -> dict:
    if not embedder.available:
        return {
            "available": False,
            "model": "all-MiniLM-L6-v2",
            "backend": embedder.backend,
            "dimension": None,
            "cosine_similarity": None,
            "note": (
                "The sentence-transformer model could not be loaded, so the demonstration "
                "reports the TF-IDF fallback instead of pretending embeddings were used."
            ),
        }
    vectors = embedder.encode([resume_text, job_text] if job_text else [resume_text])
    cosine = float(vectors[0] @ vectors[1]) if job_text else None
    return {
        "available": True,
        "model": "all-MiniLM-L6-v2",
        "backend": embedder.backend,
        "dimension": int(vectors.shape[1]),
        "cosine_similarity": round(cosine, 4) if cosine is not None else None,
        "note": (
            "Both documents are encoded into dense vectors and L2-normalised; "
            "cosine similarity is their dot product. Add a job description to compare two documents."
            if not job_text
            else "Both documents are encoded into dense vectors and L2-normalised; "
            "cosine similarity is their dot product."
        ),
    }


def run_pipeline_demo(
    text: str,
    job_text: str | None = None,
    embedder: EmbeddingModel | None = None,
    extractor: EntityExtractor | None = None,
) -> dict:
    cleaned = clean_text(text or "")
    if not cleaned:
        raise PipelineDemoError("Paste or upload some text to demonstrate the NLP pipeline.")
    if len(cleaned) > MAX_DEMO_CHARS:
        raise PipelineDemoError(f"Text is too long (maximum {MAX_DEMO_CHARS:,} characters).")

    job_clean = clean_text(job_text or "")
    extractor = extractor or EntityExtractor()
    embedder = embedder or shared_embedder()

    nlp = get_nlp()
    tokens = []
    stopwords_removed = []
    if nlp is not None:
        doc = nlp(cleaned[:MAX_DEMO_CHARS])
        for token in doc:
            if token.is_space:
                continue
            if token.is_stop:
                stopwords_removed.append(token.text.lower())
            if len(tokens) < MAX_TOKENS and not token.is_punct:
                tokens.append(
                    {
                        "text": token.text,
                        "lemma": token.lemma_ or token.text,
                        "pos": token.pos_,
                        "is_stop": bool(token.is_stop),
                        "is_alpha": bool(token.is_alpha),
                    }
                )

    entities = extractor.extract(cleaned)
    tfidf, tfidf_cosine = _tfidf_block(cleaned, job_clean)
    embedding = _embedding_block(cleaned, job_clean, embedder)
    if not embedding["available"]:
        embedding["cosine_similarity"] = round(tfidf_cosine, 4) if tfidf_cosine is not None else None

    result = {
        "text": {
            "original": cleaned[:MAX_PREVIEW_CHARS],
            "cleaned": cleaned[:MAX_PREVIEW_CHARS],
            "stats": {
                "characters": len(cleaned),
                "words": len(cleaned.split()),
                "sentences": len([s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]),
                "unique_lemmas": len(set(lemmatize(cleaned))),
            },
            "preview_truncated": len(cleaned) > MAX_PREVIEW_CHARS,
            "cleaning_steps": [
                "Unicode NFKC normalisation",
                "Control-character removal",
                "Bullet normalisation to '-'",
                "Hyphenated line-break repair",
                "Whitespace and blank-line collapse",
            ],
        },
        "tokens": tokens,
        "stopwords_removed": sorted(set(stopwords_removed))[:40],
        "lemmas": lemmatize(cleaned)[:MAX_TOKENS],
        "entities": entities,
        "vectorization": {
            "tfidf": tfidf,
            "embeddings": embedding,
        },
        "nlp_backend": nlp_status(),
    }

    if job_clean and len(job_clean.split()) >= 10:
        job_entities = extractor.extract(job_clean)
        resume_skills = {s["name"]: s for s in entities["skills"]}
        job_skills = {s["name"]: s for s in job_entities["skills"]}
        gap = analyze_skill_gap(resume_skills, job_skills)
        semantic_raw = embedding.get("cosine_similarity")
        result["matching"] = {
            "job_word_count": len(job_clean.split()),
            "job_skills": job_entities["skills"],
            "tfidf_cosine": round(tfidf_cosine, 4) if tfidf_cosine is not None else None,
            "semantic_cosine": semantic_raw,
            "semantic_backend": embedder.backend,
            "skill_coverage": gap["coverage"],
            "matching_skills": gap["matching"],
            "missing_skills": gap["missing"],
            "recommendations": gap["recommendations"],
        }
    else:
        result["matching"] = None

    return result
