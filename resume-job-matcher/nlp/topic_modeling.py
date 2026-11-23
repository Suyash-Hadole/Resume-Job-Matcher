"""Latent Dirichlet Allocation (LDA) topic modelling with scikit-learn.

Pipeline: clean -> spaCy lemmatisation / stop-word removal -> CountVectorizer
(document-term matrix) -> LatentDirichletAllocation -> topic keywords and
per-document topic distributions.

Small corpora are handled honestly: results are returned (when mathematically
possible) but flagged ``reliable = False`` with an explanation. The module never
claims that two or three documents produce meaningful themes.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .ner_extractor import DATA_DIR
from .preprocessing import clean_text, preprocess

CORPUS_PATH = os.path.join(DATA_DIR, "topic_corpus.json")
MAX_DOCS = 50
MAX_DOC_CHARS = 20_000
MAX_TOTAL_CHARS = 300_000
MAX_TOPICS = 10
MIN_DOCS_FOR_MEANINGFUL = 4
MIN_TOKENS_FOR_MEANINGFUL = 100
MIN_VOCAB_FOR_MEANINGFUL = 20
TOP_KEYWORDS = 8


class TopicModelError(ValueError):
    """Raised when the corpus cannot be modelled."""


def load_sample_corpus(path: str = CORPUS_PATH) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicModelError(f"Could not load the sample topic corpus: {exc}") from exc
    documents = data.get("documents") or []
    if not documents:
        raise TopicModelError("The sample topic corpus is empty.")
    return documents


def corpus_description(path: str = CORPUS_PATH) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("description", "")
    except (OSError, json.JSONDecodeError):
        return ""


def _normalise_documents(documents) -> tuple[list[dict], list[dict]]:
    """Validate input, drop empty documents and report what was skipped."""
    if not isinstance(documents, (list, tuple)):
        raise TopicModelError("'documents' must be a list of texts or {title, text} objects.")
    if len(documents) > MAX_DOCS:
        raise TopicModelError(f"Too many documents (maximum {MAX_DOCS}).")

    clean_docs: list[dict] = []
    skipped: list[dict] = []
    total_chars = 0
    for index, item in enumerate(documents):
        if isinstance(item, str):
            title, text, category = f"Document {index + 1}", item, "custom"
        elif isinstance(item, dict):
            title = str(item.get("title") or f"Document {index + 1}").strip()[:160]
            text = str(item.get("text") or "")
            category = str(item.get("category") or "custom")[:40]
        else:
            raise TopicModelError("Each document must be a string or an object with a 'text' field.")

        if len(text) > MAX_DOC_CHARS:
            raise TopicModelError(
                f"Document '{title}' is too long (maximum {MAX_DOC_CHARS:,} characters)."
            )
        total_chars += len(text)
        if total_chars > MAX_TOTAL_CHARS:
            raise TopicModelError(f"Corpus is too large (maximum {MAX_TOTAL_CHARS:,} characters).")

        text = clean_text(text)
        if not text:
            skipped.append({"title": title, "reason": "empty text"})
            continue
        clean_docs.append({"title": title, "text": text, "category": category})

    if len(clean_docs) < 2:
        raise TopicModelError(
            "LDA needs at least two non-empty documents. Add more documents to the corpus."
        )
    return clean_docs, skipped


def model_topics(documents, n_topics: int = 5) -> dict:
    """Run the full LDA pipeline and return topics + document distributions."""
    try:
        requested = int(n_topics)
    except (TypeError, ValueError) as exc:
        raise TopicModelError("'n_topics' must be an integer.") from exc
    if requested < 2:
        raise TopicModelError("Choose at least 2 topics.")

    docs, skipped = _normalise_documents(documents)
    processed = [preprocess(doc["text"]) for doc in docs]
    empty_after = [doc["title"] for doc, text in zip(docs, processed) if not text.strip()]
    if empty_after:
        skipped.extend({"title": title, "reason": "no usable tokens after preprocessing"} for title in empty_after)
        keep = [(doc, text) for doc, text in zip(docs, processed) if text.strip()]
        docs = [doc for doc, _ in keep]
        processed = [text for _, text in keep]
    if len(docs) < 2:
        raise TopicModelError(
            "After cleaning and stop-word removal fewer than two documents contain usable tokens."
        )

    total_tokens = sum(len(text.split()) for text in processed)
    max_df = 0.95 if len(docs) >= 8 else 1.0
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=max_df,
        max_features=5000,
    )
    try:
        matrix = vectorizer.fit_transform(processed)
    except ValueError as exc:
        raise TopicModelError(
            "The corpus has no usable vocabulary after preprocessing. Add longer documents."
        ) from exc

    vocab = vectorizer.get_feature_names_out()
    if len(vocab) < 2:
        raise TopicModelError("The corpus vocabulary is too small to fit topics.")

    actual_topics = max(2, min(requested, MAX_TOPICS, len(docs) - 1, len(vocab)))
    # With very few documents, sklearn requires n_components <= n_samples.
    actual_topics = min(actual_topics, len(docs))

    lda = LatentDirichletAllocation(
        n_components=actual_topics,
        random_state=42,
        learning_method="batch",
        max_iter=25,
        n_jobs=1,
    )
    doc_topic = lda.fit_transform(matrix)
    doc_topic = doc_topic / np.maximum(doc_topic.sum(axis=1, keepdims=True), 1e-12)

    topics = []
    for topic_idx, component in enumerate(lda.components_):
        total = float(component.sum()) or 1.0
        top_indices = component.argsort()[::-1][:TOP_KEYWORDS]
        keywords = [
            {
                "term": str(vocab[i]),
                "weight": round(float(component[i] / total), 5),
            }
            for i in top_indices
        ]
        label = " / ".join(k["term"] for k in keywords[:2])
        topics.append(
            {
                "topic": topic_idx + 1,
                "label": label,
                "keywords": keywords,
            }
        )

    # Document-level distributions, top-topic assignments and per-topic document counts.
    assignments = doc_topic.argmax(axis=1)
    counts = Counter(int(a) for a in assignments)
    for topic in topics:
        topic["document_count"] = int(counts.get(topic["topic"] - 1, 0))
        topic["share"] = round(100 * topic["document_count"] / len(docs), 1)

    document_rows = []
    for doc, text, distribution, assignment in zip(docs, processed, doc_topic, assignments):
        row = [
            {"topic": i + 1, "weight": round(float(weight), 4)}
            for i, weight in enumerate(distribution)
        ]
        top = max(row, key=lambda r: r["weight"])
        document_rows.append(
            {
                "title": doc["title"],
                "category": doc["category"],
                "top_topic": top["topic"],
                "top_weight": top["weight"],
                "distribution": row,
                "preview": doc["text"][:220] + ("…" if len(doc["text"]) > 220 else ""),
                "token_count": len(text.split()),
            }
        )

    reliable = (
        len(docs) >= MIN_DOCS_FOR_MEANINGFUL
        and total_tokens >= MIN_TOKENS_FOR_MEANINGFUL
        and len(vocab) >= MIN_VOCAB_FOR_MEANINGFUL
    )
    if reliable:
        note = (
            f"{len(docs)} documents and {total_tokens} tokens provide enough evidence "
            "for an exploratory LDA model. Topic labels are still descriptive, not ground truth."
        )
    elif len(docs) < MIN_DOCS_FOR_MEANINGFUL:
        note = (
            f"Only {len(docs)} non-empty document(s) were provided. LDA is mathematically "
            f"possible but unstable below {MIN_DOCS_FOR_MEANINGFUL} documents. Treat these "
            "topics as exploratory output, not meaningful themes."
        )
    else:
        note = (
            f"The corpus has {total_tokens} tokens and {len(vocab)} vocabulary terms, which is "
            "below the threshold for a stable LDA model. Treat the topics as exploratory."
        )

    top_terms = [
        {"term": str(vocab[i]), "count": int(matrix[:, i].sum())}
        for i in np.argsort(np.asarray(matrix.sum(axis=0)).ravel())[::-1][:15]
    ]

    return {
        "model": "scikit-learn LatentDirichletAllocation + CountVectorizer",
        "requested_topics": requested,
        "n_topics": actual_topics,
        "documents_used": len(docs),
        "documents_skipped": skipped,
        "corpus": {
            "total_tokens": total_tokens,
            "vocabulary_size": int(len(vocab)),
            "document_term_matrix": list(matrix.shape),
            "ngram_range": [1, 2],
            "preprocessing": "Unicode cleaning, spaCy lemmatisation, stop-word removal, 1-2 grams",
        },
        "reliable": reliable,
        "reliability_note": note,
        "topics": topics,
        "documents": document_rows,
        "top_terms": top_terms,
        "perplexity": round(float(lda.perplexity(matrix)), 2),
        "notes": (
            "LDA assigns every document a probability distribution over topics. "
            "Higher weight means the document uses more words associated with that topic."
        ),
    }


def topic_status() -> dict:
    try:
        corpus = load_sample_corpus()
        return {
            "available": True,
            "backend": "scikit-learn LDA + CountVectorizer",
            "sample_documents": len(corpus),
        }
    except TopicModelError as exc:
        return {"available": False, "backend": "LDA unavailable", "error": str(exc)}
