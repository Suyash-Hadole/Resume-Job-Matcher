"""Tests for the LDA topic-modelling pipeline."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.topic_modeling import (  # noqa: E402
    MAX_TOPICS,
    TopicModelError,
    corpus_description,
    load_sample_corpus,
    model_topics,
    topic_status,
)

EXTRA_DOCS = [
    "Python data scientist builds machine learning models with pandas and scikit-learn.",
    "Frontend engineer develops React and TypeScript interfaces for web applications.",
    "DevOps engineer automates cloud deployment with Docker, Kubernetes and CI pipelines.",
    "Product manager writes requirements and prioritises the roadmap with stakeholders.",
    "Database administrator tunes SQL queries, indexes and PostgreSQL backups.",
]


@pytest.fixture(scope="module")
def sample_corpus():
    return load_sample_corpus()


@pytest.fixture(scope="module")
def sample_model(sample_corpus):
    return model_topics(sample_corpus, n_topics=4)


def test_sample_corpus_is_realistic(sample_corpus):
    assert len(sample_corpus) >= 10
    assert all(doc.get("text") and len(doc["text"].split()) >= 20 for doc in sample_corpus)
    assert corpus_description()


def test_topic_status_is_available():
    status = topic_status()
    assert status["available"] is True
    assert "LDA" in status["backend"]
    assert status["sample_documents"] >= 10


def test_lda_returns_requested_topics(sample_model):
    assert sample_model["n_topics"] == 4
    assert len(sample_model["topics"]) == 4
    assert sample_model["documents_used"] == len(sample_model["documents"])
    assert sample_model["reliable"] is True
    assert sample_model["corpus"]["vocabulary_size"] > 20
    assert sample_model["corpus"]["document_term_matrix"][1] > 20


def test_topics_have_keywords_and_counts(sample_model):
    for topic in sample_model["topics"]:
        assert topic["topic"] >= 1
        assert topic["label"]
        assert len(topic["keywords"]) >= 3
        assert all(keyword["term"] for keyword in topic["keywords"])
        assert all(keyword["weight"] >= 0 for keyword in topic["keywords"])
        assert topic["document_count"] >= 0
    assert sum(topic["document_count"] for topic in sample_model["topics"]) == sample_model["documents_used"]


def test_document_topic_distributions_are_probabilities(sample_model):
    for document in sample_model["documents"]:
        weights = [row["weight"] for row in document["distribution"]]
        assert len(weights) == sample_model["n_topics"]
        assert sum(weights) == pytest.approx(1.0, abs=0.01)
        assert 1 <= document["top_topic"] <= sample_model["n_topics"]
        assert document["token_count"] > 0


def test_requested_topic_count_is_capped(sample_corpus):
    result = model_topics(sample_corpus, n_topics=99)
    assert result["requested_topics"] == 99
    assert result["n_topics"] <= MAX_TOPICS
    assert len(result["topics"]) == result["n_topics"]


def test_custom_documents_are_supported():
    result = model_topics(EXTRA_DOCS, n_topics=3)
    assert result["documents_used"] == len(EXTRA_DOCS)
    assert len(result["documents"]) == len(EXTRA_DOCS)
    assert len(result["topics"]) == 3


def test_too_few_documents_raise_a_clear_error():
    with pytest.raises(TopicModelError, match="at least two"):
        model_topics(["A single document about Python and SQL."])


def test_empty_documents_raise_a_clear_error():
    with pytest.raises(TopicModelError):
        model_topics([])
    with pytest.raises(TopicModelError):
        model_topics(["", "   "])


def test_stopword_only_documents_are_rejected():
    with pytest.raises(TopicModelError):
        model_topics(["the and of to in", "a is it for on with"])


def test_invalid_topic_count_is_rejected():
    docs = ["Python machine learning models", "React frontend web applications"]
    with pytest.raises(TopicModelError):
        model_topics(docs, n_topics=1)
    with pytest.raises(TopicModelError):
        model_topics(docs, n_topics="many")


def test_short_corpus_is_flagged_unreliable():
    docs = [
        "Python data analysis with pandas and numpy for reporting.",
        "React frontend components with TypeScript and CSS styling.",
    ]
    result = model_topics(docs, n_topics=2)
    assert result["reliable"] is False
    assert "exploratory" in result["reliability_note"].lower()


def test_too_many_documents_are_rejected():
    with pytest.raises(TopicModelError, match="Too many"):
        model_topics(["Python data"] * 51, n_topics=2)
