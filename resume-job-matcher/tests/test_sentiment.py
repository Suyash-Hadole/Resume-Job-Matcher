"""Tests for VADER sentiment analysis and its labelled evaluation."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.sentiment_analyzer import (  # noqa: E402
    LABELS,
    NEGATIVE_THRESHOLD,
    POSITIVE_THRESHOLD,
    SentimentAnalyzer,
    SentimentError,
    classify,
    load_dataset,
    sentiment_available,
    sentiment_status,
)


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_dataset_is_large_enough_and_balanced():
    data = load_dataset()
    assert len(data["examples"]) >= 30
    counts = data["class_counts"]
    assert set(counts) == set(LABELS)
    assert all(count >= 10 for count in counts.values())
    assert max(counts.values()) - min(counts.values()) <= 2


def test_vader_is_available():
    assert sentiment_available() is True
    status = sentiment_status()
    assert status["available"] is True
    assert "VADER" in status["backend"]


def test_classify_thresholds():
    assert classify(POSITIVE_THRESHOLD) == "positive"
    assert classify(POSITIVE_THRESHOLD + 0.5) == "positive"
    assert classify(NEGATIVE_THRESHOLD) == "negative"
    assert classify(NEGATIVE_THRESHOLD - 0.5) == "negative"
    assert classify(0.0) == "neutral"
    assert classify(0.04) == "neutral"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I am thrilled and delighted with the excellent, wonderful result!", "positive"),
        ("This was a terrible, awful and deeply disappointing experience.", "negative"),
        ("The meeting is scheduled for Tuesday in room four.", "neutral"),
    ],
)
def test_analyze_labels(analyzer, text, expected):
    result = analyzer.analyze(text)
    assert result["label"] == expected


def test_analyze_returns_all_scores_and_evidence(analyzer):
    result = analyzer.analyze(
        "The project was great and the team was helpful, but the deadline was stressful."
    )
    assert set(result["scores"]) == {"positive", "negative", "neutral", "compound"}
    assert -1.0 <= result["scores"]["compound"] <= 1.0
    assert all(0.0 <= result["scores"][key] <= 1.0 for key in ("positive", "negative", "neutral"))
    assert result["text_stats"]["words"] > 0
    assert result["sentences"]
    assert all({"text", "scores", "label"} <= set(row) for row in result["sentences"])
    assert result["keywords"]
    assert "compound" in result["interpretation"]
    assert result["model"].startswith("VADER")
    assert "never used to rank candidates" in result["disclaimer"]


def test_analyze_rejects_empty_text(analyzer):
    with pytest.raises(SentimentError):
        analyzer.analyze("   ")
    with pytest.raises(SentimentError):
        analyzer.analyze("")


def test_evaluation_metrics_are_real_and_complete(analyzer):
    report = analyzer.evaluate()
    assert report["dataset_size"] >= 30
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["accuracy"] >= 0.8, "VADER should clearly beat chance on this dataset"

    assert set(report["macro"]) == {"precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in report["macro"].values())
    assert report["macro"]["f1"] >= 0.8

    assert len(report["per_class"]) == 3
    for row in report["per_class"]:
        assert row["label"] in LABELS
        assert row["support"] > 0
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0

    matrix = report["confusion_matrix"]
    assert matrix["labels"] == list(LABELS)
    assert len(matrix["matrix"]) == 3
    assert all(len(row) == 3 for row in matrix["matrix"])
    assert sum(sum(row) for row in matrix["matrix"]) == report["dataset_size"]

    assert len(report["predictions"]) == report["dataset_size"]
    assert all({"expected", "predicted", "compound", "correct"} <= set(p) for p in report["predictions"])
    assert sum(1 for p in report["predictions"] if p["correct"]) / len(report["predictions"]) == pytest.approx(
        report["accuracy"], abs=0.001
    )


def test_evaluation_labels_are_dataset_labels(analyzer):
    report = analyzer.evaluate()
    for prediction in report["predictions"]:
        assert prediction["expected"] in LABELS
        assert prediction["predicted"] in LABELS


def test_sentiment_output_is_not_a_ranking_signal(analyzer):
    """The sentiment payload must never contain candidate-ranking fields."""
    result = analyzer.analyze("A wonderful, excellent team.")

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    forbidden = {"overall", "rank", "ranking", "hiring_score", "candidate_score"}
    assert forbidden.isdisjoint(set(keys(result)))
    assert "score" not in result
