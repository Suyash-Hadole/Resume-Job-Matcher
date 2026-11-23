"""VADER sentiment analysis with a reproducible labelled evaluation.

This module is deliberately independent of the candidate-ranking pipeline.
Sentiment describes the *tone of a piece of text* (a job review, recruiter
feedback, cover-letter paragraph, ...). It is never used to score candidates,
infer personality, or make hiring decisions.

The evaluation uses ``data/sentiment_dataset.json``: 36 manually labelled
sentences, balanced across negative / neutral / positive. Metrics are computed
with scikit-learn at request time, so the dashboard always shows real numbers
for the current analyser rather than hard-coded values.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .ner_extractor import DATA_DIR
from .preprocessing import clean_text, sentences

MAX_TEXT_CHARS = 20_000
MAX_SENTENCES = 120
DATASET_PATH = os.path.join(DATA_DIR, "sentiment_dataset.json")
LABELS = ("negative", "neutral", "positive")
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

DISCLAIMER = (
    "Sentiment describes the tone of the supplied text only. It is not a "
    "personality assessment and is never used to rank candidates or make "
    "hiring decisions."
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


class SentimentError(ValueError):
    """Raised when sentiment analysis cannot be performed."""


@lru_cache(maxsize=1)
def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:  # pragma: no cover - dependency problem
        raise SentimentError(
            "Sentiment analysis requires the 'vaderSentiment' package. "
            "Install it with: pip install vaderSentiment"
        ) from exc
    return SentimentIntensityAnalyzer()


def sentiment_available() -> bool:
    try:
        _vader()
        return True
    except SentimentError:
        return False


def classify(compound: float) -> str:
    """Map a VADER compound score to the standard three-way label."""
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def _round_scores(raw: dict) -> dict:
    return {
        "positive": round(float(raw.get("pos", 0.0)), 4),
        "negative": round(float(raw.get("neg", 0.0)), 4),
        "neutral": round(float(raw.get("neu", 0.0)), 4),
        "compound": round(float(raw.get("compound", 0.0)), 4),
    }


def _keyword_hits(text: str) -> list[dict]:
    """Return lexicon words that actually occur in the text (with valence)."""
    analyzer = _vader()
    lexicon = analyzer.lexicon
    seen: dict[str, dict] = {}
    for match in _WORD_RE.finditer(text.lower()):
        word = match.group(0)
        valence = lexicon.get(word)
        if valence is None:
            continue
        entry = seen.setdefault(
            word,
            {
                "word": word,
                "valence": round(float(valence), 2),
                "label": "positive" if valence > 0 else "negative",
                "count": 0,
            },
        )
        entry["count"] += 1
    return sorted(seen.values(), key=lambda e: (-abs(e["valence"]), -e["count"], e["word"]))[:12]


@lru_cache(maxsize=1)
def load_dataset(path: str = DATASET_PATH) -> dict:
    """Load and validate the manually labelled evaluation dataset."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SentimentError(f"Could not load the sentiment dataset: {exc}") from exc

    examples = data.get("examples") or []
    if len(examples) < 30:
        raise SentimentError("The sentiment dataset must contain at least 30 labelled examples.")
    counts = {label: 0 for label in LABELS}
    for item in examples:
        label = str(item.get("label", "")).lower()
        if label not in LABELS:
            raise SentimentError(f"Invalid dataset label '{item.get('label')}'.")
        if not str(item.get("text", "")).strip():
            raise SentimentError("The sentiment dataset contains an empty sentence.")
        counts[label] += 1
    data["examples"] = examples
    data["class_counts"] = counts
    return data


class SentimentAnalyzer:
    """Thin, stateless wrapper around VADER plus dataset evaluation."""

    model_name = "VADER (vaderSentiment)"

    def __init__(self) -> None:
        self._analyzer = _vader()

    def analyze(self, text: str) -> dict:
        cleaned = clean_text(text or "")
        if not cleaned:
            raise SentimentError("Enter some text to analyse.")
        if len(cleaned) > MAX_TEXT_CHARS:
            raise SentimentError(f"Text is too long (maximum {MAX_TEXT_CHARS:,} characters).")

        raw = self._analyzer.polarity_scores(cleaned)
        scores = _round_scores(raw)
        label = classify(scores["compound"])

        sentence_rows = []
        for sentence in sentences(cleaned)[:MAX_SENTENCES]:
            sentence_scores = _round_scores(self._analyzer.polarity_scores(sentence))
            sentence_rows.append(
                {
                    "text": sentence,
                    "scores": sentence_scores,
                    "label": classify(sentence_scores["compound"]),
                }
            )

        words = len(cleaned.split())
        return {
            "label": label,
            "scores": scores,
            "text_stats": {
                "characters": len(cleaned),
                "words": words,
                "sentences": len(sentence_rows),
            },
            "sentences": sentence_rows,
            "keywords": _keyword_hits(cleaned),
            "interpretation": self._interpret(label, scores, words),
            "model": self.model_name,
            "thresholds": {
                "positive": POSITIVE_THRESHOLD,
                "negative": NEGATIVE_THRESHOLD,
            },
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _interpret(label: str, scores: dict, words: int) -> str:
        compound = scores["compound"]
        if label == "positive":
            tone = "predominantly positive"
        elif label == "negative":
            tone = "predominantly negative"
        else:
            tone = "neutral / mixed with no strong polarity"
        return (
            f"The text is {tone} (compound {compound:+.3f}; "
            f"positive {scores['positive']:.3f}, negative {scores['negative']:.3f}, "
            f"neutral {scores['neutral']:.3f}) across {words} words."
        )

    def evaluate(self) -> dict:
        """Run the labelled dataset through VADER and compute real metrics."""
        data = load_dataset()
        examples = data["examples"]
        y_true = [str(e["label"]).lower() for e in examples]
        y_pred = [classify(self._analyzer.polarity_scores(e["text"])["compound"]) for e in examples]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(LABELS), zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0
        )
        matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))

        per_class = [
            {
                "label": label,
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        ]
        predictions = [
            {
                "id": item.get("id"),
                "text": item["text"],
                "category": item.get("category", ""),
                "expected": true,
                "predicted": pred,
                "compound": round(
                    float(self._analyzer.polarity_scores(item["text"])["compound"]), 4
                ),
                "correct": true == pred,
            }
            for item, true, pred in zip(examples, y_true, y_pred)
        ]

        return {
            "model": self.model_name,
            "dataset_size": len(examples),
            "class_counts": data.get("class_counts", {}),
            "dataset_description": data.get("description", ""),
            "labels": list(LABELS),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro": {
                "precision": round(float(macro_p), 4),
                "recall": round(float(macro_r), 4),
                "f1": round(float(macro_f1), 4),
            },
            "per_class": per_class,
            "confusion_matrix": {
                "labels": list(LABELS),
                "matrix": matrix.tolist(),
                "rows": "expected label",
                "columns": "predicted label",
            },
            "predictions": predictions,
            "disclaimer": DISCLAIMER,
        }


def sentiment_status() -> dict:
    try:
        _vader()
        return {
            "available": True,
            "backend": "VADER (vaderSentiment)",
            "dataset_examples": len(load_dataset()["examples"]),
        }
    except SentimentError as exc:
        return {"available": False, "backend": "VADER unavailable", "error": str(exc)}
