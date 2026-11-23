"""Tests for structured entity and skill extraction."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.ie_evaluation import evaluate_extraction, load_cases  # noqa: E402
from nlp.ner_extractor import EntityExtractor, SkillExtractor  # noqa: E402

RESUME = """Ananya Iyer
ananya.iyer@example.com | +91 99887 76655 | linkedin.com/in/ananyaiyer
Data Engineer with 6 years of experience building data platforms.
Skills: Python, SQL, Spark, ReactJS, Node.js, scikit-learn, Docker, AWS
Experience
Data Engineer, Northwind Labs, Bengaluru (2018 - Present)
Education
B.Tech in Information Technology, Delhi University
"""


@pytest.fixture(scope="module")
def extractor():
    return EntityExtractor()


def test_contact_and_identity_fields(extractor):
    result = extractor.extract(RESUME)
    assert "Ananya" in (result["name"] or "")
    assert "ananya.iyer@example.com" in result["emails"]
    assert any("99887" in phone for phone in result["phones"])
    assert any("linkedin.com/in/ananyaiyer" in link for link in result["links"])
    assert result["persons"]


def test_dates_education_and_experience(extractor):
    result = extractor.extract(RESUME)
    assert result["dates"]
    assert any("2018" in date for date in result["dates"])
    assert any("b.tech" in degree.lower() for degree in result["education"])
    assert result["experience"]["years"] == 6


def test_skill_normalisation_handles_aliases(extractor):
    result = extractor.extract(RESUME)
    names = {skill["name"] for skill in result["skills"]}
    assert {"python", "sql", "react", "node.js", "scikit-learn", "docker", "aws"} <= names
    display = {skill["display"] for skill in result["skills"]}
    assert "React" in display
    assert "scikit-learn" in display


def test_skills_are_grouped_by_category(extractor):
    result = extractor.extract(RESUME)
    groups = result["skills_by_category"]
    assert groups
    assert all(group["category"] for group in groups)
    assert all(group["skills"] for group in groups)
    grouped = {skill["name"] for group in groups for skill in group["skills"]}
    assert grouped == {skill["name"] for skill in result["skills"]}


def test_entity_counts_match_lists(extractor):
    result = extractor.extract(RESUME)
    counts = result["entity_counts"]
    assert counts["emails"] == len(result["emails"])
    assert counts["phones"] == len(result["phones"])
    assert counts["links"] == len(result["links"])
    assert counts["skills"] == len(result["skills"])
    assert counts["persons"] == len(result["persons"])


def test_extraction_of_empty_text_is_safe(extractor):
    result = extractor.extract("")
    assert result["skills"] == []
    assert result["emails"] == []
    assert result["name"] is None
    assert result["entity_counts"]["skills"] == 0


def test_skill_extractor_does_not_fire_on_substrings():
    extractor = SkillExtractor()
    # "Go" and "C" must not match inside ordinary words.
    found = extractor.extract("I go to the office and see a cat.")
    assert "go" not in found
    assert "c" not in found


def test_ie_benchmark_has_annotated_cases():
    cases = load_cases()
    assert len(cases) >= 5
    assert all(case.get("text") for case in cases)
    assert all("expected" in case for case in cases)


def test_ie_evaluation_reports_real_metrics():
    report = evaluate_extraction()
    metrics = report["skill_metrics"]
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["true_positives"] > 0
    assert report["dataset_size"] == len(report["cases"])
    assert report["field_accuracy"]["checks"] > 0
    assert 0.0 <= report["field_accuracy"]["overall"] <= 1.0
    assert all(0.0 <= field["accuracy"] <= 1.0 for field in report["field_accuracy"]["fields"])
