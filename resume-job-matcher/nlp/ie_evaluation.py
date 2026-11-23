"""Small labelled benchmark for information and skill extraction.

The dashboard can show real precision / recall / F1 for skill extraction plus
per-field accuracy for contact, education and experience extraction. The
annotations live in ``data/ie_eval_cases.json`` and are compared with the
current :class:`~nlp.ner_extractor.EntityExtractor` output at request time.
"""

from __future__ import annotations

import json
import os
import re

from .ner_extractor import DATA_DIR, EntityExtractor

CASES_PATH = os.path.join(DATA_DIR, "ie_eval_cases.json")


class IEEvaluationError(ValueError):
    """Raised when the extraction benchmark cannot be loaded or run."""


def load_cases(path: str = CASES_PATH) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise IEEvaluationError(f"Could not load the extraction benchmark: {exc}") from exc
    cases = data.get("cases") or []
    if not cases:
        raise IEEvaluationError("The extraction benchmark contains no cases.")
    return cases


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def evaluate_extraction(extractor: EntityExtractor | None = None) -> dict:
    extractor = extractor or EntityExtractor()
    cases = load_cases()

    tp = fp = fn = 0
    field_totals: dict[str, dict[str, int]] = {}
    rows = []

    def field_result(name: str, correct: bool) -> None:
        bucket = field_totals.setdefault(name, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(correct)

    for case in cases:
        expected = case.get("expected") or {}
        actual = extractor.extract(case["text"])
        predicted_skills = {s["name"] for s in actual["skills"]}
        expected_skills = set(expected.get("skills") or [])
        case_tp = len(predicted_skills & expected_skills)
        case_fp = len(predicted_skills - expected_skills)
        case_fn = len(expected_skills - predicted_skills)
        tp += case_tp
        fp += case_fp
        fn += case_fn

        checks = {}
        if "name_contains" in expected:
            name = actual.get("name") or ""
            checks["name"] = expected["name_contains"].lower() in name.lower()
        if "emails" in expected:
            checks["emails"] = set(actual.get("emails") or []) == set(expected["emails"])
        if "phones_contain" in expected:
            actual_digits = " ".join(_digits(p) for p in actual.get("phones") or [])
            checks["phones"] = all(_digits(p) in actual_digits for p in expected["phones_contain"])
        if "links_contain" in expected:
            links = " ".join(actual.get("links") or []).lower()
            checks["links"] = all(link.lower() in links for link in expected["links_contain"])
        if "education_contains" in expected:
            education = " ".join(actual.get("education") or []).lower()
            checks["education"] = all(term.lower() in education for term in expected["education_contains"])
        if "experience_years" in expected:
            checks["experience_years"] = actual.get("experience", {}).get("years") == expected["experience_years"]

        for field, correct in checks.items():
            field_result(field, correct)

        rows.append(
            {
                "id": case.get("id"),
                "predicted_skills": sorted(predicted_skills),
                "expected_skills": sorted(expected_skills),
                "skill_true_positives": case_tp,
                "skill_false_positives": sorted(predicted_skills - expected_skills),
                "skill_false_negatives": sorted(expected_skills - predicted_skills),
                "fields": checks,
                "extracted": {
                    "name": actual.get("name"),
                    "emails": actual.get("emails"),
                    "phones": actual.get("phones"),
                    "links": actual.get("links"),
                    "education": actual.get("education"),
                    "experience": actual.get("experience"),
                },
            }
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fields = [
        {
            "field": field,
            "correct": values["correct"],
            "total": values["total"],
            "accuracy": round(values["correct"] / values["total"], 4) if values["total"] else 0.0,
        }
        for field, values in sorted(field_totals.items())
    ]
    total_correct = sum(f["correct"] for f in fields)
    total_checks = sum(f["total"] for f in fields)

    return {
        "dataset_size": len(cases),
        "skill_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "definition": "Micro-averaged over all cases; predicted canonical skills are compared with manual annotations.",
        },
        "field_accuracy": {
            "overall": round(total_correct / total_checks, 4) if total_checks else 0.0,
            "correct": total_correct,
            "checks": total_checks,
            "fields": fields,
        },
        "cases": rows,
        "notes": (
            "The benchmark is intentionally small and is used for classroom evaluation, "
            "not as a claim of production accuracy. Dictionary skills that are not annotated "
            "in a case count as false positives."
        ),
    }
