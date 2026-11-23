"""Skill-gap analysis: compare resume skills against a job's required skills."""

from __future__ import annotations

from collections import defaultdict

from .ner_extractor import CATEGORY_LABELS

# Groups of broadly interchangeable skills: a candidate who knows one can
# usually pick up another quickly. Reported as "transferable", never counted
# as a match.
RELATED_GROUPS = [
    {"pytorch", "tensorflow", "keras"},
    {"postgresql", "mysql", "sqlite", "oracle", "sql"},
    {"react", "angular", "vue", "next.js"},
    {"aws", "azure", "gcp"},
    {"flask", "django", "fastapi", "express", "spring boot"},
    {"mongodb", "cassandra", "nosql", "redis", "firebase"},
    {"tableau", "power bi", "matplotlib", "seaborn", "plotly"},
    {"apache spark", "hadoop", "kafka", "airflow"},
    {"spacy", "nltk", "transformers", "sentence transformers"},
    {"docker", "kubernetes"},
    {"scikit-learn", "xgboost", "lightgbm"},
    {"java", "c#", "kotlin", "scala"},
    {"javascript", "typescript"},
]


def _related(skill: str, have: set[str]) -> list[str]:
    related: set[str] = set()
    for group in RELATED_GROUPS:
        if skill in group:
            related |= (group & have) - {skill}
    return sorted(related)


def _priority(count: int, max_count: int) -> str:
    if count >= 2 or (max_count > 1 and count >= max_count):
        return "high"
    return "medium"


def analyze_skill_gap(resume_skills: dict[str, dict], job_skills: dict[str, dict]) -> dict:
    """Compare two ``{canonical: info}`` skill maps.

    Returns matching / missing / extra skills, overall and per-category
    coverage, transferable skills for each gap and plain-language advice.
    """
    resume_set, job_set = set(resume_skills), set(job_skills)
    matching = sorted(job_set & resume_set, key=lambda s: (-job_skills[s]["count"], s))
    missing = sorted(job_set - resume_set, key=lambda s: (-job_skills[s]["count"], s))
    extra = sorted(resume_set - job_set, key=lambda s: (-resume_skills[s]["count"], s))
    max_count = max((job_skills[s]["count"] for s in job_set), default=1)

    def item(skill: str, source: dict) -> dict:
        info = source[skill]
        return {
            "skill": skill,
            "display": info["display"],
            "category": info["category_label"],
            "job_mentions": job_skills.get(skill, {}).get("count", 0),
            "resume_mentions": resume_skills.get(skill, {}).get("count", 0),
        }

    matching_items = [item(s, job_skills) for s in matching]
    missing_items = []
    for s in missing:
        entry = item(s, job_skills)
        entry["priority"] = _priority(job_skills[s]["count"], max_count)
        entry["transferable_from"] = [
            resume_skills[r]["display"] for r in _related(s, resume_set)
        ]
        missing_items.append(entry)
    extra_items = [item(s, resume_skills) for s in extra]

    by_cat = defaultdict(lambda: {"required": 0, "matched": 0})
    for s in job_set:
        cat = job_skills[s]["category"]
        by_cat[cat]["required"] += 1
        if s in resume_set:
            by_cat[cat]["matched"] += 1
    categories = [
        {
            "category": CATEGORY_LABELS.get(cat, cat),
            "required": v["required"],
            "matched": v["matched"],
            "coverage": round(100 * v["matched"] / v["required"], 1),
        }
        for cat, v in sorted(by_cat.items(), key=lambda kv: -kv[1]["required"])
    ]

    coverage = round(100 * len(matching) / len(job_set), 1) if job_set else None

    return {
        "coverage": coverage,
        "required_count": len(job_set),
        "matching": matching_items,
        "missing": missing_items,
        "extra": extra_items,
        "categories": categories,
        "recommendations": _recommendations(missing_items, coverage, bool(job_set)),
    }


def _recommendations(missing: list[dict], coverage: float | None, has_job_skills: bool) -> list[str]:
    if not has_job_skills:
        return ["No recognised technical skills were found in the job description, so no skill gap can be computed."]
    if not missing:
        return ["Resume covers every recognised skill in the job description."]
    tips = []
    high = [m for m in missing if m["priority"] == "high"]
    if high:
        tips.append("Highest-priority gaps: " + ", ".join(m["display"] for m in high[:5]) + ".")
    for m in missing:
        if m["transferable_from"]:
            tips.append(
                f"{m['display']} is missing, but related experience "
                f"({', '.join(m['transferable_from'])}) is transferable - mention it explicitly if applicable."
            )
        if len(tips) >= 6:
            break
    if coverage is not None and coverage < 50:
        tips.append("Less than half of the required skills are covered; consider targeted upskilling or projects before applying.")
    elif coverage is not None and coverage < 80:
        tips.append("Add real projects or experience using the missing skills to the resume to improve fit.")
    return tips
