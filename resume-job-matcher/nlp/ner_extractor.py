"""Named-entity extraction and technical-skill normalisation.

* spaCy NER supplies PERSON / ORG / GPE / DATE entities when the
  ``en_core_web_sm`` model is installed.
* Regular expressions extract contact details, degrees and experience.
* ``SkillExtractor`` maps free text onto a canonical skill taxonomy
  (``data/skills.json``) so "ReactJS", "react.js" and "React" all become "React".
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

from .preprocessing import clean_text, get_nlp, nlp_status

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SKILLS_PATH = os.path.join(DATA_DIR, "skills.json")

CATEGORY_LABELS = {
    "programming_languages": "Programming Languages",
    "web": "Web & Frameworks",
    "databases": "Databases",
    "ml_ai": "Machine Learning & AI",
    "data": "Data & Analytics",
    "devops_cloud": "DevOps & Cloud",
    "practices": "Engineering Practices",
}

# Display names for skills whose title-cased form would look wrong.
DISPLAY_OVERRIDES = {
    "sql": "SQL", "html": "HTML", "css": "CSS", "aws": "AWS", "gcp": "GCP",
    "nlp": "NLP", "ner": "NER", "php": "PHP", "etl": "ETL", "rag": "RAG",
    "jwt": "JWT", "oauth": "OAuth", "ci/cd": "CI/CD", "a/b testing": "A/B Testing",
    "c++": "C++", "c#": "C#", "c": "C", "r": "R", "go": "Go",
    "node.js": "Node.js", "next.js": "Next.js", "mysql": "MySQL",
    "postgresql": "PostgreSQL", "mongodb": "MongoDB", "nosql": "NoSQL",
    "graphql": "GraphQL", "javascript": "JavaScript", "typescript": "TypeScript",
    "fastapi": "FastAPI", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "opencv": "OpenCV", "mlops": "MLOps", "mlflow": "MLflow", "spacy": "spaCy",
    "nltk": "NLTK", "bert": "BERT", "gpt": "LLMs", "github": "GitHub",
    "numpy": "NumPy", "scipy": "SciPy", "pandas": "pandas", "power bi": "Power BI",
    "rest api": "REST API", "langchain": "LangChain", "tailwind css": "Tailwind CSS",
    "linux": "Linux", "github actions": "GitHub Actions", "matlab": "MATLAB",
    "sqlite": "SQLite", "elasticsearch": "Elasticsearch", "apache spark": "Apache Spark",
    "apache airflow": "Airflow", "airflow": "Airflow", "kafka": "Kafka",
    "tdd": "TDD", "oop": "OOP", "dsa": "DSA", "api design": "API Design",
    "object oriented programming": "Object-Oriented Programming",
    "natural language processing": "Natural Language Processing",
    "named entity recognition": "Named Entity Recognition",
    "unit testing": "Unit Testing", "react": "React", "vue": "Vue",
}

# Aliases of at most this many characters are matched case-sensitively so that
# "C", "R", "Go", "ML" do not fire on ordinary words.
_SHORT_ALIAS_LEN = 2
_CASE_SENSITIVE_FORMS = {"c": ["C"], "r": ["R"], "go": ["Go"], "ml": ["ML"], "dl": ["DL"], "rl": ["RL"], "js": ["JS", "js"]}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().\-]{8,18}\d(?![\w])")
_YEAR_PAIR_RE = re.compile(r"(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)\d{2}")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s,;|)]+|(?:linkedin\.com|github\.com)/[^\s,;|)]+", re.I)
_DATE_HINT_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|current)\b",
    re.I,
)
_DEGREE_RE = re.compile(
    r"\b(?:ph\.?d\.?|doctorate|m\.?\s?tech|b\.?\s?tech|b\.?\s?e\.?|m\.?\s?e\.?|b\.?\s?sc\.?|m\.?\s?sc\.?|"
    r"b\.?\s?c\.?a\.?|m\.?\s?c\.?a\.?|m\.?b\.?a\.?|bachelor(?:'s)?(?: of [A-Za-z ]{3,30})?|"
    r"master(?:'s)?(?: of [A-Za-z ]{3,30})?|diploma(?: in [A-Za-z ]{3,30})?)\b",
    re.I,
)
_YEARS_EXPLICIT_RE = re.compile(r"(\d{1,2})(?:\.\d)?\s*\+?\s*(?:years?|yrs?)\b(?:\s+of)?\s+(?:[A-Za-z]+\s+){0,3}?experience", re.I)
_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now|ongoing)",
    re.I,
)
_EDU_LINE_RE = re.compile(
    r"\b(b\.?\s?tech|m\.?\s?tech|bachelor|master|university|college|school|institute|degree|diploma|cgpa|gpa|ph\.?d)\b",
    re.I,
)
_ORG_HINT_RE = re.compile(
    r"\b[A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){0,4}\s+"
    r"(?:Inc\.?|Ltd\.?|Pvt\.?\s*Ltd\.?|LLC|Corp\.?|Corporation|Technologies|Systems|Solutions|Labs|"
    r"University|Institute|College)\b"
)
_NON_ORG_TERMS = {
    "computer science", "information technology", "engineering", "data science",
    "artificial intelligence", "computer engineering", "electronics", "mathematics",
    "statistics", "skills", "education", "experience", "projects", "summary",
}
_NOISE_ENTITY_RE = re.compile(r"^[\W\d_]+$")
_CURRENT_YEAR = 2026


@lru_cache(maxsize=1)
def _load_taxonomy():
    with open(SKILLS_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    meta: dict[str, dict] = {}
    for category, skills in raw.items():
        for canonical, aliases in skills.items():
            meta[canonical] = {
                "category": category,
                "display": DISPLAY_OVERRIDES.get(canonical, canonical.title()),
                "aliases": aliases,
            }
    return meta


def skill_metadata() -> dict[str, dict]:
    return _load_taxonomy()


def _flex(alias: str) -> str:
    """Regex for an alias that tolerates spaces/hyphens/underscores between words."""
    parts = re.split(r"[\s\-_]+", alias.strip())
    return r"[\s\-_]+".join(re.escape(p) for p in parts)


class SkillExtractor:
    """Dictionary + regex skill matcher with alias normalisation."""

    def __init__(self):
        self.meta = _load_taxonomy()
        self.alias_to_skill: dict[str, str] = {}
        self.cs_to_skill: dict[str, str] = {}
        for canonical, info in self.meta.items():
            for alias in [canonical, *info["aliases"]]:
                key = alias.lower()
                if len(key) <= _SHORT_ALIAS_LEN:
                    forms = _CASE_SENSITIVE_FORMS.get(key, [alias.upper()])
                    for form in forms:
                        self.cs_to_skill[form] = canonical
                else:
                    self.alias_to_skill[key] = canonical
        pre = r"(?<![\w+#.\-/])"
        post = r"(?![\w+#]|\.\w)"
        ordered = sorted(self.alias_to_skill, key=len, reverse=True)
        self._regex = re.compile(pre + "(" + "|".join(_flex(a) for a in ordered) + ")" + post, re.I)
        cs_forms = sorted(self.cs_to_skill, key=len, reverse=True)
        self._cs_regex = re.compile(
            r"(?:(?<=^)|(?<=[\s,;:/(|\-]))(" + "|".join(re.escape(f) for f in cs_forms) + r")(?=$|[\s,;/)|]|\.(?:\s|$))",
            re.M,
        ) if cs_forms else None

    @staticmethod
    def _norm(match_text: str) -> str:
        return re.sub(r"[\s\-_]+", " ", match_text.lower()).strip()

    def extract(self, text: str) -> dict[str, dict]:
        """Return ``{canonical: {display, category, count}}`` for skills found."""
        text = clean_text(text)
        counts: Counter = Counter()
        for m in self._regex.finditer(text):
            key = self._norm(m.group(1))
            canonical = self.alias_to_skill.get(key)
            if canonical is None:
                # alias contained a hyphen/underscore in the dictionary
                for alias, canon in self.alias_to_skill.items():
                    if self._norm(alias) == key:
                        canonical = canon
                        break
            if canonical:
                counts[canonical] += 1
        if self._cs_regex is not None:
            for m in self._cs_regex.finditer(text):
                canonical = self.cs_to_skill.get(m.group(1))
                if canonical:
                    counts[canonical] += 1
        result = {}
        for canonical, count in counts.items():
            info = self.meta[canonical]
            result[canonical] = {
                "name": canonical,
                "display": info["display"],
                "category": info["category"],
                "category_label": CATEGORY_LABELS.get(info["category"], info["category"]),
                "count": count,
            }
        return result


def _dedupe(items, limit=15):
    seen, out = set(), []
    for item in items:
        key = re.sub(r"\W+", "", item.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _guess_name(text: str, person_entities: list[str]) -> str | None:
    head = [ln.strip() for ln in text.split("\n") if ln.strip()][:6]
    for line in head:
        words = line.split()
        if 2 <= len(words) <= 4 and not re.search(r"[\d@:/|]", line):
            if all(re.match(r"^[A-Z][A-Za-z'.\-]*$", w) or w.isupper() for w in words):
                lowered = line.lower()
                if not any(k in lowered for k in ("resume", "curriculum", "vitae", "profile", "summary", "engineer", "developer", "objective")):
                    return line.title() if line.isupper() else line
    head_text = "\n".join(head)
    for person in person_entities:
        if person in head_text and 1 <= len(person.split()) <= 4:
            return person
    return None


def _estimate_experience(text: str) -> dict:
    explicit = [int(m.group(1)) for m in _YEARS_EXPLICIT_RE.finditer(text)]
    if explicit:
        return {"years": max(explicit), "source": "stated in resume"}
    spans = []
    for line in text.split("\n"):
        if _EDU_LINE_RE.search(line):
            continue
        for m in _RANGE_RE.finditer(line):
            start = int(m.group(1))
            end_raw = m.group(2).lower()
            end = _CURRENT_YEAR if end_raw in {"present", "current", "now", "ongoing"} else int(end_raw)
            if 1980 <= start <= end <= _CURRENT_YEAR + 1:
                spans.append((start, end))
    if not spans:
        return {"years": None, "source": "not found"}
    spans.sort()
    total, cur_s, cur_e = 0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return {"years": min(total, 45), "source": "estimated from date ranges"}


class EntityExtractor:
    """Combine spaCy NER, regexes and the skill taxonomy into one profile."""

    MAX_NER_CHARS = 100_000

    def __init__(self, skill_extractor: SkillExtractor | None = None):
        self.skills = skill_extractor or SkillExtractor()

    def _spacy_entities(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = defaultdict(list)
        nlp = get_nlp()
        if nlp is None or "ner" not in nlp.pipe_names:
            return found
        skill_words = {a for info in self.skills.meta.values() for a in [info["display"].lower()]}
        for ent in nlp(text[: self.MAX_NER_CHARS]).ents:
            value = re.sub(r"\s+", " ", ent.text).strip(" ,.-:|•")
            if len(value) < 2 or _NOISE_ENTITY_RE.match(value) or value.lower() in skill_words:
                continue
            if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "LOC"}:
                found[ent.label_].append(value)
        return found

    @staticmethod
    def _plausible_org(value: str, skill_terms: set[str]) -> bool:
        """Reject spaCy ORG false positives such as skill lists and degree fields."""
        if len(value) > 60 or value.lower() in _NON_ORG_TERMS:
            return False
        parts = [p.strip().lower() for p in re.split(r"[,/&|]", value) if p.strip()]
        return not (parts and all(p in skill_terms for p in parts))

    def extract_skills(self, text: str) -> dict[str, dict]:
        return self.skills.extract(text)

    def extract(self, text: str) -> dict:
        text = clean_text(text)
        spacy_ents = self._spacy_entities(text)

        organizations = list(spacy_ents.get("ORG", []))
        organizations += [m.group(0).strip() for m in _ORG_HINT_RE.finditer(text)]
        skill_names = {i["display"].lower() for i in self.skills.meta.values()}
        skill_terms = skill_names | {a for k, i in self.skills.meta.items() for a in [k, *i["aliases"]]}
        organizations = [o for o in organizations if self._plausible_org(o, skill_terms)]

        emails = _dedupe(_EMAIL_RE.findall(text), 3)
        phones = []
        for m in _PHONE_RE.finditer(text):
            candidate = m.group(0).strip()
            digits = re.sub(r"\D", "", candidate)
            if not 10 <= len(digits) <= 13 or _YEAR_PAIR_RE.search(candidate):
                continue
            phones.append(re.sub(r"\s+", " ", candidate))
        links = _dedupe([u.rstrip(".") for u in _URL_RE.findall(text)], 6)

        degrees = _dedupe([re.sub(r"\s+", " ", d.group(0)).strip() for d in _DEGREE_RE.finditer(text)], 6)
        skills = self.extract_skills(text)
        status = nlp_status()
        locations = _dedupe(spacy_ents.get("GPE", []) + spacy_ents.get("LOC", []), 6)
        name = _guess_name(text, spacy_ents.get("PERSON", []))
        persons = _dedupe(([name] if name else []) + spacy_ents.get("PERSON", []), 6)
        dates = _dedupe(
            spacy_ents.get("DATE", [])
            + [m.group(0).strip() for m in _DATE_HINT_RE.finditer(text)],
            8,
        )
        skill_list = sorted(skills.values(), key=lambda s: (-s["count"], s["display"]))
        skills_by_category = {}
        for skill in skill_list:
            skills_by_category.setdefault(skill["category_label"], []).append(skill)

        return {
            "name": name,
            "persons": persons,
            "emails": emails,
            "phones": _dedupe(phones, 3),
            "links": links,
            "organizations": _dedupe(organizations, 8),
            "locations": locations,
            "dates": dates,
            "education": degrees,
            "experience": _estimate_experience(text),
            "skills": skill_list,
            "skills_by_category": [
                {"category": category, "skills": items}
                for category, items in sorted(skills_by_category.items())
            ],
            "entity_counts": {
                "persons": len(persons),
                "organizations": len(_dedupe(organizations, 8)),
                "locations": len(locations),
                "dates": len(dates),
                "emails": len(emails),
                "phones": len(_dedupe(phones, 3)),
                "links": len(links),
                "education": len(degrees),
                "skills": len(skill_list),
            },
            "ner_backend": status["backend"],
            "ner_model_loaded": status["ner_model"],
        }
