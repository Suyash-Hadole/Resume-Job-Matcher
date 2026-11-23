"""Text cleaning and spaCy-based preprocessing.

``get_nlp`` prefers the ``en_core_web_sm`` model (tokenizer, lemmatizer and NER).
If that model is not installed it falls back to a blank English pipeline with a
sentence splitter so the app still runs; ``nlp_status`` reports which backend is
active so the UI never pretends NER ran when it did not.
"""

from __future__ import annotations

import functools
import re
import unicodedata

_BULLETS = "•●▪■◦○◆▶►✔✓➢➤*"
_BULLET_RE = re.compile(f"[{re.escape(_BULLETS)}]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES_RE = re.compile(r"[ \t ]+")
_BLANKS_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

SPACY_MODEL = "en_core_web_sm"


def clean_text(text: str) -> str:
    """Normalise unicode, bullets, control characters and whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _BULLET_RE.sub("\n- ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip()


@functools.lru_cache(maxsize=1)
def _load_pipeline():
    """Load spaCy once. Returns (nlp, backend_description, has_ner)."""
    try:
        import spacy
    except ImportError:
        return None, "spaCy not installed (regex-only mode)", False

    try:
        nlp = spacy.load(SPACY_MODEL)
        return nlp, f"spaCy {spacy.__version__} / {SPACY_MODEL}", "ner" in nlp.pipe_names
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return (
            nlp,
            f"spaCy {spacy.__version__} blank pipeline "
            f"('{SPACY_MODEL}' not installed - NER limited)",
            False,
        )


def get_nlp():
    """Return the spaCy pipeline or ``None`` if spaCy is unavailable."""
    return _load_pipeline()[0]


def nlp_status() -> dict:
    nlp, backend, has_ner = _load_pipeline()
    return {"backend": backend, "ner_model": bool(has_ner), "available": nlp is not None}


def lemmatize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Return lower-cased lemmas, dropping punctuation, numbers and stop words."""
    nlp = get_nlp()
    text = clean_text(text)
    if nlp is None:
        words = re.findall(r"[A-Za-z][A-Za-z+#.\-]{1,}", text.lower())
        return words
    lemmas: list[str] = []
    has_lemmatizer = "lemmatizer" in nlp.pipe_names or "attribute_ruler" in nlp.pipe_names
    for token in nlp(text[:nlp.max_length - 1]):
        if token.is_space or token.is_punct or token.like_num:
            continue
        if not keep_stopwords and token.is_stop:
            continue
        value = token.lemma_ if has_lemmatizer and token.lemma_ else token.text
        value = value.lower().strip()
        if len(value) > 1 or value in {"c", "r"}:
            lemmas.append(value)
    return lemmas


def preprocess(text: str) -> str:
    """Cleaned, lemmatised text used for the TF-IDF baseline."""
    return " ".join(lemmatize(text))


def sentences(text: str) -> list[str]:
    nlp = get_nlp()
    text = clean_text(text)
    if nlp is None:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [s.text.strip() for s in nlp(text[:nlp.max_length - 1]).sents if s.text.strip()]
