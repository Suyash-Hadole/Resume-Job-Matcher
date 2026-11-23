"""Sentence-embedding wrapper around ``all-MiniLM-L6-v2``.

The model is loaded lazily on first use. Long documents are split into
overlapping word windows (MiniLM truncates after ~256 word pieces), each window
is embedded, and the window vectors are averaged and L2-normalised.

If ``sentence-transformers`` or the model weights cannot be loaded (for example
no internet on first run) ``available`` is ``False`` and ``error`` explains why.
The matcher then reports a TF-IDF fallback instead of silently faking embeddings.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def shared_embedder() -> "EmbeddingModel":
    """One lazily loaded MiniLM instance shared by matching and the demo."""
    return EmbeddingModel()


def chunk_words(text: str, max_words: int = 160, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max(1, max_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return chunks


class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._tried = False
        self.error: str | None = None

    # -- loading -----------------------------------------------------------
    def _load(self):
        with self._lock:
            if self._tried:
                return
            self._tried = True
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001 - report any load failure
                self._model = None
                self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        if not self._tried:
            self._load()
        return self._model is not None

    @property
    def backend(self) -> str:
        if self.available:
            return f"sentence-transformers / {self.model_name}"
        return "tfidf-fallback (sentence-transformers model unavailable)"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- encoding ----------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, d)`` array of L2-normalised document embeddings."""
        if not self.available:
            raise RuntimeError(f"Embedding model unavailable: {self.error}")
        all_chunks: list[str] = []
        owners: list[int] = []
        for idx, text in enumerate(texts):
            chunks = chunk_words(text) or [""]
            all_chunks.extend(chunks)
            owners.extend([idx] * len(chunks))
        vectors = self._model.encode(
            all_chunks,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        dim = vectors.shape[1]
        out = np.zeros((len(texts), dim), dtype=np.float32)
        counts = np.zeros(len(texts))
        for vec, owner in zip(vectors, owners):
            out[owner] += vec
            counts[owner] += 1
        out /= np.maximum(counts[:, None], 1)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def similarity(self, resume_texts: list[str], job_text: str) -> list[float]:
        """Cosine similarity of each resume to the job description."""
        vectors = self.encode([job_text, *resume_texts])
        job_vec, resume_vecs = vectors[0], vectors[1:]
        return [float(v) for v in resume_vecs @ job_vec]
