"""Embedding creation utilities for document chunks."""

from typing import Sequence

from sentence_transformers import SentenceTransformer

# ████████████████████████████████████████████████████████████████████
# ██ MODEL
# ████████████████████████████████████████████████████████████████████

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ████████████████████████████████████████████████████████████████████
# ██ EMBEDDINGS
# ████████████████████████████████████████████████████████████████████

def create_embeddings(texts: Sequence[str]) -> list[list[float]]:
    """Convert a list of texts into vector embeddings."""
    return _get_model().encode(list(texts)).tolist()


def create_query_embedding(question: str) -> list[float]:
    """Convert a user question into a vector embedding for retrieval."""
    return _get_model().encode(question).tolist()
