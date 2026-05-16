"""Similarity search and retrieval helpers."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ████████████████████████████████████████████████████████████████████
# ██ VECTOR STORE
# ████████████████████████████████████████████████████████████████████

@dataclass
class VectorStore:
    chunks: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)


# ████████████████████████████████████████████████████████████████████
# ██ STORAGE
# ████████████████████████████████████████████████████████████████████

def store_embeddings(
    store: VectorStore,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> None:
    """Store text chunks and their embeddings in the given VectorStore."""
    store.chunks.extend(chunks)
    store.embeddings.extend(embeddings)


# ████████████████████████████████████████████████████████████████████
# ██ RETRIEVAL
# ████████████████████████████████████████████████████████████████████

def retrieve_relevant_chunks(
    store: VectorStore,
    question_embedding: Sequence[float],
    top_k: int = 8,
) -> list[str]:
    """Retrieve the most relevant chunks using cosine similarity search."""
    if not store.embeddings:
        return []

    query = np.array(question_embedding)
    matrix = np.array(store.embeddings)

    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    similarities = matrix_norm @ query_norm

    k = min(top_k, len(store.chunks))
    top_indices = np.argsort(similarities)[::-1][:k]
    return [store.chunks[i] for i in top_indices]
