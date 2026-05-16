"""Unit tests for app/retrieval.py."""

import pytest

from app.retrieval import VectorStore, retrieve_relevant_chunks, store_embeddings

# ████████████████████████████████████████████████████████████████████
# ██ STORE EMBEDDINGS
# ████████████████████████████████████████████████████████████████████

def test_store_embeddings_populates_chunks():
    store = VectorStore()
    store_embeddings(store, ["chunk one", "chunk two"], [[0.1, 0.2], [0.3, 0.4]])
    assert store.chunks == ["chunk one", "chunk two"]


def test_store_embeddings_populates_embeddings():
    store = VectorStore()
    store_embeddings(store, ["chunk one"], [[0.1, 0.2, 0.3]])
    assert store.embeddings == [[0.1, 0.2, 0.3]]


def test_store_embeddings_appends_on_multiple_calls():
    store = VectorStore()
    store_embeddings(store, ["chunk one"], [[0.1, 0.2]])
    store_embeddings(store, ["chunk two"], [[0.3, 0.4]])
    assert len(store.chunks) == 2
    assert len(store.embeddings) == 2


def test_store_embeddings_does_not_share_state_between_stores():
    store_a = VectorStore()
    store_b = VectorStore()
    store_embeddings(store_a, ["only in A"], [[0.1, 0.2]])
    assert store_b.chunks == []


# ████████████████████████████████████████████████████████████████████
# ██ RETRIEVE RELEVANT CHUNKS
# ████████████████████████████████████████████████████████████████████

def test_retrieve_returns_empty_list_for_empty_store():
    store = VectorStore()
    result = retrieve_relevant_chunks(store, [0.1, 0.2], top_k=3)
    assert result == []


def test_retrieve_returns_most_similar_chunk():
    store = VectorStore()
    store.chunks = ["about AI", "about cooking"]
    store.embeddings = [[1.0, 0.0], [0.0, 1.0]]
    result = retrieve_relevant_chunks(store, [1.0, 0.0], top_k=1)
    assert result == ["about AI"]


def test_retrieve_respects_top_k():
    store = VectorStore()
    store.chunks = ["a", "b", "c", "d", "e"]
    store.embeddings = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.0, 1.0],
    ]
    result = retrieve_relevant_chunks(store, [1.0, 0.0], top_k=3)
    assert len(result) == 3


def test_retrieve_returns_chunks_in_similarity_order():
    store = VectorStore()
    store.chunks = ["best match", "second match", "worst match"]
    store.embeddings = [
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
    ]
    result = retrieve_relevant_chunks(store, [1.0, 0.0], top_k=3)
    assert result[0] == "best match"
    assert result[1] == "second match"


def test_retrieve_caps_results_when_top_k_exceeds_store_size():
    store = VectorStore()
    store.chunks = ["only one"]
    store.embeddings = [[1.0, 0.0]]
    result = retrieve_relevant_chunks(store, [1.0, 0.0], top_k=10)
    assert len(result) == 1


def test_retrieve_correct_chunks_from_multiple():
    store = VectorStore()
    store.chunks = ["about AI", "about cooking", "about ML"]
    store.embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ]
    result = retrieve_relevant_chunks(store, [1.0, 0.0], top_k=2)
    assert "about AI" in result
    assert "about ML" in result
    assert "about cooking" not in result
