"""Unit tests for app/embeddings.py."""

from unittest.mock import MagicMock, patch

import numpy as np

from app.embeddings import create_embeddings, create_query_embedding

# ████████████████████████████████████████████████████████████████████
# ██ HELPERS
# ████████████████████████████████████████████████████████████████████

def _make_mock_model(vectors) -> MagicMock:
    mock = MagicMock()
    mock.encode.return_value = np.array(vectors)
    return mock


# ████████████████████████████████████████████████████████████████████
# ██ CREATE EMBEDDINGS
# ████████████████████████████████████████████████████████████████████

def test_create_embeddings_returns_list_of_lists():
    mock = _make_mock_model([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    with patch("app.embeddings._get_model", return_value=mock):
        result = create_embeddings(["text one", "text two"])
    assert isinstance(result, list)
    assert isinstance(result[0], list)


def test_create_embeddings_count_matches_input():
    mock = _make_mock_model([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    with patch("app.embeddings._get_model", return_value=mock):
        result = create_embeddings(["a", "b", "c"])
    assert len(result) == 3


def test_create_embeddings_vector_dimensions():
    mock = _make_mock_model([[0.1, 0.2, 0.3, 0.4]])
    with patch("app.embeddings._get_model", return_value=mock):
        result = create_embeddings(["text"])
    assert len(result[0]) == 4


def test_create_embeddings_returns_floats():
    mock = _make_mock_model([[0.1, 0.2, 0.3]])
    with patch("app.embeddings._get_model", return_value=mock):
        result = create_embeddings(["text"])
    assert all(isinstance(v, float) for v in result[0])


# ████████████████████████████████████████████████████████████████████
# ██ CREATE QUERY EMBEDDING
# ████████████████████████████████████████████████████████████████████

def test_create_query_embedding_returns_flat_list():
    mock = _make_mock_model([0.1, 0.2, 0.3])
    with patch("app.embeddings._get_model", return_value=mock):
        result = create_query_embedding("what is AI?")
    assert isinstance(result, list)
    assert isinstance(result[0], float)


def test_create_query_embedding_correct_dimensions():
    mock = _make_mock_model([0.1, 0.2, 0.3, 0.4, 0.5])
    with patch("app.embeddings._get_model", return_value=mock):
        result = create_query_embedding("question")
    assert len(result) == 5


def test_create_query_embedding_returns_floats():
    mock = _make_mock_model([0.1, 0.2, 0.3])
    with patch("app.embeddings._get_model", return_value=mock):
        result = create_query_embedding("question")
    assert all(isinstance(v, float) for v in result)
