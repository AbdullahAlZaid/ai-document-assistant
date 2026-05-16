"""Unit tests for app/rag_pipeline.py."""

from pathlib import Path
from unittest.mock import call, patch

from app.rag_pipeline import answer_question_about_document, ingest_document, run_pipeline
from app.retrieval import VectorStore

# ████████████████████████████████████████████████████████████████████
# ██ INGEST DOCUMENT
# ████████████████████████████████████████████████████████████████████

def test_ingest_document_calls_load(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("content", encoding="utf-8")
    store = VectorStore()

    with patch("app.rag_pipeline.load_document", return_value="content") as mock_load, \
         patch("app.rag_pipeline.split_text_into_chunks", return_value=["chunk"]), \
         patch("app.rag_pipeline.create_embeddings", return_value=[[0.1, 0.2]]), \
         patch("app.rag_pipeline.store_embeddings"):
        ingest_document(file, store)

    mock_load.assert_called_once_with(file)


def test_ingest_document_passes_text_to_chunker(tmp_path):
    file = tmp_path / "doc.txt"
    store = VectorStore()

    with patch("app.rag_pipeline.load_document", return_value="raw text"), \
         patch("app.rag_pipeline.split_text_into_chunks", return_value=["chunk"]) as mock_split, \
         patch("app.rag_pipeline.create_embeddings", return_value=[[0.1, 0.2]]), \
         patch("app.rag_pipeline.store_embeddings"):
        ingest_document(file, store)

    mock_split.assert_called_once_with("raw text")


def test_ingest_document_passes_chunks_to_embedder(tmp_path):
    file = tmp_path / "doc.txt"
    store = VectorStore()

    with patch("app.rag_pipeline.load_document", return_value="text"), \
         patch("app.rag_pipeline.split_text_into_chunks", return_value=["chunk a", "chunk b"]), \
         patch("app.rag_pipeline.create_embeddings", return_value=[[0.1], [0.2]]) as mock_embed, \
         patch("app.rag_pipeline.store_embeddings"):
        ingest_document(file, store)

    mock_embed.assert_called_once_with(["chunk a", "chunk b"])


def test_ingest_document_stores_chunks_and_embeddings(tmp_path):
    file = tmp_path / "doc.txt"
    store = VectorStore()

    with patch("app.rag_pipeline.load_document", return_value="text"), \
         patch("app.rag_pipeline.split_text_into_chunks", return_value=["chunk"]), \
         patch("app.rag_pipeline.create_embeddings", return_value=[[0.1, 0.2]]), \
         patch("app.rag_pipeline.store_embeddings") as mock_store:
        ingest_document(file, store)

    mock_store.assert_called_once_with(store, ["chunk"], [[0.1, 0.2]])


# ████████████████████████████████████████████████████████████████████
# ██ ANSWER QUESTION
# ████████████████████████████████████████████████████████████████████

def test_answer_question_embeds_the_question():
    store = VectorStore()

    with patch("app.rag_pipeline.create_query_embedding", return_value=[0.1, 0.2]) as mock_embed, \
         patch("app.rag_pipeline.retrieve_relevant_chunks", return_value=["chunk"]), \
         patch("app.rag_pipeline.generate_answer", return_value="answer"):
        answer_question_about_document(store, "what is ML?")

    mock_embed.assert_called_once_with("what is ML?")


def test_answer_question_retrieves_with_query_embedding():
    store = VectorStore()

    with patch("app.rag_pipeline.create_query_embedding", return_value=[0.1, 0.2]), \
         patch("app.rag_pipeline.retrieve_relevant_chunks", return_value=["chunk"]) as mock_retrieve, \
         patch("app.rag_pipeline.generate_answer", return_value="answer"):
        answer_question_about_document(store, "question")

    mock_retrieve.assert_called_once_with(store, [0.1, 0.2])


def test_answer_question_returns_generated_answer():
    store = VectorStore()

    with patch("app.rag_pipeline.create_query_embedding", return_value=[0.1]), \
         patch("app.rag_pipeline.retrieve_relevant_chunks", return_value=["chunk"]), \
         patch("app.rag_pipeline.generate_answer", return_value="the final answer"):
        result = answer_question_about_document(store, "question")

    assert result == "the final answer"


# ████████████████████████████████████████████████████████████████████
# ██ RUN PIPELINE
# ████████████████████████████████████████████████████████████████████

def test_run_pipeline_returns_answer():
    with patch("app.rag_pipeline.ingest_document"), \
         patch("app.rag_pipeline.answer_question_about_document", return_value="pipeline answer"):
        result = run_pipeline(Path("doc.txt"), "question")

    assert result == "pipeline answer"


def test_run_pipeline_calls_ingest_then_answer():
    call_order = []

    with patch("app.rag_pipeline.ingest_document", side_effect=lambda *a: call_order.append("ingest")), \
         patch("app.rag_pipeline.answer_question_about_document", side_effect=lambda *a: call_order.append("answer") or "ans"):
        run_pipeline(Path("doc.txt"), "question")

    assert call_order == ["ingest", "answer"]
