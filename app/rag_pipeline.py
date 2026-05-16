"""Main orchestration logic for the RAG pipeline."""

import logging
from pathlib import Path

_log = logging.getLogger(__name__)

from app.chunking import split_text_into_chunks
from app.embeddings import create_embeddings, create_query_embedding
from app.llm import generate_answer, generate_answer_with_tools, summarize_document
from app.loader import load_docx_as_html, load_document
from app.retrieval import VectorStore, retrieve_relevant_chunks, store_embeddings
from app.tool_gate import should_use_tools
from app.utils import detect_language

# ████████████████████████████████████████████████████████████████████
# ██ INGESTION
# ████████████████████████████████████████████████████████████████████

def ingest_document(file_path: Path, store: VectorStore) -> None:
    """Load a document, split it into chunks, embed them, and store them."""
    text = load_document(file_path)
    chunks = split_text_into_chunks(text)
    embeddings = create_embeddings(chunks)
    store_embeddings(store, chunks, embeddings)


# ████████████████████████████████████████████████████████████████████
# ██ QUERYING
# ████████████████████████████████████████████████████████████████████

def answer_question_about_document(
    store: VectorStore,
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Embed the question, retrieve relevant chunks, and generate an answer.

    If the gate detects a computational question, tools are exposed to the LLM.
    On any tool-path failure the pipeline falls back to normal RAG generation.
    Retrieval always runs first regardless of whether tools are used.
    """
    _log.debug("ℹ️  [PIPELINE]: Q&A started | question: %.80r", question)
    language = detect_language(question)
    question_embedding = create_query_embedding(question)
    chunks = retrieve_relevant_chunks(store, question_embedding)
    _log.debug("🔍 [PIPELINE]: Retrieved %d chunk(s)", len(chunks))

    if should_use_tools(question):
        _log.debug("ℹ️  [PIPELINE]: Tool path activated")
        try:
            answer = generate_answer_with_tools(chunks, question, language, history)
            _log.debug("✅ [PIPELINE]: Tool-augmented answer returned")
            return answer
        except Exception as exc:
            _log.warning("⚠️  [PIPELINE]: Tool path failed (%s) — falling back to standard RAG", exc)

    return generate_answer(chunks, question, language, history)


# ████████████████████████████████████████████████████████████████████
# ██ PIPELINE
# ████████████████████████████████████████████████████████████████████

def run_pipeline(file_path: Path, question: str) -> str:
    """Create a VectorStore, ingest a document, and answer a question about it."""
    store = VectorStore()
    ingest_document(file_path, store)
    return answer_question_about_document(store, question)


# ████████████████████████████████████████████████████████████████████
# ██ SUMMARIZATION
# ████████████████████████████████████████████████████████████████████

def summarize_file(file_path: Path, doc_type: str = "Article") -> str:
    """Load a document, chunk it, and return a summary shaped by doc_type."""
    text = load_document(file_path)
    chunks = split_text_into_chunks(text)
    language = detect_language(text[:2000])
    return summarize_document(chunks, doc_type, language)


# ████████████████████████████████████████████████████████████████████
# ██ PREVIEW
# ████████████████████████████████████████████████████████████████████

def get_document_text(file_path: Path) -> str:
    """Return the raw extracted text from a document file for preview."""
    return load_document(file_path)


def get_document_preview_html(file_path: Path) -> str:
    """Return styled HTML for rendering a DOCX document as an inline browser preview."""
    return load_docx_as_html(file_path)
