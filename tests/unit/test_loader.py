"""Unit tests for app/loader.py."""

from unittest.mock import MagicMock, patch

import pytest

from app.loader import load_document, load_docx_file, load_pdf_file, load_txt_file

# ████████████████████████████████████████████████████████████████████
# ██ TXT LOADING
# ████████████████████████████████████████████████████████████████████

def test_load_txt_file_returns_content(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("hello world", encoding="utf-8")
    assert load_txt_file(file) == "hello world"


def test_load_txt_file_preserves_newlines(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("line one\nline two\nline three", encoding="utf-8")
    assert load_txt_file(file) == "line one\nline two\nline three"


# ████████████████████████████████████████████████████████████████████
# ██ PDF LOADING
# ████████████████████████████████████████████████████████████████████

def test_load_pdf_file_extracts_single_page(tmp_path):
    file = tmp_path / "doc.pdf"
    file.touch()

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF page content"

    with patch("app.loader.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [mock_page]
        result = load_pdf_file(file)

    assert result == "PDF page content"


def test_load_pdf_file_joins_multiple_pages(tmp_path):
    file = tmp_path / "doc.pdf"
    file.touch()

    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "Page one"
    pages[1].extract_text.return_value = "Page two"

    with patch("app.loader.PdfReader") as mock_reader:
        mock_reader.return_value.pages = pages
        result = load_pdf_file(file)

    assert result == "Page one\nPage two"


def test_load_pdf_file_handles_empty_page(tmp_path):
    file = tmp_path / "doc.pdf"
    file.touch()

    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "Page one"
    pages[1].extract_text.return_value = None

    with patch("app.loader.PdfReader") as mock_reader:
        mock_reader.return_value.pages = pages
        result = load_pdf_file(file)

    assert result == "Page one\n"


# ████████████████████████████████████████████████████████████████████
# ██ DISPATCHER
# ████████████████████████████████████████████████████████████████████

def test_load_document_routes_txt(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("hello world", encoding="utf-8")
    assert load_document(file) == "hello world"


def test_load_document_routes_pdf(tmp_path):
    file = tmp_path / "doc.pdf"
    file.touch()

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF content"

    with patch("app.loader.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [mock_page]
        result = load_document(file)

    assert result == "PDF content"


def test_load_document_routes_docx(tmp_path):
    file = tmp_path / "doc.docx"
    file.touch()

    mock_para = MagicMock()
    mock_para.text = "Docx paragraph"

    with patch("app.loader.Document") as mock_doc:
        mock_doc.return_value.paragraphs = [mock_para]
        result = load_document(file)

    assert result == "Docx paragraph"


def test_load_document_raises_for_unsupported_extension(tmp_path):
    file = tmp_path / "doc.csv"
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file)


# ████████████████████████████████████████████████████████████████████
# ██ DOCX LOADING
# ████████████████████████████████████████████████████████████████████

def test_load_docx_file_extracts_paragraphs(tmp_path):
    file = tmp_path / "doc.docx"
    file.touch()

    paragraphs = [MagicMock(), MagicMock()]
    paragraphs[0].text = "First paragraph"
    paragraphs[1].text = "Second paragraph"

    with patch("app.loader.Document") as mock_doc:
        mock_doc.return_value.paragraphs = paragraphs
        result = load_docx_file(file)

    assert result == "First paragraph\nSecond paragraph"


def test_load_docx_file_handles_empty_paragraph(tmp_path):
    file = tmp_path / "doc.docx"
    file.touch()

    paragraphs = [MagicMock(), MagicMock()]
    paragraphs[0].text = "Only paragraph"
    paragraphs[1].text = ""

    with patch("app.loader.Document") as mock_doc:
        mock_doc.return_value.paragraphs = paragraphs
        result = load_docx_file(file)

    assert result == "Only paragraph\n"
