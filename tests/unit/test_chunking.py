"""Unit tests for app/chunking.py."""

import pytest

from app.chunking import clean_chunk_text, split_text_into_chunks

# ████████████████████████████████████████████████████████████████████
# ██ CLEAN CHUNK TEXT
# ████████████████████████████████████████████████████████████████████

def test_clean_strips_leading_and_trailing_whitespace():
    assert clean_chunk_text("  hello  ") == "hello"


def test_clean_normalizes_crlf_line_endings():
    assert clean_chunk_text("line one\r\nline two") == "line one\nline two"


def test_clean_normalizes_cr_line_endings():
    assert clean_chunk_text("line one\rline two") == "line one\nline two"


def test_clean_collapses_multiple_blank_lines():
    assert clean_chunk_text("a\n\n\n\nb") == "a\n\nb"


def test_clean_preserves_single_blank_line():
    assert clean_chunk_text("a\n\nb") == "a\n\nb"


def test_clean_empty_string_returns_empty():
    assert clean_chunk_text("") == ""


# ████████████████████████████████████████████████████████████████████
# ██ SPLIT TEXT INTO CHUNKS
# ████████████████████████████████████████████████████████████████████

def test_empty_string_returns_empty_list():
    assert split_text_into_chunks("") == []


def test_short_text_returns_single_chunk():
    text = "a" * 50
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunks_do_not_exceed_chunk_size():
    text = "a" * 3000
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks)


def test_overlap_between_consecutive_chunks():
    text = "a" * 2000
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=100)
    assert chunks[0][-100:] == chunks[1][:100]


def test_correct_number_of_chunks():
    # length=2000, chunk_size=1000, overlap=100, step=900
    # start=0 → chunk 1, start=900 → chunk 2, start=1800 → chunk 3
    text = "a" * 2000
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=100)
    assert len(chunks) == 3


def test_chunks_cover_full_text():
    text = "abcdefghij" * 100
    chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)
    assert chunks[0].startswith(text[:10])
    assert text[-10:] in chunks[-1]


def test_whitespace_only_input_returns_empty_list():
    assert split_text_into_chunks("   \n\n   ") == []
