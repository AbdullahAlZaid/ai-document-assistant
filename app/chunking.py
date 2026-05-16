"""Document chunking utilities."""

import re


# ████████████████████████████████████████████████████████████████████
# ██ CLEANING
# ████████████████████████████████████████████████████████████████████

def clean_chunk_text(text: str) -> str:
    """Normalize raw document text before chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ████████████████████████████████████████████████████████████████████
# ██ SPLITTING
# ████████████████████████████████████████████████████████████████████

def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """Split document text into overlapping chunks of fixed character size."""
    text = clean_chunk_text(text)
    if not text:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step

    return chunks
