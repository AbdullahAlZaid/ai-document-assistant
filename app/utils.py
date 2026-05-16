"""General helper functions for the document assistant."""

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display


# ████████████████████████████████████████████████████████████████████
# ██ TEXT DISPLAY
# ████████████████████████████████████████████████████████████████████

def fix_rtl_text(text: str) -> str:
    """Reorder and reshape RTL text for correct display in LTR environments."""
    lines = text.split("\n")
    fixed = [get_display(arabic_reshaper.reshape(line)) for line in lines]
    return "\n".join(fixed)


def detect_language(text: str) -> str:
    """Return 'Arabic' if the text is predominantly Arabic, otherwise 'English'."""
    arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
    ratio = arabic_chars / max(len(text.strip()), 1)
    return "Arabic" if ratio > 0.15 else "English"


# ████████████████████████████████████████████████████████████████████
# ██ FILE HELPERS
# ████████████████████████████████████████████████████████████████████

def validate_file_path(file_path: Path) -> bool:
    """
    Check whether a document path exists and points to a supported file type.

    To be implemented later.
    """
    pass


def get_file_extension(file_path: Path) -> str:
    """
    Return the normalized file extension for a document path.

    To be implemented later.
    """
    pass
