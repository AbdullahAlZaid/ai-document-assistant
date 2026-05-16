"""Application entry point for the AI document assistant."""

import sys
from pathlib import Path

from dotenv import load_dotenv

from app.loader import load_document
from app.rag_pipeline import summarize_file
from app.utils import fix_rtl_text

load_dotenv()


def run_app() -> None:
    """Load a sample document and print its contents."""
    sys.stdout.reconfigure(encoding="utf-8")
    file_path = Path(r"C:\Users\abinzaid.t\Downloads\_عقد_توريد_معدات_v1.pdf")
    content = load_document(file_path)
    print(fix_rtl_text(content))


def run_summary_test() -> None:
    """Summarize sample.txt and print the result."""
    sys.stdout.reconfigure(encoding="utf-8")
    file_path = Path("data/documents/sample.txt")
    print("=== Document Summary ===")
    print(summarize_file(file_path))


def main() -> None:
    """Execute the application from the command line."""
    run_summary_test()


if __name__ == "__main__":
    main()
