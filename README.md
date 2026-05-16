# AI Document Assistant

A local RAG (Retrieval-Augmented Generation) application that lets you upload a document, ask questions about it in a conversational chat, and get a structured summary — powered by the Groq API and running through a Streamlit web UI.

![Start Page](<screenshots/start page.png>)

---

## Features

### Q&A
Upload a TXT, PDF, or DOCX file and ask anything about it. The assistant retrieves the most relevant sections and generates a grounded answer using only the document content — never general knowledge.

- Answers are grounded exclusively in the document
- Multi-turn conversation memory (up to 5 turns)
- Responds in the same language as the question (Arabic or English)
- Deterministic tool calling for date and financial calculations

![Q&A Answers](<screenshots/answers in the Q&A.png>)

### Summarization
Generate a structured summary shaped to the document type:

| Type | What the summary covers |
|---|---|
| Article | Main idea, key points, supporting details, conclusion |
| Story | Overview, key events, characters, main message |
| Contract | Parties, key terms, obligations, conditions, clauses |
| Technical | Topic, key concepts, how it works, findings |

![Document Type Selection](<screenshots/document type.png>)

![Summary Result](<screenshots/summary result.png>)

### Document Preview
Click the filename badge after uploading to open an inline preview:
- **PDF** — browser's native PDF viewer via base64 iframe
- **DOCX** — converted to styled HTML preserving headings, bold/italic, tables, and Arabic RTL
- **TXT** — raw text in a scrollable area

![Document Preview](<screenshots/document preview.png>)

### Scanned PDF Support
If a PDF has no selectable text, the app automatically falls back to Tesseract OCR. Supports English and Arabic.

---

## Supported File Types

| Format | Extension | Notes |
|---|---|---|
| Plain text | `.txt` | UTF-8 encoded |
| PDF | `.pdf` | Digital and scanned (OCR fallback) |
| Word document | `.docx` | Microsoft Word format |

---

## Architecture

The app follows a strict clean architecture rule: **no module imports from its siblings**. All wiring flows through `rag_pipeline.py`. The UI only calls `rag_pipeline.py`.

```
ui/app.py
    └── calls only → rag_pipeline.py
                         ├── loader.py
                         ├── chunking.py
                         ├── embeddings.py
                         ├── retrieval.py
                         ├── llm.py
                         │     ├── tools.py
                         │     └── (tool_gate via rag_pipeline)
                         ├── tool_gate.py
                         └── utils.py
```

---

## How It Works

### RAG Pipeline (Q&A)

#### Ingestion — runs once per uploaded file
```
load_document()         → raw text string
split_text_into_chunks()→ overlapping 1000-char pieces (100-char overlap)
create_embeddings()     → 384-dim vectors via all-MiniLM-L6-v2 (local)
store_embeddings()      → stored in VectorStore (in-memory, per session)
```

#### Answering — runs on every question
```
create_query_embedding()        → embed the question
retrieve_relevant_chunks()      → cosine similarity → top 5 chunks
should_use_tools(question)      → regex gate → True or False
    │
    ├── False → standard RAG path
    │             build_prompt(chunks + question) → 1 API call → answer
    │
    └── True  → tool path (see below)
```

### Summarization Pipeline
```
load_document() → split_text_into_chunks() → detect_language()
→ summarize_document(all chunks, doc_type) → 1 API call → summary
```
No embedding or retrieval. All chunks sent at once (capped at 16,000 chars).

---

## Tool Calling

For questions involving dates, arithmetic, or financial calculations, the pipeline uses deterministic Python tools instead of letting the LLM compute the answer. This eliminates arithmetic hallucination.

### How it works

```
API call 1: LLM reads document + tool schemas → decides which tool to call
Tool runs:  Python executes the function → exact result
    │
    ├── Pure deterministic tools (days_until, get_weekday)
    │     → return result directly, skip API call 2
    │
    └── Other tools (calculator, financial)
          → inject verbatim instruction into system message
          → API call 2: LLM frames result in natural language
          → validate: check exact number appears in response
          → if not: return tool result directly as fallback
```

### Available tools

| Tool | What it computes |
|---|---|
| `days_until(date_string)` | Days from today to a date |
| `get_weekday(date_string)` | Day of the week for a date |
| `calculator(expression)` | Safe arithmetic via AST evaluator |
| `percentage_change(old, new)` | `(new - old) / abs(old) × 100` |
| `prorated_amount(full, days_used, total_days)` | `full × (days_used / total_days)` |
| `amount_after_rate(base, rate%)` | `base × (1 + rate/100)` — tax, VAT, markup |
| `amount_after_discount(base, discount%)` | `base × (1 - rate/100)` — discounts |

All tools accept dates in multiple formats: `2032-02-06`, `February 6, 2032`, `6 February 2032`, `Feb 6, 2032`, etc.

### Tool gate

Before exposing tools to the LLM, a regex gate (`tool_gate.py`) checks the question against ~20 patterns:
- Date/time: "how many days", "days remaining", "time until"
- Day of week: "what day", "falls on", "day of the week"
- Arithmetic: "calculate", "compute", "total", "how much is"
- Financial: "percentage change", "prorated", "after tax", "with VAT", "net amount", "gross total", "after discount"

Returns `True` → tools exposed. Returns `False` → standard RAG, no tools.

### Reliability hardening

| Layer | Purpose |
|---|---|
| `_VERBATIM_INSTRUCTION` | Tells LLM to quote tool result exactly, not recompute |
| `_PURE_DETERMINISTIC_TOOLS` | Skips API call 2 entirely for date tools |
| Validation + fallback | If LLM drifts, tool result is returned directly |

---

## Project Structure

```
ai-document-assistant/
├── app/
│   ├── rag_pipeline.py     # Orchestrator — wires all modules
│   ├── loader.py           # File loading (TXT/PDF/DOCX) + OCR + DOCX HTML preview
│   ├── chunking.py         # Text cleaning and splitting
│   ├── embeddings.py       # Vector generation (sentence-transformers)
│   ├── retrieval.py        # VectorStore + cosine similarity search
│   ├── llm.py              # Prompt building + LLM calls + tool loop
│   ├── tools.py            # Deterministic tool functions + JSON schemas
│   ├── tool_gate.py        # Regex gate for tool activation
│   ├── utils.py            # detect_language, fix_rtl_text
│   └── main.py             # CLI entry point (dev use)
│
├── ui/
│   └── app.py              # Streamlit UI (Summary + Q&A tabs)
│
├── data/
│   └── documents/          # Uploaded files saved here
│
├── tests/
│   └── test_basic.py
│
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

| File | Owns | Must not |
|---|---|---|
| `loader.py` | Reading files, extracting raw text | Know about chunks or embeddings |
| `chunking.py` | Splitting and cleaning text | Know about files or embeddings |
| `embeddings.py` | Creating vectors from text | Know about storage or retrieval |
| `retrieval.py` | VectorStore, storing, searching | Know about files, LLM, or tools |
| `llm.py` | Prompts and LLM API calls | Know about files or vectors |
| `tools.py` | Deterministic computation functions | Have side effects or state |
| `tool_gate.py` | Yes/no tool decision | Use LLM or external calls |
| `rag_pipeline.py` | Orchestrating all of the above | Contain domain logic itself |
| `ui/app.py` | Streamlit rendering | Call any module except rag_pipeline |

---

## Setup

### Prerequisites
- Python 3.12+
- A free [Groq API key](https://console.groq.com)
- Conda or any virtual environment manager

### 1. Create and activate environment

```bash
conda create -n ai-project python=3.12
conda activate ai-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 4. OCR for scanned PDFs (optional)

Install system tools:
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/)

Install Python wrappers:
```bash
pip install pytesseract pdf2image
```

Update paths in [app/loader.py](app/loader.py):
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\path\to\tesseract.exe"
_POPPLER_PATH = r"C:\path\to\poppler\bin"
```

---

## Running the App

```bash
python -m streamlit run ui/app.py
```

---

## Running Tests

```bash
pytest
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Web interface |
| LLM | Groq API — `llama-3.1-8b-instant` | Answer generation |
| LLM client | `openai` package | OpenAI-compatible Groq endpoint |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local 384-dim vectors |
| Vector search | NumPy cosine similarity | In-memory retrieval |
| PDF parsing | `pypdf` | Text extraction |
| DOCX parsing | `python-docx` | Text + HTML preview |
| OCR | Tesseract + `pytesseract` + `pdf2image` | Scanned PDF support |
| Language detection | Unicode character ratio | Zero dependencies |
| RTL text | `arabic-reshaper` + `python-bidi` | Arabic display |
| Testing | Pytest | Unit tests |

---

## Design Principles

- **Single responsibility** — each module owns exactly one concern
- **One orchestrator** — only `rag_pipeline.py` wires modules together
- **No global state** — `VectorStore` is created per session and passed explicitly
- **Grounded answers** — LLM is constrained to document context only
- **Deterministic computation** — arithmetic and dates are handled by Python, not the LLM
- **Graceful fallback** — tool path failures fall back to standard RAG automatically
- **Bilingual** — language detection is automatic, no user configuration needed
