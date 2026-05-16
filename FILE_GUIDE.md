# File Guide — AI Document Assistant

Every file in the project, what it does, and how it connects to others.

---

## Architecture rule (read this first)

No file in `app/` imports from its siblings. The only file that knows about all modules is
`rag_pipeline.py`. The UI (`ui/app.py`) only calls `rag_pipeline.py`. This keeps every module
independently testable and replaceable.

```
ui/app.py
    └── calls → rag_pipeline.py
                    ├── loader.py
                    ├── chunking.py
                    ├── embeddings.py
                    ├── retrieval.py
                    ├── llm.py
                    │     ├── tools.py
                    │     └── tool_gate.py (via rag_pipeline)
                    └── utils.py
```

---

## `app/rag_pipeline.py` — The Orchestrator

**What it does:** Wires every module together. Contains no domain logic itself — only calls other modules in the right order.

**Key functions:**
- `ingest_document(file_path, store)` — load → chunk → embed → store
- `answer_question_about_document(store, question, history)` — retrieve → gate check → generate
- `summarize_file(file_path, doc_type)` — load → chunk → summarize
- `get_document_text(file_path)` — raw text for TXT preview
- `get_document_preview_html(file_path)` — HTML for DOCX preview

**Connects to:** every other `app/` module. It is the only file that does so.

**Connected from:** `ui/app.py` only.

---

## `app/loader.py` — Document Loading

**What it does:** Reads a file from disk and returns its content as a plain Python string. Also converts DOCX to styled HTML for the browser preview.

**Key functions:**
- `load_document(file_path)` — dispatcher: routes to the right loader by file extension
- `load_txt_file(file_path)` — `Path.read_text(encoding="utf-8")`
- `load_pdf_file(file_path)` — `pypdf.PdfReader`; if extracted text < 200 chars, falls back to OCR
- `load_docx_file(file_path)` — `python-docx`, joins all paragraph texts
- `_ocr_pdf(file_path)` — Tesseract OCR for scanned PDFs, supports English + Arabic
- `load_docx_as_html(file_path)` — converts DOCX paragraphs, headings, and tables to HTML

**Connects to:** nothing inside `app/`. Uses `pypdf`, `python-docx`, `pytesseract`, `pdf2image`.

**Connected from:** `rag_pipeline.py` calls `load_document` and `load_docx_as_html`.

---

## `app/chunking.py` — Text Splitting

**What it does:** Cleans raw text and splits it into overlapping fixed-size pieces ready for embedding.

**Key functions:**
- `clean_chunk_text(text)` — normalizes line endings, collapses excess blank lines
- `split_text_into_chunks(text, chunk_size=1000, overlap=100)` — sliding window over characters

**How overlap works:** With chunk_size=1000 and overlap=100, the window moves 900 characters each step. The 100-character tail of each chunk is repeated at the start of the next, so sentences at boundaries are never cut off and lost.

**Connects to:** nothing. Pure string logic, no imports from `app/`.

**Connected from:** `rag_pipeline.py` calls `split_text_into_chunks` after loading.

---

## `app/embeddings.py` — Vector Generation

**What it does:** Converts text strings into numerical vectors using a local sentence-transformer model. Vectors represent meaning — similar texts produce similar vectors.

**Key functions:**
- `create_embeddings(texts)` — batch encodes a list of chunk strings → list of 384-number vectors
- `create_query_embedding(question)` — encodes the user's question into one vector

**Model:** `all-MiniLM-L6-v2` — runs locally on CPU, no API cost. Loaded once and cached in a module-level variable.

**Connects to:** nothing inside `app/`. Uses `sentence-transformers`.

**Connected from:** `rag_pipeline.py` calls `create_embeddings` during ingestion and `create_query_embedding` before retrieval.

---

## `app/retrieval.py` — Vector Store & Search

**What it does:** Stores chunks and their vectors in memory, and finds the most relevant chunks for a given question using cosine similarity.

**Key components:**
- `VectorStore` — a dataclass with two parallel lists: `chunks: list[str]` and `embeddings: list[list[float]]`. Index `i` in both lists always refers to the same piece of text.
- `store_embeddings(store, chunks, embeddings)` — populates the store
- `retrieve_relevant_chunks(store, question_embedding, top_k=5)` — normalizes vectors, computes dot products with numpy, returns the 5 highest-scoring chunks

**Cosine similarity in one line:**
```python
similarities = matrix_norm @ query_norm  # dot product of unit vectors = cosine similarity
```

**Connects to:** `numpy` only. No other `app/` modules.

**Connected from:** `rag_pipeline.py` calls `store_embeddings` during ingestion and `retrieve_relevant_chunks` before answering. `ui/app.py` imports `VectorStore` directly (it is a data container, not logic).

---

## `app/llm.py` — LLM Calls

**What it does:** Builds prompts and sends them to the Groq API (LLaMA 3.1 via OpenAI-compatible client). Handles both standard RAG answers and the full tool-calling loop.

**Key functions:**
- `build_prompt(chunks, question, language)` — assembles context + question into a single prompt string
- `generate_answer(chunks, question, language, history)` — one API call, multi-turn aware
- `generate_answer_with_tools(chunks, question, language, history)` — full two-call tool loop with hardening
- `summarize_document(chunks, doc_type, language)` — structured summarization with typed templates

**Tool loop inside `generate_answer_with_tools`:**
1. API call 1 — LLM decides whether to call a tool
2. If tool called → execute it in Python → get exact result
3. If the tool is `days_until` or `get_weekday` → return result directly, skip call 2
4. Otherwise → inject verbatim instruction → API call 2 → validate result → return

**Key constants:**
- `_VERBATIM_INSTRUCTION` — tells the LLM not to recompute tool results
- `_PURE_DETERMINISTIC_TOOLS` — frozenset of tools that bypass API call 2
- `_SCOPE_INSTRUCTION` — tells the LLM not to add unsolicited extras

**Connects to:** `tools.py` (imports `TOOL_FUNCTIONS`, `TOOL_SCHEMAS`). Uses `openai` package pointed at Groq's base URL.

**Connected from:** `rag_pipeline.py` calls `generate_answer`, `generate_answer_with_tools`, and `summarize_document`.

---

## `app/tools.py` — Deterministic Tool Functions

**What it does:** Defines every Python function the LLM is allowed to call, plus their JSON schemas that describe them to the API.

**Tools:**

| Function | Formula / Logic |
|---|---|
| `days_until(date_string)` | `(target - today).days` |
| `get_weekday(date_string)` | `target.strftime('%A')` |
| `calculator(expression)` | AST-based safe evaluator (no `eval`) |
| `percentage_change(old, new)` | `(new - old) / abs(old) * 100` |
| `prorated_amount(full, days_used, total_days)` | `full * (days_used / total_days)` |
| `amount_after_rate(base, rate%)` | `base * (1 + rate/100)` — for tax/VAT/markup |
| `amount_after_discount(base, discount%)` | `base * (1 - rate/100)` — for discounts |

**`_parse_date(date_string)`** — helper used by `days_until` and `get_weekday`. Tries 9 date formats in sequence (`YYYY-MM-DD`, `February 6, 2032`, `6 February 2032`, etc.) so documents with natural language dates work correctly.

**`TOOL_FUNCTIONS`** — dict mapping name → callable, used by `llm.py` to dispatch calls.

**`TOOL_SCHEMAS`** — list of OpenAI-compatible JSON schemas sent to the API, used by the LLM to decide which tool to call and what arguments to pass.

**Connects to:** nothing inside `app/`. Pure Python stdlib (`ast`, `datetime`, `math`).

**Connected from:** `llm.py` imports `TOOL_FUNCTIONS` and `TOOL_SCHEMAS`.

---

## `app/tool_gate.py` — Tool Decision Gate

**What it does:** Decides whether to expose tools to the LLM for a given question. Returns True or False based purely on regex matching — no LLM, no API call.

**Key function:**
- `should_use_tools(question)` — runs the question against ~20 compiled regex patterns

**Pattern categories:**
- Date/time: "how many days", "days remaining", "time until"
- Day of week: "what day", "falls on", "day of the week"
- Arithmetic: "calculate", "compute", "total", "how much is"
- Financial: "percentage change", "prorated", "after tax", "with VAT", "net amount", "gross total"

**Why it exists:** Sending tool schemas to the LLM on every question wastes tokens and risks the LLM calling tools it doesn't need. The gate is fast, deterministic, and independently testable.

**Connects to:** nothing inside `app/`. Uses `re` from stdlib.

**Connected from:** `rag_pipeline.py` calls `should_use_tools` before deciding which generation path to take.

---

## `app/utils.py` — Shared Helpers

**What it does:** Small utility functions used by different parts of the app.

**Key functions:**
- `detect_language(text)` — counts characters in the Arabic Unicode range (`U+0600–U+06FF`). If more than 15% are Arabic, returns `"Arabic"`, otherwise `"English"`.
- `fix_rtl_text(text)` — applies Arabic reshaping and the BiDi algorithm for correct terminal display of Arabic text.

**Connects to:** uses `arabic-reshaper` and `python-bidi`.

**Connected from:** `rag_pipeline.py` calls `detect_language` on the question (for Q&A) and on the document text (for summarization). The detected language is passed into `llm.py` prompts.

---

## `ui/app.py` — Streamlit UI

**What it does:** The entire frontend. Two tabs: Summary and Q&A. Handles file upload, session state, chat rendering, and document preview.

**Key functions:**
- `_init_session_state()` — initializes 8 session state keys (VectorStore, filename, history, preview toggle, etc.)
- `_save_uploaded_file(uploaded_file)` — writes the in-memory upload to `data/documents/` so backend functions can open it by path
- `_render_summary_page()` — summary tab: upload → summarize → display
- `_render_qa_page()` — Q&A tab: upload → ingest → chat loop
- `_render_document_preview(filename)` — embeds PDF/DOCX as base64 `data:` URI in an `<iframe>`, or shows TXT in a text area
- `_file_pill_clickable(filename, preview_key)` — renders the clickable filename badge that toggles preview

**Session state keys:**

| Key | Holds |
|---|---|
| `qa_store` | The `VectorStore` for the current document |
| `qa_file` | Filename of the indexed document |
| `qa_history` | List of `{"question": ..., "answer": ...}` dicts |
| `qa_preview_open` | Bool — whether preview panel is visible |
| `sum_text` | The generated summary string |
| `sum_file` | Filename of the summarized document |
| `sum_doc_type` | Selected doc type (Article/Story/etc.) |
| `sum_preview_open` | Bool — whether preview panel is visible |

**Connects to:** imports `ingest_document`, `answer_question_about_document`, `summarize_file`, `get_document_text`, `get_document_preview_html` from `rag_pipeline.py`. Also imports `VectorStore` from `retrieval.py` (data container only).

**Connected from:** nothing. This is the entry point.

---

## `app/main.py` — CLI Entry Point

**What it does:** Development/testing entry point. Loads a hardcoded file and prints a summary to the terminal. Not used by the Streamlit app at all.

**Connects to:** `loader.py`, `rag_pipeline.py`, `utils.py`.

---

## `requirements.txt` — Dependencies

| Package | Used by |
|---|---|
| `streamlit` | `ui/app.py` |
| `pypdf` | `loader.py` — PDF text extraction |
| `python-docx` | `loader.py` — DOCX text extraction and HTML preview |
| `sentence-transformers` | `embeddings.py` — local embedding model |
| `openai` | `llm.py` — OpenAI-compatible client for Groq |
| `numpy` | `retrieval.py` — cosine similarity |
| `python-dotenv` | `ui/app.py`, `main.py` — loads `.env` for API key |
| `arabic-reshaper` | `utils.py` — Arabic text reshaping |
| `python-bidi` | `utils.py` — BiDi algorithm for RTL display |
| `pytest` | `tests/` — test runner |

OCR packages (`pytesseract`, `pdf2image`, `Pillow`) are installed separately and not in `requirements.txt` as they require system-level dependencies (Tesseract binary, Poppler).
