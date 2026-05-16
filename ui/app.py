"""Streamlit UI for the AI document assistant."""

import base64
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so app.* imports resolve when
# Streamlit launches this file directly from the ui/ subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.rag_pipeline import answer_question_about_document, get_document_preview_html, get_document_text, ingest_document, summarize_file
from app.retrieval import VectorStore


# ████████████████████████████████████████████████████████████████████
# ██ STYLES
# ████████████████████████████████████████████████████████████████████

_CSS = """
<style>

/* ── Layout ────────────────────────────────────────────────────────── */
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
}

/* ── Sidebar background ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0f172a;
}

/* ── Sidebar radio → styled as nav menu ─────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    flex-direction: column;
    gap: 2px;
    padding: 0 0.25rem;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center;
    padding: 0.65rem 1rem !important;
    border-radius: 8px !important;
    font-size: 0.93rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    cursor: pointer;
    transition: background 0.12s, color 0.12s;
    margin: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.07) !important;
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(99, 102, 241, 0.18) !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

/* ── Sidebar text defaults ──────────────────────────────────────────── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #64748b;
}

/* ── Page title & subtitle ──────────────────────────────────────────── */
.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* ── File status pill ───────────────────────────────────────────────── */
.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #15803d;
    margin-bottom: 1.25rem;
}

/* ── Empty state ────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    background: #f8fafc;
    border-radius: 14px;
    border: 2px dashed #e2e8f0;
    margin-top: 1rem;
}
.empty-state .es-icon { font-size: 2.2rem; display: block; margin-bottom: 0.75rem; }
.empty-state .es-title {
    font-size: 1rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.35rem;
}
.empty-state .es-hint { font-size: 0.85rem; color: #94a3b8; }

/* ── Result cards ───────────────────────────────────────────────────── */
.summary-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    line-height: 1.85;
    font-size: 0.95rem;
    color: #1e293b;
}
.answer-card {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    line-height: 1.85;
    font-size: 0.95rem;
    color: #1e293b;
}

/* ── Section label above a card ─────────────────────────────────────── */
.card-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 0.6rem;
}

/* ── Custom upload box ──────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    position: relative !important;
    border: 2.5px dashed #94a3b8 !important;
    border-radius: 16px !important;
    background: #f8fafc !important;
    min-height: 200px !important;
    cursor: pointer !important;
    transition: border-color 0.22s, background 0.22s !important;
    overflow: hidden !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6366f1 !important;
    background: #eef2ff !important;
}
/* Hide all default Streamlit inner content (SVG, text, browse button label) */
[data-testid="stFileUploaderDropzone"] > * {
    display: none !important;
}
/* Re-expose the browse button invisibly — it fills the whole box so any click triggers the file picker */
[data-testid="stFileUploaderDropzone"] button {
    display: block !important;
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    cursor: pointer !important;
    z-index: 5 !important;
}
/* Centered file icon */
[data-testid="stFileUploaderDropzone"]::before {
    content: "📄";
    display: block;
    font-size: 3rem;
    line-height: 1;
    position: absolute;
    top: calc(50% - 3.5rem);
    left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
    z-index: 2;
}
/* "Click to upload" instruction */
[data-testid="stFileUploaderDropzone"]::after {
    content: "Click to upload your file";
    display: block;
    font-size: 0.9rem;
    font-weight: 600;
    color: #64748b;
    position: absolute;
    top: calc(50% + 1.2rem);
    left: 50%;
    transform: translateX(-50%);
    pointer-events: none;
    white-space: nowrap;
    z-index: 2;
    letter-spacing: 0.01em;
}

/* ── Chat bubbles ───────────────────────────────────────────────────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
    background: #e0e7ff !important;
    border-radius: 18px 4px 18px 18px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 4px 18px 18px 18px !important;
}

/* ── Uploaded filename badge (below the drop zone) ──────────────────── */
.upload-filename {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 0.35rem 0.85rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #2563eb;
    margin-top: 0.55rem;
}

/* ── Clickable file-pill button (preview trigger) ───────────────────── */
div:has(.preview-pill-marker) + div button {
    display: inline-flex !important;
    align-items: center !important;
    background: #f0fdf4 !important;
    border: 1px solid #86efac !important;
    border-radius: 20px !important;
    padding: 0.3rem 0.9rem !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #15803d !important;
    text-decoration: underline !important;
    text-underline-offset: 2px !important;
    width: auto !important;
    min-height: unset !important;
    line-height: 1.5 !important;
    margin-bottom: 1.25rem !important;
    cursor: pointer !important;
}
div:has(.preview-pill-marker) + div button:hover {
    background: #dcfce7 !important;
    border-color: #4ade80 !important;
    color: #166534 !important;
}
div:has(.preview-pill-marker) + div button:focus:not(:active) {
    box-shadow: none !important;
    outline: none !important;
}

/* ── Document preview panel ─────────────────────────────────────────── */
.preview-panel {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    background: #f8fafc;
}

</style>
"""


# ████████████████████████████████████████████████████████████████████
# ██ SESSION STATE
# ████████████████████████████████████████████████████████████████████

def _init_session_state() -> None:
    """Initialise all session state keys to safe defaults."""
    defaults = {
        "sum_file": None,
        "sum_doc_type": None,
        "sum_text": None,
        "sum_preview_open": False,
        "qa_store": None,
        "qa_file": None,
        "qa_history": [],
        "qa_preview_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ████████████████████████████████████████████████████████████████████
# ██ HELPERS
# ████████████████████████████████████████████████████████████████████

def _check_api_key() -> bool:
    """Return True when GROQ_API_KEY is present in the environment."""
    return bool(os.environ.get("GROQ_API_KEY"))


def _save_uploaded_file(uploaded_file) -> Path:
    """Persist the Streamlit uploaded file to disk and return its path."""
    save_dir = Path("data/documents")
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / uploaded_file.name
    uploaded_file.seek(0)
    file_path.write_bytes(uploaded_file.read())
    return file_path


def _file_pill_clickable(filename: str, preview_key: str) -> None:
    """Render the active filename as a clickable pill that toggles document preview."""
    st.markdown('<div class="preview-pill-marker"></div>', unsafe_allow_html=True)
    if st.button(f"✓  {filename}", key=f"pill_btn_{preview_key}"):
        st.session_state[preview_key] = not st.session_state[preview_key]


def _render_document_preview(filename: str, key_prefix: str) -> None:
    """Display an inline preview of the uploaded document."""
    file_path = Path("data/documents") / filename
    if not file_path.exists():
        st.warning("Preview unavailable — file not found on disk.")
        return

    ext = file_path.suffix.lower()

    if ext == ".pdf":
        b64 = base64.b64encode(file_path.read_bytes()).decode()
        mime = "application/pdf"
    elif ext == ".docx":
        html = get_document_preview_html(file_path)
        b64 = base64.b64encode(html.encode("utf-8")).decode()
        mime = "text/html"
    else:
        text = get_document_text(file_path)
        st.text_area(
            "",
            value=text,
            height=400,
            disabled=True,
            label_visibility="collapsed",
            key=f"preview_ta_{key_prefix}",
        )
        return

    st.markdown(
        f'<div class="preview-panel">'
        f'<iframe src="data:{mime};base64,{b64}" '
        f'width="100%" height="600" style="border:none;display:block;"></iframe>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _upload_filename_badge(filename: str) -> None:
    """Render a blue filename badge directly below the upload drop zone."""
    st.markdown(
        f'<div class="upload-filename">&#128196;&nbsp; {filename}</div>',
        unsafe_allow_html=True,
    )


# ████████████████████████████████████████████████████████████████████
# ██ PAGES
# ████████████████████████████████████████████████████████████████████

def _render_summary_page() -> None:
    """Upload a document and display an AI-generated summary."""
    st.markdown('<div class="page-title">Document Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Select a document type, upload your file, '
        'and get an instant AI-generated summary.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    doc_type = st.selectbox(
        "Document type",
        ["Article", "Story", "Contract", "Technical"],
        help="Choosing the right type gives you a more accurate, structured summary.",
    )

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "docx"],
        key="sum_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        _upload_filename_badge(uploaded_file.name)

    # ── No file, no previous result → empty state ────────────────────
    if uploaded_file is None and st.session_state.sum_text is None:
        st.markdown("""
        <div class="empty-state">
            <span class="es-icon">📄</span>
            <div class="es-title">No document uploaded yet</div>
            <div class="es-hint">Upload a file above and the summary will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── File uploaded → process if new or doc_type changed ──────────
    if uploaded_file is not None:
        file_changed = st.session_state.sum_file != uploaded_file.name
        type_changed = st.session_state.sum_doc_type != doc_type
        if file_changed or type_changed:
            with st.spinner("Summarizing your document..."):
                file_path = _save_uploaded_file(uploaded_file)
                st.session_state.sum_text = summarize_file(file_path, doc_type)
                st.session_state.sum_file = uploaded_file.name
                st.session_state.sum_doc_type = doc_type
                st.session_state.sum_preview_open = False

    # ── RESULT ───────────────────────────────────────────────────────
    if st.session_state.sum_text is not None:
        _file_pill_clickable(st.session_state.sum_file, "sum_preview_open")
        if st.session_state.sum_preview_open:
            _render_document_preview(st.session_state.sum_file, "sum")
        st.markdown('<div class="card-label">Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-card">{st.session_state.sum_text}</div>',
            unsafe_allow_html=True,
        )


def _render_qa_page() -> None:
    """Upload a document and ask questions about it."""
    st.markdown('<div class="page-title">Q&amp;A</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Upload a document, then ask anything '
        'about its content.</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "docx"],
        key="qa_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        _upload_filename_badge(uploaded_file.name)

    # ── No file, no previous store → empty state ────────────────────
    if uploaded_file is None and st.session_state.qa_store is None:
        st.markdown("""
        <div class="empty-state">
            <span class="es-icon">💬</span>
            <div class="es-title">No document uploaded yet</div>
            <div class="es-hint">Upload a file above and start asking questions.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── File uploaded → index if new ────────────────────────────────
    if uploaded_file is not None:
        if st.session_state.qa_file != uploaded_file.name:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                file_path = _save_uploaded_file(uploaded_file)
                store = VectorStore()
                ingest_document(file_path, store)
                st.session_state.qa_store = store
                st.session_state.qa_file = uploaded_file.name
                st.session_state.qa_history = []
                st.session_state.qa_preview_open = False

    # ── CHAT ─────────────────────────────────────────────────────────
    if st.session_state.qa_store is not None:
        _file_pill_clickable(st.session_state.qa_file, "qa_preview_open")
        if st.session_state.qa_preview_open:
            _render_document_preview(st.session_state.qa_file, "qa")

        # ── Message history (scrollable container) ───────────────────
        if st.session_state.qa_history:
            _, clear_col = st.columns([6, 1])
            with clear_col:
                if st.button("Clear", type="secondary"):
                    st.session_state.qa_history = []
                    st.rerun()
            with st.container(height=500):
                for turn in st.session_state.qa_history:
                    with st.chat_message("user"):
                        st.write(turn["question"])
                    with st.chat_message("assistant"):
                        st.write(turn["answer"])

        # ── Input row (always visible below the messages) ────────────
        with st.form("qa_form", clear_on_submit=True):
            input_col, send_col = st.columns([8, 1])
            with input_col:
                question = st.text_input(
                    "",
                    placeholder="Ask a question about your document...",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

        if submitted and question.strip():
            with st.spinner("Thinking..."):
                answer = answer_question_about_document(
                    st.session_state.qa_store,
                    question.strip(),
                    st.session_state.qa_history,
                )
            st.session_state.qa_history.append({
                "question": question.strip(),
                "answer": answer,
            })
            st.rerun()


# ████████████████████████████████████████████████████████████████████
# ██ ENTRY POINT
# ████████████████████████████████████████████████████████████████████

def render_app() -> None:
    """Configure the page and render Summary and Q&A as tabs on the main page."""
    st.set_page_config(
        page_title="AI Document Assistant",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_session_state()

    if not _check_api_key():
        st.error(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file and restart the app."
        )
        st.stop()

    # ── HEADER ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.75rem;">
        <div style="font-size:1.7rem;font-weight:700;color:#0f172a;letter-spacing:-0.02em;">
            AI Document Assistant
        </div>
        <div style="font-size:0.85rem;color:#64748b;margin-top:0.25rem;">
            Powered by Groq &middot; LLaMA 3.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["Summary", "Q&A"])
    with tab1:
        _render_summary_page()
    with tab2:
        _render_qa_page()


if __name__ == "__main__":
    render_app()
