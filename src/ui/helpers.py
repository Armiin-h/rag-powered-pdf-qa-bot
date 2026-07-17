from pathlib import Path

from langchain_core.documents import Document

from src.config import settings

SESSION_MESSAGES = "messages"
SESSION_INDEXED = "indexed_source"
SESSION_TOP_K = "top_k"


def init_session_state() -> None:
    import streamlit as st

    if SESSION_MESSAGES not in st.session_state:
        st.session_state[SESSION_MESSAGES] = []
    if SESSION_INDEXED not in st.session_state:
        st.session_state[SESSION_INDEXED] = None
    if SESSION_TOP_K not in st.session_state:
        st.session_state[SESSION_TOP_K] = settings.top_k


def save_uploaded_pdf(file_bytes: bytes, filename: str, upload_dir: Path | None = None) -> Path:
    """Write uploaded PDF bytes to disk and return the saved path."""
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")

    target_dir = upload_dir or settings.upload_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / Path(filename).name
    dest.write_bytes(file_bytes)
    return dest


def source_label(doc: Document) -> str:
    page = doc.metadata.get("page", "?")
    source = doc.metadata.get("source", "document")
    return f"{source} (page {page})"


def source_snippet(doc: Document, max_len: int = 280) -> str:
    text = doc.page_content.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."
