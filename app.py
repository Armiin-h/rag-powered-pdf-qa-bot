"""Streamlit chat UI for PDF document Q&A."""

import streamlit as st

from src.config import settings
from src.embeddings import get_embeddings
from src.indexing import index_pdf
from src.rag import ask
from src.ui.helpers import (
    SESSION_INDEXED,
    SESSION_MESSAGES,
    SESSION_TOP_K,
    init_session_state,
    save_uploaded_pdf,
    source_label,
    source_snippet,
)
from src.vectorstore import clear_vectorstore


def render_sidebar() -> None:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    st.session_state[SESSION_TOP_K] = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=int(st.session_state[SESSION_TOP_K]),
    )

    if uploaded is not None:
        if st.button("Index document", type="primary", use_container_width=True):
            _index_upload(uploaded)

    if st.session_state[SESSION_INDEXED]:
        st.success(f"Indexed: **{st.session_state[SESSION_INDEXED]}**")
    else:
        st.info("Upload and index a PDF to start chatting.")

    st.divider()
    st.caption("Models (Ollama)")
    st.text(f"Embed: {settings.embedding_model}")
    st.text(f"Chat:  {settings.chat_model}")


def _index_upload(uploaded) -> None:
    try:
        pdf_path = save_uploaded_pdf(uploaded.getvalue(), uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        return

    embeddings = get_embeddings()
    if st.session_state[SESSION_INDEXED] != uploaded.name:
        removed = clear_vectorstore(embeddings)
        if removed:
            st.caption(f"Cleared {removed} previous chunk(s) from the index.")

    with st.spinner("Indexing document… this may take a minute."):
        try:
            stats = index_pdf(pdf_path)
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            return

    st.session_state[SESSION_INDEXED] = uploaded.name
    st.session_state[SESSION_MESSAGES] = []
    st.success(
        f"Indexed **{stats['source']}** — {stats['pages']} pages, "
        f"{stats['chunks_indexed']} chunks."
    )


def render_chat() -> None:
    st.title("PDF Q&A")
    st.caption("Answers are grounded in your uploaded document only.")

    for message in st.session_state[SESSION_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for i, doc in enumerate(message["sources"], start=1):
                        st.markdown(f"**[{i}] {source_label(doc)}**")
                        st.caption(source_snippet(doc))

    if prompt := st.chat_input("Ask a question about the document"):
        if not st.session_state[SESSION_INDEXED]:
            st.warning("Index a PDF in the sidebar before asking questions.")
            return

        st.session_state[SESSION_MESSAGES].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = ask(prompt, k=int(st.session_state[SESSION_TOP_K]))
                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")
                    return

            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for i, doc in enumerate(result["sources"], start=1):
                        st.markdown(f"**[{i}] {source_label(doc)}**")
                        st.caption(source_snippet(doc))

        st.session_state[SESSION_MESSAGES].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )


def main() -> None:
    st.set_page_config(
        page_title="PDF Q&A",
        page_icon="📄",
        layout="wide",
    )
    init_session_state()
    with st.sidebar:
        render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
