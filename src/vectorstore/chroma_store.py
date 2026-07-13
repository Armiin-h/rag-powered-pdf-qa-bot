from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings


def get_vectorstore(
    embeddings: Embeddings,
    *,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Open or create a persisted Chroma collection."""
    return Chroma(
        collection_name=collection_name or settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir or settings.chroma_persist_dir),
    )


def add_documents(
    documents: list[Document],
    embeddings: Embeddings,
    *,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Embed document chunks and store them in ChromaDB."""
    if not documents:
        raise ValueError("No documents to index")

    store = get_vectorstore(
        embeddings,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    store.add_documents(documents)
    return store


def search_similar(
    query: str,
    embeddings: Embeddings,
    *,
    k: int | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> list[Document]:
    """Return the most similar indexed chunks for a query."""
    store = get_vectorstore(
        embeddings,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    return store.similarity_search(query, k=k or settings.top_k)
