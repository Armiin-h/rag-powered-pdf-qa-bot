from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.embeddings import get_embeddings
from src.ingestion import load_pdf, split_documents
from src.vectorstore import add_documents, search_similar


def index_pdf(
    pdf_path: str | Path,
    embeddings: Embeddings | None = None,
    *,
    persist_dir: Path | None = None,
) -> dict:
    """Load a PDF, chunk it, embed the chunks, and persist them in ChromaDB."""
    path = Path(pdf_path)
    pages = load_pdf(path)
    chunks = split_documents(pages)
    emb = embeddings or get_embeddings()
    add_documents(chunks, emb, persist_dir=persist_dir)
    target_dir = persist_dir or settings.chroma_persist_dir

    return {
        "source": path.name,
        "pages": len(pages),
        "chunks_indexed": len(chunks),
        "collection": settings.collection_name,
        "persist_dir": str(target_dir),
    }


def search_index(
    query: str,
    *,
    k: int | None = None,
    embeddings: Embeddings | None = None,
    persist_dir: Path | None = None,
) -> list[Document]:
    """Search indexed chunks for content similar to the query."""
    emb = embeddings or get_embeddings()
    return search_similar(query, emb, k=k, persist_dir=persist_dir)
