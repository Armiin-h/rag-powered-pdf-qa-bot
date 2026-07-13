from pathlib import Path

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from src.vectorstore.chroma_store import add_documents, get_vectorstore, search_similar


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=384)


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


def test_add_documents_persists_chunks(fake_embeddings, chroma_dir):
    docs = [
        Document(page_content="Transformers use self-attention.", metadata={"page": 1}),
        Document(page_content="Positional encoding adds sequence order.", metadata={"page": 2}),
    ]

    add_documents(docs, fake_embeddings, persist_dir=chroma_dir)
    reopened = get_vectorstore(fake_embeddings, persist_dir=chroma_dir)
    stored = reopened.get()

    assert len(stored["ids"]) == 2


def test_search_similar_returns_relevant_chunk(fake_embeddings, chroma_dir):
    docs = [
        Document(page_content="The model uses multi-head self-attention.", metadata={"page": 3}),
        Document(page_content="Training used the Adam optimizer.", metadata={"page": 8}),
    ]
    add_documents(docs, fake_embeddings, persist_dir=chroma_dir)

    results = search_similar(
        "How does attention work?",
        fake_embeddings,
        k=1,
        persist_dir=chroma_dir,
    )

    assert len(results) == 1
    assert results[0].page_content in {doc.page_content for doc in docs}


def test_add_documents_rejects_empty_list(fake_embeddings, chroma_dir):
    with pytest.raises(ValueError, match="No documents to index"):
        add_documents([], fake_embeddings, persist_dir=chroma_dir)
