from pathlib import Path

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from src.indexing.pipeline import index_pdf, search_index
from src.vectorstore.chroma_store import add_documents


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=384)


def test_index_pdf_indexes_chunks(monkeypatch, fake_embeddings, tmp_path):
    chroma_dir = tmp_path / "chroma"
    pages = [
        Document(page_content="Attention is all you need.", metadata={"page": 1}),
        Document(page_content="We propose the Transformer.", metadata={"page": 2}),
    ]
    chunks = pages + [
        Document(page_content="Self-attention layers are fast.", metadata={"page": 2}),
    ]

    monkeypatch.setattr("src.indexing.pipeline.load_pdf", lambda _path: pages)
    monkeypatch.setattr("src.indexing.pipeline.split_documents", lambda _pages: chunks)

    stats = index_pdf("paper.pdf", embeddings=fake_embeddings, persist_dir=chroma_dir)

    assert stats["chunks_indexed"] == 3
    assert stats["pages"] == 2
    assert stats["persist_dir"] == str(chroma_dir)


def test_search_index_returns_matches(fake_embeddings, tmp_path):
    chroma_dir = tmp_path / "chroma"
    docs = [
        Document(page_content="Scaled dot-product attention.", metadata={"page": 4}),
        Document(page_content="Learning rate warmup schedule.", metadata={"page": 9}),
    ]
    add_documents(docs, fake_embeddings, persist_dir=chroma_dir)

    results = search_index(
        "attention mechanism",
        embeddings=fake_embeddings,
        k=1,
        persist_dir=chroma_dir,
    )

    assert len(results) == 1
    assert results[0].page_content in {doc.page_content for doc in docs}
