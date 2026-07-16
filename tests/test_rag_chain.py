from pathlib import Path

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.rag.chain import ask, build_qa_chain
from src.vectorstore.chroma_store import add_documents


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(size=384)


@pytest.fixture
def indexed_store(fake_embeddings, tmp_path: Path) -> Path:
    chroma_dir = tmp_path / "chroma"
    docs = [
        Document(
            page_content="The Transformer uses multi-head self-attention.",
            metadata={"page": 2, "source": "attention.pdf"},
        ),
        Document(
            page_content="Training used Adam with a warmup learning-rate schedule.",
            metadata={"page": 7, "source": "attention.pdf"},
        ),
    ]
    add_documents(docs, fake_embeddings, persist_dir=chroma_dir)
    return chroma_dir


def test_ask_returns_answer_and_sources(fake_embeddings, indexed_store):
    llm = FakeListChatModel(responses=["The model uses multi-head self-attention."])

    result = ask(
        "What attention mechanism is used?",
        llm=llm,
        embeddings=fake_embeddings,
        persist_dir=indexed_store,
        k=2,
    )

    assert "self-attention" in result["answer"].lower()
    assert result["question"] == "What attention mechanism is used?"
    assert len(result["sources"]) == 2


def test_ask_rejects_empty_question(fake_embeddings, indexed_store):
    llm = FakeListChatModel(responses=["unused"])

    with pytest.raises(ValueError, match="empty"):
        ask(
            "   ",
            llm=llm,
            embeddings=fake_embeddings,
            persist_dir=indexed_store,
        )


def test_build_qa_chain_invokes(fake_embeddings, indexed_store):
    llm = FakeListChatModel(responses=["Based on the document, it uses self-attention."])
    chain = build_qa_chain(
        llm=llm,
        embeddings=fake_embeddings,
        persist_dir=indexed_store,
        k=1,
    )

    answer = chain.invoke("What is used?")
    assert "self-attention" in answer.lower()
