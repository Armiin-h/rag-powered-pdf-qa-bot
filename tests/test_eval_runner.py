from pathlib import Path

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.evaluation.runner import run_evaluation
from src.vectorstore.chroma_store import add_documents


def test_run_evaluation_with_fake_models(tmp_path: Path):
    chroma_dir = tmp_path / "chroma"
    dataset = tmp_path / "eval.json"
    dataset.write_text(
        """[
          {
            "question": "What attention is used?",
            "expected_keywords": ["attention"],
            "expected_pages": [2]
          }
        ]""",
        encoding="utf-8",
    )

    docs = [
        Document(
            page_content="Multi-head self-attention is used.",
            metadata={"page": 2, "source": "paper.pdf"},
        )
    ]
    embeddings = FakeEmbeddings(size=384)
    add_documents(docs, embeddings, persist_dir=chroma_dir)

    llm = FakeListChatModel(responses=["The model uses self-attention."])
    report = run_evaluation(
        dataset,
        llm=llm,
        embeddings=embeddings,
        persist_dir=chroma_dir,
        k=1,
    )

    assert report["summary"]["cases"] == 1
    assert report["results"][0]["keyword_recall"] == 1.0
