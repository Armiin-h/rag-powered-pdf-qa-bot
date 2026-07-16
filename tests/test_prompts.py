from langchain_core.documents import Document

from src.rag.prompts import SYSTEM_PROMPT, format_docs


def test_format_docs_includes_page_and_source():
    docs = [
        Document(
            page_content="Self-attention computes weighted sums.",
            metadata={"page": 3, "source": "paper.pdf"},
        )
    ]

    text = format_docs(docs)

    assert "page=3" in text
    assert "source=paper.pdf" in text
    assert "Self-attention computes weighted sums." in text


def test_format_docs_handles_empty_list():
    assert format_docs([]) == "No relevant context found."


def test_system_prompt_requires_grounded_answers():
    assert "ONLY" in SYSTEM_PROMPT
    assert "don't know" in SYSTEM_PROMPT.lower()
