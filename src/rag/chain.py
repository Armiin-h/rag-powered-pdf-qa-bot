from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.config import settings
from src.embeddings import get_embeddings
from src.llm import get_chat_model
from src.rag.prompts import QA_PROMPT, format_docs
from src.vectorstore import get_vectorstore


def get_retriever(
    embeddings: Embeddings | None = None,
    *,
    k: int | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
):
    """Return a Chroma similarity retriever for indexed PDF chunks."""
    emb = embeddings or get_embeddings()
    store = get_vectorstore(
        emb,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    return store.as_retriever(search_kwargs={"k": k or settings.top_k})


def build_qa_chain(
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    *,
    k: int | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> Runnable:
    """Build an LCEL RAG chain that retrieves context and answers a question."""
    retriever = get_retriever(
        embeddings,
        k=k,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    model = llm or get_chat_model()

    return (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | QA_PROMPT
        | model
        | StrOutputParser()
    )


def ask(
    question: str,
    *,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    k: int | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Answer a question from indexed PDF chunks and return sources."""
    if not question.strip():
        raise ValueError("Question must not be empty")

    emb = embeddings or get_embeddings()
    retriever = get_retriever(
        emb,
        k=k,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    sources: list[Document] = retriever.invoke(question)

    chain = (
        {
            "context": lambda _: format_docs(sources),
            "input": RunnablePassthrough(),
        }
        | QA_PROMPT
        | (llm or get_chat_model())
        | StrOutputParser()
    )
    answer = chain.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
