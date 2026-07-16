from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a careful document Q&A assistant.
Answer the user's question using ONLY the provided context from the PDF.
If the context does not contain enough information, say you don't know based on the document.
Do not invent facts that are not supported by the context.
Keep answers concise and cite page numbers when available."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:",
        ),
    ]
)


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string."""
    if not docs:
        return "No relevant context found."

    parts: list[str] = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "document")
        parts.append(f"[source={source} | page={page}]\n{doc.page_content}")
    return "\n\n".join(parts)
