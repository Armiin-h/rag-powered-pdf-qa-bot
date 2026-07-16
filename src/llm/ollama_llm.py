from langchain_ollama import ChatOllama

from src.config import settings


def get_chat_model(*, temperature: float = 0.0) -> ChatOllama:
    """Return an Ollama chat model configured from settings."""
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )
