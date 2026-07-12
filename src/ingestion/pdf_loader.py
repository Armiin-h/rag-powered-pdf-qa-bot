from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Extract text from a PDF, one LangChain Document per page."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    if not pages:
        raise ValueError(f"No text could be extracted from: {path.name}")

    source = path.name
    for page in pages:
        page.metadata["source"] = source
        page.metadata["page"] = int(page.metadata.get("page", 0)) + 1

    return pages
