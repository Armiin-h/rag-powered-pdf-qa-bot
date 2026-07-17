from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ui.helpers import save_uploaded_pdf, source_label, source_snippet


def test_save_uploaded_pdf_writes_file(tmp_path: Path):
    data = b"%PDF-1.4 minimal"
    path = save_uploaded_pdf(data, "notes.pdf", upload_dir=tmp_path)

    assert path.exists()
    assert path.name == "notes.pdf"
    assert path.read_bytes() == data


def test_save_uploaded_pdf_rejects_non_pdf(tmp_path: Path):
    with pytest.raises(ValueError, match="PDF"):
        save_uploaded_pdf(b"hello", "notes.txt", upload_dir=tmp_path)


def test_source_label_and_snippet():
    doc = Document(
        page_content="A" * 400,
        metadata={"page": 5, "source": "manual.pdf"},
    )

    assert source_label(doc) == "manual.pdf (page 5)"
    assert source_snippet(doc, max_len=50).endswith("...")
    assert len(source_snippet(doc, max_len=50)) == 53
