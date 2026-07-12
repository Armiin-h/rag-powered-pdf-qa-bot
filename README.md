# RAG-powered PDF Q&A bot

A retrieval-augmented generation (RAG) application that lets you upload large PDF documents and ask questions answered strictly from the document's content.

**Repository:** [github.com/Armiin-h/rag-powered-pdf-qa-bot](https://github.com/Armiin-h/rag-powered-pdf-qa-bot)

## Features

- PDF text extraction and intelligent chunking
- Vector embeddings stored in a local vector database
- Semantic search over document chunks
- LLM-powered answers grounded in retrieved context
- Simple chat interface for document Q&A

## Tech Stack

| Layer | Tool |
|-------|------|
| PDF parsing | PyPDF |
| Orchestration | LangChain |
| Embeddings & LLM | Ollama (local) |
| Vector store | ChromaDB |
| UI | Streamlit |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- 8 GB RAM minimum (16 GB recommended)

Pull required models after installing Ollama:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # adjust settings if needed
```

## Usage

### Test PDF ingestion (Day 1)

```bash
python scripts/ingest_pdf.py path\to\your\document.pdf
```

This loads the PDF, splits it into chunks, and prints ingestion stats plus a short preview.

### Full app (coming soon)

```bash
streamlit run app.py
```

## Project Structure

```
src/
  config.py              # Environment-based settings
  ingestion/
    pdf_loader.py        # PyPDF text extraction
    text_splitter.py     # Recursive character splitting
scripts/
  ingest_pdf.py          # CLI to preview ingestion
```

## Project Status

| Day | Milestone | Status |
|-----|-----------|--------|
| 1 | PDF loader + text splitter | Done |
| 2 | Embeddings + ChromaDB | Planned |
| 3 | Retrieval chain + prompt | Planned |
| 4 | Streamlit UI | Planned |

## License

MIT
