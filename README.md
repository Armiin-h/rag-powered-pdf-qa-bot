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

### Index and search (Day 2)

Index a PDF into the local ChromaDB store (requires Ollama with `nomic-embed-text`):

```bash
python scripts/ingest_pdf.py path\to\your\document.pdf --index
```

Search indexed chunks:

```bash
python scripts/ingest_pdf.py --query "What is self-attention?"
```

Index and search in one run:

```bash
python scripts/ingest_pdf.py path\to\your\document.pdf --index --query "How many layers?"
```

### Ask questions (Day 3)

After indexing, ask grounded questions via the RAG chain (requires Ollama with `llama3.2`):

```bash
python scripts/ask_pdf.py "What is self-attention?"
```

Show retrieved source chunks:

```bash
python scripts/ask_pdf.py "How many encoder layers?" --show-sources
```

Run unit tests (no Ollama required):

```bash
pytest
```

### Full app (coming soon)

```bash
streamlit run app.py
```

## Project Structure

```
src/
  config.py              # Environment-based settings
  embeddings/
    ollama_embeddings.py # Ollama embedding model factory
  ingestion/
    pdf_loader.py        # PyPDF text extraction
    text_splitter.py     # Recursive character splitting
  indexing/
    pipeline.py          # PDF -> chunks -> ChromaDB pipeline
  llm/
    ollama_llm.py        # Ollama chat model factory
  rag/
    prompts.py           # Grounded QA prompt + context formatting
    chain.py             # Retriever + LCEL RAG chain
  vectorstore/
    chroma_store.py      # ChromaDB persistence and search
scripts/
  ingest_pdf.py          # CLI for ingestion, indexing, and search
  ask_pdf.py             # CLI for document Q&A
tests/
  test_chroma_store.py
  test_indexing_pipeline.py
  test_prompts.py
  test_rag_chain.py
```

## Project Status

| Day | Milestone | Status |
|-----|-----------|--------|
| 1 | PDF loader + text splitter | Done |
| 2 | Embeddings + ChromaDB | Done |
| 3 | Retrieval chain + prompt | Done |
| 4 | Streamlit UI | Planned |

## License

MIT
