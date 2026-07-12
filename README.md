# PDF Document Q&A Bot (RAG)

A retrieval-augmented generation (RAG) application that lets you upload large PDF documents and ask questions answered strictly from the document's content.

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
cp .env.example .env            # adjust settings if needed
```

## Usage

```bash
streamlit run app.py
```

Upload a PDF, wait for indexing to complete, then ask questions about the document.

## Project Status

🚧 **In development** — initial scaffolding. See commit history for progress.

## License

MIT
