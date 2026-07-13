"""CLI helper to load a PDF, preview chunks, and optionally index or search."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing import index_pdf, search_index  # noqa: E402
from src.ingestion import load_pdf, split_documents  # noqa: E402


def _print_chunk_preview(chunks, preview: int) -> None:
    print("\n--- Preview ---")
    for i, chunk in enumerate(chunks[:preview], start=1):
        page = chunk.metadata.get("page", "?")
        snippet = chunk.page_content[:200].replace("\n", " ")
        print(f"\n[{i}] page {page} ({len(chunk.page_content)} chars)")
        print(f"    {snippet}...")


def _print_search_results(query: str, results) -> None:
    print(f"\nQuery: {query}")
    print(f"Matches: {len(results)}")
    for i, doc in enumerate(results, start=1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"\n[{i}] {source} (page {page})")
        print(f"    {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a PDF, split it into chunks, and optionally index or search."
    )
    parser.add_argument("pdf_path", nargs="?", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--preview",
        type=int,
        default=2,
        help="Number of chunks to print (default: 2, use 0 to skip)",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Embed chunks and persist them in ChromaDB",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search indexed chunks for content similar to this query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of search results to return (default: from settings)",
    )
    args = parser.parse_args()

    if args.query and not args.pdf_path:
        results = search_index(args.query, k=args.top_k)
        _print_search_results(args.query, results)
        return

    if not args.pdf_path:
        parser.error("pdf_path is required unless using --query without a new PDF")

    pages = load_pdf(args.pdf_path)
    chunks = split_documents(pages)

    total_chars = sum(len(page.page_content) for page in pages)
    avg_chunk = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)

    print(f"Source:       {args.pdf_path.name}")
    print(f"Pages:        {len(pages)}")
    print(f"Total chars:  {total_chars:,}")
    print(f"Chunks:       {len(chunks)}")
    print(f"Avg chunk:    {avg_chunk:.0f} chars")

    if args.preview > 0 and not args.index:
        _print_chunk_preview(chunks, args.preview)

    if args.index:
        stats = index_pdf(args.pdf_path)
        print("\n--- Indexed ---")
        print(f"Collection:   {stats['collection']}")
        print(f"Persist dir:  {stats['persist_dir']}")
        print(f"Chunks added: {stats['chunks_indexed']}")

    if args.query:
        results = search_index(args.query, k=args.top_k)
        _print_search_results(args.query, results)


if __name__ == "__main__":
    main()
