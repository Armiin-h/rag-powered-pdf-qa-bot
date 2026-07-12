"""CLI helper to load a PDF and preview chunking output."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import load_pdf, split_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a PDF and split it into chunks.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--preview",
        type=int,
        default=2,
        help="Number of chunks to print (default: 2, use 0 to skip)",
    )
    args = parser.parse_args()

    pages = load_pdf(args.pdf_path)
    chunks = split_documents(pages)

    total_chars = sum(len(page.page_content) for page in pages)
    avg_chunk = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)

    print(f"Source:       {args.pdf_path.name}")
    print(f"Pages:        {len(pages)}")
    print(f"Total chars:  {total_chars:,}")
    print(f"Chunks:       {len(chunks)}")
    print(f"Avg chunk:    {avg_chunk:.0f} chars")

    if args.preview > 0:
        print("\n--- Preview ---")
        for i, chunk in enumerate(chunks[: args.preview], start=1):
            page = chunk.metadata.get("page", "?")
            snippet = chunk.page_content[:200].replace("\n", " ")
            print(f"\n[{i}] page {page} ({len(chunk.page_content)} chars)")
            print(f"    {snippet}...")


if __name__ == "__main__":
    main()
