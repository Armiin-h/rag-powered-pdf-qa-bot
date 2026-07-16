"""CLI to ask questions against indexed PDF chunks via the RAG chain."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import ask  # noqa: E402


def _print_sources(sources) -> None:
    print("\n--- Sources ---")
    for i, doc in enumerate(sources, start=1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        snippet = doc.page_content[:180].replace("\n", " ")
        print(f"[{i}] {source} (page {page})")
        print(f"    {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question answered from indexed PDF content."
    )
    parser.add_argument("question", type=str, help="Question to ask about the document")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: from settings)",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved source chunks after the answer",
    )
    args = parser.parse_args()

    result = ask(args.question, k=args.top_k)

    print(f"Question: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")

    if args.show_sources:
        _print_sources(result["sources"])


if __name__ == "__main__":
    main()
