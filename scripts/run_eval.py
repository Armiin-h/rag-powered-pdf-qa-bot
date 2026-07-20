"""Run RAG evaluation against a labeled question set."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import default_dataset_path, run_evaluation  # noqa: E402


def _safe_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_report(report: dict) -> None:
    summary = report["summary"]
    print(f"Dataset:             {report['dataset']}")
    print(f"Cases:               {summary['cases']}")
    print(f"Avg keyword recall:  {summary['avg_keyword_recall']:.3f}")
    print(f"Page hit rate:       {summary['page_hit_rate']:.3f}")
    print()
    for i, result in enumerate(report["results"], start=1):
        page_hit = result["page_hit"]
        page_label = "n/a" if page_hit is None else ("yes" if page_hit else "no")
        print(f"[{i}] {result['question']}")
        print(f"    keyword recall: {result['keyword_recall']:.2f} | page hit: {page_label}")
        print(f"    pages retrieved: {result['retrieved_pages']}")
        print(f"    answer: {_safe_text(result['answer_preview'])}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers against a labeled question set."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path(),
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write full JSON report",
    )
    args = parser.parse_args()

    report = run_evaluation(args.dataset, k=args.top_k)
    _print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
