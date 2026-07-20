from dataclasses import dataclass
from pathlib import Path
import json

from src.config import PROJECT_ROOT


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_keywords: list[str]
    expected_pages: list[int] | None = None


def load_eval_dataset(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {dataset_path}")

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Eval dataset must be a non-empty JSON array")

    cases: list[EvalCase] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(f"Invalid eval case at index {i}")

        keywords = item.get("expected_keywords", [])
        pages = item.get("expected_pages")
        cases.append(
            EvalCase(
                question=str(item["question"]),
                expected_keywords=[str(k) for k in keywords],
                expected_pages=[int(p) for p in pages] if pages else None,
            )
        )
    return cases


def default_dataset_path() -> Path:
    return PROJECT_ROOT / "data" / "eval" / "attention_is_all_you_need.json"
