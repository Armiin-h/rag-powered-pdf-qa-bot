from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.evaluation.dataset import EvalCase, load_eval_dataset
from src.evaluation.metrics import score_case, summarize_results
from src.rag import ask


def run_evaluation(
    dataset_path: str | Path,
    *,
    k: int | None = None,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    persist_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the RAG pipeline over an eval dataset and return scored results."""
    cases = load_eval_dataset(dataset_path)
    case_results: list[dict] = []

    for case in cases:
        result = ask(
            case.question,
            llm=llm,
            embeddings=embeddings,
            k=k,
            persist_dir=persist_dir,
        )
        case_results.append(
            score_case(case, result["answer"], result["sources"])
        )

    summary = summarize_results(case_results)
    return {
        "dataset": str(dataset_path),
        "summary": summary,
        "results": case_results,
    }
