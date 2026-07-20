from langchain_core.documents import Document

from src.evaluation.dataset import EvalCase


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def page_hit(sources: list[Document], expected_pages: list[int] | None) -> bool | None:
    """True if any retrieved chunk comes from an expected page."""
    if not expected_pages:
        return None

    retrieved_pages = {
        int(source.metadata.get("page"))
        for source in sources
        if source.metadata.get("page") is not None
    }
    return bool(retrieved_pages.intersection(set(expected_pages)))


def score_case(case: EvalCase, answer: str, sources: list[Document]) -> dict:
    """Score a single evaluation case."""
    kw_score = keyword_recall(answer, case.expected_keywords)
    page_match = page_hit(sources, case.expected_pages)

    return {
        "question": case.question,
        "keyword_recall": kw_score,
        "page_hit": page_match,
        "answer_preview": answer[:200],
        "retrieved_pages": sorted(
            {
                int(doc.metadata.get("page"))
                for doc in sources
                if doc.metadata.get("page") is not None
            }
        ),
    }


def summarize_results(case_results: list[dict]) -> dict:
    """Aggregate per-case scores into summary metrics."""
    if not case_results:
        return {
            "cases": 0,
            "avg_keyword_recall": 0.0,
            "page_hit_rate": 0.0,
        }

    avg_keyword = sum(r["keyword_recall"] for r in case_results) / len(case_results)

    page_scored = [r for r in case_results if r["page_hit"] is not None]
    page_hit_rate = (
        sum(1 for r in page_scored if r["page_hit"]) / len(page_scored)
        if page_scored
        else 0.0
    )

    return {
        "cases": len(case_results),
        "avg_keyword_recall": round(avg_keyword, 3),
        "page_hit_rate": round(page_hit_rate, 3),
    }
