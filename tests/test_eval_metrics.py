import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.evaluation.dataset import EvalCase, load_eval_dataset
from src.evaluation.metrics import keyword_recall, page_hit, score_case, summarize_results


def test_load_eval_dataset(tmp_path: Path):
    data = [
        {
            "question": "What is attention?",
            "expected_keywords": ["attention"],
            "expected_pages": [2],
        }
    ]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    cases = load_eval_dataset(path)

    assert len(cases) == 1
    assert cases[0].question == "What is attention?"
    assert cases[0].expected_keywords == ["attention"]
    assert cases[0].expected_pages == [2]


def test_load_eval_dataset_rejects_empty(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        load_eval_dataset(path)


def test_keyword_recall_partial_match():
    score = keyword_recall("Self-attention relates positions.", ["attention", "transformer"])
    assert score == 0.5


def test_page_hit_detects_expected_page():
    sources = [
        Document(page_content="text", metadata={"page": 3}),
        Document(page_content="text", metadata={"page": 8}),
    ]

    assert page_hit(sources, [3, 6]) is True
    assert page_hit(sources, [1]) is False
    assert page_hit(sources, None) is None


def test_score_case_and_summary():
    case = EvalCase(
        question="How many layers?",
        expected_keywords=["6", "layer"],
        expected_pages=[3],
    )
    sources = [Document(page_content="N = 6 layers", metadata={"page": 3})]
    result = score_case(case, "The model has 6 layers.", sources)

    assert result["keyword_recall"] == 1.0
    assert result["page_hit"] is True

    summary = summarize_results([result])
    assert summary["cases"] == 1
    assert summary["avg_keyword_recall"] == 1.0
    assert summary["page_hit_rate"] == 1.0
