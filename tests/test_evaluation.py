from __future__ import annotations

import pandas as pd

import pytest

from movie_rag.evaluation import (
    build_annotation_extension,
    dcg_at_k,
    evaluate_results,
    ndcg_at_k,
)


def test_perfect_ndcg_is_one():
    assert dcg_at_k([2, 1, 0], 3) > 0
    assert ndcg_at_k([2, 1, 0], [0, 2, 1], 3) == 1.0


def test_evaluate_results_matches_expected_precision_and_recall():
    results = pd.DataFrame(
        [
            {"query_id": "Q1", "experiment": "D", "system": "dense", "rank": 1, "movieId": 1},
            {"query_id": "Q1", "experiment": "D", "system": "dense", "rank": 2, "movieId": 3},
        ]
    )
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "movieId": 1, "relevance": 2},
            {"query_id": "Q1", "movieId": 2, "relevance": 1},
            {"query_id": "Q1", "movieId": 3, "relevance": 0},
        ]
    )
    metrics = evaluate_results(results, qrels, k=2).iloc[0]
    assert metrics["recall@2"] == 0.5
    assert metrics["precision@2"] == 0.5


def test_expected_query_without_results_is_scored_as_empty_run():
    results = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "experiment": "D",
                "system": "dense",
                "rank": 1,
                "movieId": 1,
            }
        ]
    )
    qrels = pd.DataFrame(
        [
            {"query_id": "Q1", "movieId": 1, "relevance": 2},
            {"query_id": "Q2", "movieId": 2, "relevance": 2},
        ]
    )

    metrics = evaluate_results(
        results,
        qrels,
        k=1,
        expected_query_ids=["Q1", "Q2"],
    ).set_index("query_id")

    assert metrics.loc["Q1", "ndcg@1"] == 1.0
    assert metrics.loc["Q2", "ndcg@1"] == 0.0
    assert metrics.loc["Q2", "recall@1"] == 0.0
    assert metrics.loc["Q2", "precision@1"] == 0.0
    assert metrics.loc["Q2", "retrieved_items_at_k"] == 0


def test_annotation_extension_and_strict_evaluation():
    results = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "query": "memory",
                "experiment": "D",
                "system": "dense+reranker",
                "rank": 1,
                "movieId": 2,
                "canonical_title": "New result",
            }
        ]
    )
    qrels = pd.DataFrame(
        [{"query_id": "Q1", "movieId": 1, "relevance": 2}]
    )

    extension = build_annotation_extension(results, qrels)
    assert extension[["query_id", "movieId"]].values.tolist() == [["Q1", 2]]
    assert pd.isna(extension.iloc[0]["relevance"])

    with pytest.raises(ValueError, match="1 unjudged"):
        evaluate_results(results, qrels, k=1, strict_judgments=True)

    completed = pd.concat(
        [
            qrels,
            pd.DataFrame([{"query_id": "Q1", "movieId": 2, "relevance": 1}]),
        ],
        ignore_index=True,
    )
    assert build_annotation_extension(results, completed).empty
    metrics = evaluate_results(results, completed, k=1, strict_judgments=True)
    assert len(metrics) == 1
