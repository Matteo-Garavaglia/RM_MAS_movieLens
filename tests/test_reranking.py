from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movie_rag import RERANKER
from movie_rag.reranking import (
    BGEV2M3Reranker,
    RerankedMovieRetriever,
    rerank_candidates,
)


class StaticScorer:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return self.scores.copy()


class FakeFlagReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def compute_score(self, pairs, batch_size, max_length, normalize):
        self.calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "max_length": max_length,
                "normalize": normalize,
            }
        )
        return self.scores


class FakeDenseRetriever:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.candidates.head(kwargs["top_k"]).copy()

    def get_movies(self, movie_ids):
        return pd.DataFrame({"movieId": list(movie_ids)})


@pytest.fixture
def candidates():
    return pd.DataFrame(
        [
            {"rank": 1, "movieId": 10, "score": 0.90, "document": "first"},
            {"rank": 2, "movieId": 20, "score": 0.80, "document": "second"},
            {"rank": 3, "movieId": 30, "score": 0.70, "document": "third"},
        ]
    )


def test_bge_adapter_passes_the_frozen_inference_parameters():
    reranker = BGEV2M3Reranker(use_fp16=False)
    fake_model = FakeFlagReranker([0.25, 0.75])
    reranker._model = fake_model

    scores = reranker.score("memory", ["doc one", "doc two"])

    assert scores.tolist() == [0.25, 0.75]
    assert fake_model.calls == [
        {
            "pairs": [["memory", "doc one"], ["memory", "doc two"]],
            "batch_size": RERANKER.batch_size,
            "max_length": RERANKER.max_length,
            "normalize": RERANKER.normalize_scores,
        }
    ]


def test_reranking_preserves_dense_provenance_and_stable_ties(candidates):
    original = candidates.copy(deep=True)
    scorer = StaticScorer([0.10, 0.90, 0.90])

    result = rerank_candidates("query", candidates, scorer, top_k=3)

    assert result["movieId"].tolist() == [20, 30, 10]
    assert result["rank"].tolist() == [1, 2, 3]
    assert result["dense_rank"].tolist() == [2, 3, 1]
    assert result["dense_score"].tolist() == [0.80, 0.70, 0.90]
    assert result["reranker_score"].tolist() == [0.90, 0.90, 0.10]
    assert result["score"].equals(result["reranker_score"])
    pd.testing.assert_frame_equal(candidates, original)


def test_reranked_retriever_uses_dense_top_n_before_reranking(candidates):
    dense = FakeDenseRetriever(candidates)
    scorer = StaticScorer([0.1, 0.9, 0.2])
    pipeline = RerankedMovieRetriever(dense_retriever=dense, reranker=scorer)

    result = pipeline.search(
        "query",
        filters={"max_runtime": 120},
        candidate_pool_size=3,
        top_k=2,
        exact=True,
    )

    assert result["movieId"].tolist() == [20, 30]
    assert dense.calls == [
        {
            "query": "query",
            "filters": {"max_runtime": 120},
            "top_k": 3,
            "exact": True,
            "score_threshold": None,
        }
    ]


def test_reranked_retriever_rejects_top_k_larger_than_pool(candidates):
    pipeline = RerankedMovieRetriever(
        dense_retriever=FakeDenseRetriever(candidates),
        reranker=StaticScorer([0.1, 0.2, 0.3]),
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        pipeline.search("query", candidate_pool_size=2, top_k=3)


def test_reranking_requires_document_payload(candidates):
    with pytest.raises(KeyError, match="document"):
        rerank_candidates(
            "query",
            candidates.drop(columns="document"),
            StaticScorer([0.1, 0.2, 0.3]),
        )


def test_empty_candidate_frame_does_not_call_the_model():
    scorer = StaticScorer([])
    result = rerank_candidates("query", pd.DataFrame(), scorer)
    assert result.empty
    assert scorer.calls == []
    assert {"dense_rank", "dense_score", "reranker_score", "score"} <= set(
        result.columns
    )
