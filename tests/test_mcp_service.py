from __future__ import annotations

import pandas as pd
import pytest

from movie_rag.mcp_contract import MovieFilterInput, SearchExecutionOverrides
from movie_rag.mcp_profiles import load_profile_catalog
from movie_rag.mcp_service import MovieSearchService


def search_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "movieId": 1,
                "canonical_title": "Memory Film (1999)",
                "score": 0.81,
                "overview": "A reflective story about memory and identity.",
                "release_year": 1999,
                "runtime": 110,
                "genres": ["Drama", "Science Fiction"],
                "original_language": "en",
                "countries": ["US"],
                "directors": "Jane Doe",
                "cast_top": "Actor One | Actor Two",
            }
        ]
    )


class FakeRetriever:
    def __init__(self, frame: pd.DataFrame | None = None):
        self.frame = search_frame() if frame is None else frame
        self.search_calls: list[dict] = []
        self.detail_calls: list[list[int]] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.frame.copy()

    def get_movies(self, movie_ids):
        self.detail_calls.append(list(movie_ids))
        data = search_frame().drop(columns=["rank", "score"])
        data["imdbId"] = 101
        data["tmdbId"] = 201
        data["writers"] = "Writer One"
        return data[data["movieId"].isin(movie_ids)].copy()


def profile(profile_id: str):
    return load_profile_catalog().get(profile_id)


def test_dense_service_hides_execution_defaults_and_returns_stable_shape():
    retriever = FakeRetriever()
    service = MovieSearchService(profile("dense_v1"), retriever)

    response = service.search_movies(
        "memory and identity",
        filters=MovieFilterInput(max_runtime=120, exclude_genres=["Horror"]),
        limit=10,
    )

    assert response.status == "ok"
    assert response.pipeline.profile_id == "dense_v1"
    assert response.pipeline.reranker_enabled is False
    assert response.results[0].scores.final == 0.81
    assert response.results[0].scores.dense == 0.81
    assert response.results[0].scores.reranker is None
    assert retriever.search_calls == [
        {
            "query": "memory and identity",
            "filters": {"max_runtime": 120.0, "exclude_genres": ["Horror"]},
            "top_k": 10,
            "exact": True,
            "score_threshold": None,
        }
    ]


def test_reranking_profile_and_per_call_overrides_are_atomic():
    reranked_frame = search_frame().assign(
        dense_rank=7,
        dense_score=0.44,
        reranker_score=0.92,
        score=0.92,
    )
    retriever = FakeRetriever(reranked_frame)
    service = MovieSearchService(profile("dense_rerank_v1"), retriever)

    response = service.search_movies(
        "memory",
        limit=10,
        execution_overrides=SearchExecutionOverrides(
            candidate_pool_size=30,
            exact=False,
            score_threshold=0.2,
        ),
    )

    assert response.pipeline.profile_id == "dense_rerank_v1"
    assert response.pipeline.candidate_pool_size == 30
    assert response.pipeline.exact is False
    assert response.pipeline.score_threshold == 0.2
    assert response.results[0].dense_rank == 7
    assert response.results[0].scores.model_dump() == {
        "final": 0.92,
        "dense": 0.44,
        "reranker": 0.92,
    }
    assert retriever.search_calls[0]["candidate_pool_size"] == 30

    service.search_movies("second call", limit=10)
    assert retriever.search_calls[1]["candidate_pool_size"] == 20
    assert retriever.search_calls[1]["exact"] is True
    assert retriever.search_calls[1]["score_threshold"] is None


def test_candidate_pool_override_is_rejected_for_dense_profile():
    service = MovieSearchService(profile("dense_v1"), FakeRetriever())

    with pytest.raises(ValueError, match="only for a reranking profile"):
        service.search_movies(
            "memory",
            execution_overrides={"candidate_pool_size": 30},
        )


def test_empty_retrieval_is_a_successful_no_results_response():
    service = MovieSearchService(profile("dense_rerank_v1"), FakeRetriever(pd.DataFrame()))

    response = service.search_movies(
        "film italiano",
        filters={"languages": ["it"]},
    )

    assert response.status == "no_results"
    assert response.returned_count == 0
    assert response.results == []
    assert response.message


def test_details_preserve_request_order_and_report_missing_ids():
    retriever = FakeRetriever()
    service = MovieSearchService(profile("dense_v1"), retriever)

    response = service.get_movie_details([2, 1])

    assert response.requested_movie_ids == [2, 1]
    assert response.missing_movie_ids == [2]
    assert [movie.movie_id for movie in response.movies] == [1]
    assert response.movies[0].writers == ["Writer One"]


def test_filters_reject_inverted_ranges_and_unknown_fields():
    with pytest.raises(ValueError, match="min_year"):
        MovieFilterInput(min_year=2020, max_year=2000)
    with pytest.raises(ValueError, match="extra_forbidden"):
        MovieFilterInput.model_validate({"director": "Jane Doe"})

