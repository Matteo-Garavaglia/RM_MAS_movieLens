from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from movie_rag.resource_adapters import (
    MOVIELENS_V2_DOCUMENT_COLUMN,
    MovieLensV2Adapter,
    movie_filters_to_resource_filters,
)
from movie_rag.resource_models import ResourceDocument, ResourceFilters, ResourceSearchHit
from movie_rag.resource_qdrant import qdrant_point_id
from movie_rag.resource_service import RetrievalService, ScoreRerankerAdapter


def _movie_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "movieId": 1,
            "tmdb_title": "Clean Title",
            "canonical_title": "Clean Title (2000)",
            "tmdb_genres": "Drama, Science Fiction",
            "overview": "A story about memory.",
            "tagline": "Remember.",
            "tmdb_keyword_text": "memory, identity",
            "ml_tags": "thought-provoking",
            "directors": "Director A",
            "writers": "Writer A",
            "composers": "Composer A",
            "cast_top": "Actor A",
            "cast_characters_top": "Character A",
            "collection_name": "",
            "production_companies": "Studio A",
            "release_year": 2000,
            "runtime": 110,
            "original_language": "en",
            "origin_countries": "US",
            "production_countries": "US, GB",
            "adult": False,
        }
    ])


def test_movielens_v2_document_uses_tmdb_title():
    adapter = MovieLensV2Adapter()
    prepared = adapter.prepare(_movie_frame())
    document = prepared.iloc[0][MOVIELENS_V2_DOCUMENT_COLUMN]
    assert "Title: Clean Title" in document
    assert "Clean Title (2000)" not in document


def test_movie_adapter_is_domain_agnostic_at_output_boundary():
    resource = MovieLensV2Adapter().to_resources(_movie_frame())[0]
    assert resource.resource_id == 1
    assert resource.title == "Clean Title"
    assert resource.domain == "movielens_movies"
    assert "Science Fiction" in resource.metadata["genres"]
    assert resource.metadata["runtime"] == 110


def test_movie_filter_translation():
    filters = movie_filters_to_resource_filters({
        "max_runtime": 120,
        "min_year": 1990,
        "include_genres": ["Science Fiction"],
        "exclude_genres": ["Horror"],
        "languages": ["en"],
    })
    assert filters.ranges["runtime"].lte == 120
    assert filters.ranges["year"].gte == 1990
    assert filters.any_of["genres"] == ["Science Fiction"]
    assert filters.none_of["genres"] == ["Horror"]
    assert filters.any_of["language"] == ["en"]


def test_arbitrary_string_ids_get_stable_uuid():
    first = qdrant_point_id("whoteach:chunk:abc")
    second = qdrant_point_id("whoteach:chunk:abc")
    assert first == second
    assert isinstance(first, str)
    assert qdrant_point_id(42) == 42


class FakeRetriever:
    name = "fake_dense"

    def __init__(self):
        self.last_top_k = None

    def search(self, query, *, filters=None, top_k=10, exact=False):
        self.last_top_k = top_k
        return [
            ResourceSearchHit(
                rank=i,
                resource_id=i,
                title=f"R{i}",
                text="x" * i,
                metadata={},
                score=float(100 - i),
                retrieval_score=float(100 - i),
                first_stage_rank=i,
            )
            for i in range(1, top_k + 1)
        ]

    def get(self, resource_ids):
        return [
            ResourceDocument(resource_id=i, title=f"R{i}", text="x", metadata={})
            for i in resource_ids
        ]


class FakeScorer:
    model_name = "fake_reranker"

    def score(self, query, documents):
        return np.asarray([len(doc) for doc in documents], dtype=float)


def test_fast_profile_skips_reranking():
    retriever = FakeRetriever()
    service = RetrievalService(retriever)
    result = service.search("q", profile_id="fast")
    assert retriever.last_top_k == 10
    assert result.pipeline.rerank_enabled is False
    assert result.returned_count == 10


def test_quality_profile_uses_top20_then_top10():
    retriever = FakeRetriever()
    reranker = ScoreRerankerAdapter(FakeScorer())
    service = RetrievalService(retriever, reranker=reranker)
    result = service.search("q", profile_id="quality")
    assert retriever.last_top_k == 20
    assert result.returned_count == 10
    assert result.results[0].resource_id == 20
    assert result.pipeline.rerank_enabled is True
