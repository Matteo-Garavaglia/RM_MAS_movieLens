"""Public, versioned data contract exposed by the Movie RAG MCP server."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .mcp_profiles import MCP_CONTRACT_VERSION, RetrievalProfile


class ContractModel(BaseModel):
    """Base class that rejects accidental additions to MCP tool inputs."""

    model_config = ConfigDict(extra="forbid")


class MovieFilterInput(ContractModel):
    """Exact catalog constraints accepted by ``search_movies``.

    Text in ``query`` expresses soft preferences. These fields express hard
    constraints and are applied by Qdrant before ranking.
    """

    min_runtime: float | None = Field(default=None, ge=0)
    max_runtime: float | None = Field(default=None, ge=0)
    min_year: int | None = None
    max_year: int | None = None
    languages: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    include_genres: list[str] = Field(default_factory=list)
    exclude_genres: list[str] = Field(default_factory=list)
    adult: bool | None = None

    @field_validator(
        "languages",
        "countries",
        "include_genres",
        "exclude_genres",
        mode="before",
    )
    @classmethod
    def _clean_string_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("Expected a list of strings")
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = str(item).strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result

    @model_validator(mode="after")
    def _validate_ranges(self) -> "MovieFilterInput":
        if (
            self.min_runtime is not None
            and self.max_runtime is not None
            and self.min_runtime > self.max_runtime
        ):
            raise ValueError("min_runtime cannot exceed max_runtime")
        if self.min_year is not None and self.max_year is not None:
            if self.min_year > self.max_year:
                raise ValueError("min_year cannot exceed max_year")
        return self

    def to_retriever_mapping(self) -> dict[str, Any]:
        """Return only active constraints using the existing retriever names."""

        return self.model_dump(exclude_none=True, exclude_defaults=True)


class SearchExecutionOverrides(ContractModel):
    """Opt-in technical overrides for the experimental MCP tool only."""

    candidate_pool_size: int | None = Field(default=None, ge=1, le=1000)
    exact: bool | None = None
    score_threshold: float | None = None

    @field_validator("score_threshold")
    @classmethod
    def _finite_threshold(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score_threshold must be finite")
        return value


class RetrievalScores(ContractModel):
    """Scores from the active pipeline, kept comparable across profiles."""

    final: float
    dense: float
    reranker: float | None = None

    @field_validator("final", "dense", "reranker")
    @classmethod
    def _finite_scores(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Retrieval scores must be finite")
        return value


class RetrievalPipelineInfo(ContractModel):
    """Read-only provenance; callers can inspect but cannot select it per call."""

    profile_id: str
    pipeline: Literal["dense", "dense_rerank"]
    vector_store: Literal["qdrant"] = "qdrant"
    collection_name: str
    representation_version: str
    encoder_model: str
    reranker_enabled: bool
    reranker_model: str | None = None
    candidate_pool_size: int | None = None
    exact: bool
    score_threshold: float | None = None

    @classmethod
    def from_profile(cls, profile: RetrievalProfile) -> "RetrievalPipelineInfo":
        reranker = profile.reranker_config
        return cls(
            profile_id=profile.profile_id,
            pipeline=profile.pipeline,
            collection_name=profile.dense_config.collection_name,
            representation_version=profile.dense_config.representation_version,
            encoder_model=profile.dense_config.embedding_model,
            reranker_enabled=profile.reranker_enabled,
            reranker_model=None if reranker is None else reranker.model_name,
            candidate_pool_size=(
                None if reranker is None else reranker.candidate_pool_size
            ),
            exact=profile.exact,
            score_threshold=profile.score_threshold,
        )


class MovieSearchResult(ContractModel):
    rank: int = Field(ge=1)
    dense_rank: int = Field(ge=1)
    movie_id: int = Field(ge=0)
    title: str
    overview: str | None = None
    year: int | None = None
    runtime_minutes: int | None = Field(default=None, ge=0)
    genres: list[str] = Field(default_factory=list)
    original_language: str | None = None
    countries: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    main_cast: list[str] = Field(default_factory=list)
    scores: RetrievalScores


class SearchMoviesResponse(ContractModel):
    contract_version: Literal["movie-search/v1"] = MCP_CONTRACT_VERSION
    status: Literal["ok", "no_results"]
    query: str
    applied_filters: MovieFilterInput
    requested_limit: int = Field(ge=1)
    returned_count: int = Field(ge=0)
    pipeline: RetrievalPipelineInfo
    results: list[MovieSearchResult]
    message: str | None = None


class MovieDetails(ContractModel):
    movie_id: int = Field(ge=0)
    imdb_id: int | None = Field(default=None, ge=0)
    tmdb_id: int | None = Field(default=None, ge=0)
    title: str
    original_title: str | None = None
    overview: str | None = None
    tagline: str | None = None
    release_date: str | None = None
    year: int | None = None
    runtime_minutes: int | None = Field(default=None, ge=0)
    genres: list[str] = Field(default_factory=list)
    original_language: str | None = None
    countries: list[str] = Field(default_factory=list)
    spoken_languages: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    composers: list[str] = Field(default_factory=list)
    main_cast: list[str] = Field(default_factory=list)
    main_characters: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    user_tags: list[str] = Field(default_factory=list)
    collection: str | None = None
    production_companies: list[str] = Field(default_factory=list)


class MovieDetailsResponse(ContractModel):
    contract_version: Literal["movie-search/v1"] = MCP_CONTRACT_VERSION
    requested_movie_ids: list[int]
    returned_count: int = Field(ge=0)
    missing_movie_ids: list[int]
    movies: list[MovieDetails]
