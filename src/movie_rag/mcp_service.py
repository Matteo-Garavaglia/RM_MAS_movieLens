"""Application service behind the MCP transport adapter."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Mapping, Sequence

import pandas as pd

from .documents import clean_text_value, coerce_point_id, split_multi_value
from .embeddings import BGEM3DenseEncoder
from .mcp_contract import (
    MovieDetails,
    MovieDetailsResponse,
    MovieFilterInput,
    MovieSearchResult,
    RetrievalPipelineInfo,
    RetrievalScores,
    SearchExecutionOverrides,
    SearchMoviesResponse,
)
from .mcp_profiles import (
    RetrievalProfile,
    RetrievalProfileCatalog,
    load_profile_catalog,
    selected_profile_id,
)
from .qdrant_store import (
    DenseMovieRetriever,
    connect_qdrant,
    validate_dense_collection,
)
from .reranking import BGEV2M3Reranker, RerankedMovieRetriever


QDRANT_URL_ENV_VAR = "QDRANT_URL"
QDRANT_API_KEY_ENV_VAR = "QDRANT_API_KEY"
QDRANT_TIMEOUT_ENV_VAR = "QDRANT_TIMEOUT_SECONDS"
BGE_USE_FP16_ENV_VAR = "MOVIE_RAG_USE_FP16"


def _optional_text(value: Any) -> str | None:
    text = clean_text_value(value)
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _required_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Result field {field_name!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"Result field {field_name!r} must be finite")
    return result


def _row_value(
    row: Mapping[str, Any] | pd.Series,
    key: str,
    default: Any = None,
) -> Any:
    return row[key] if key in row else default


def _combined_list(row: Mapping[str, Any] | pd.Series, *keys: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in row:
            continue
        for item in split_multi_value(row[key]):
            normalized = item.casefold()
            if normalized not in seen:
                seen.add(normalized)
                result.append(item)
    return result


def _parse_bool_env(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {BGE_USE_FP16_ENV_VAR}: {value!r}")


@dataclass
class MovieSearchService:
    """Stable capability layer independent of MCP transport and model profile."""

    profile: RetrievalProfile
    retriever: Any
    _search_lock: Any = field(default_factory=RLock, init=False, repr=False)

    @property
    def pipeline_info(self) -> RetrievalPipelineInfo:
        return RetrievalPipelineInfo.from_profile(self.profile)

    def search_movies(
        self,
        query: str,
        filters: MovieFilterInput | Mapping[str, Any] | None = None,
        limit: int | None = None,
        execution_overrides: (
            SearchExecutionOverrides | Mapping[str, Any] | None
        ) = None,
    ) -> SearchMoviesResponse:
        cleaned_query = str(query).strip()
        if not cleaned_query:
            raise ValueError("query cannot be empty")
        parsed_filters = (
            filters
            if isinstance(filters, MovieFilterInput)
            else MovieFilterInput.model_validate(filters or {})
        )
        parsed_overrides = (
            execution_overrides
            if isinstance(execution_overrides, SearchExecutionOverrides)
            else SearchExecutionOverrides.model_validate(execution_overrides or {})
        )
        requested_limit = self.profile.default_limit if limit is None else int(limit)
        if requested_limit <= 0:
            raise ValueError("limit must be positive")
        if requested_limit > self.profile.max_limit:
            raise ValueError(
                f"limit cannot exceed {self.profile.max_limit} for the active profile"
            )

        exact = (
            self.profile.exact
            if parsed_overrides.exact is None
            else parsed_overrides.exact
        )
        score_threshold = (
            self.profile.score_threshold
            if parsed_overrides.score_threshold is None
            else parsed_overrides.score_threshold
        )
        search_kwargs = {
            "query": cleaned_query,
            "filters": parsed_filters.to_retriever_mapping(),
            "top_k": requested_limit,
            "exact": exact,
            "score_threshold": score_threshold,
        }
        candidate_pool_size: int | None = None
        if self.profile.reranker_enabled:
            reranker_config = self.profile.reranker_config
            if reranker_config is None:  # defensive; profile validation already rejects this
                raise RuntimeError("The active reranking profile is incomplete")
            candidate_pool_size = (
                reranker_config.candidate_pool_size
                if parsed_overrides.candidate_pool_size is None
                else parsed_overrides.candidate_pool_size
            )
            if candidate_pool_size < requested_limit:
                raise ValueError("candidate_pool_size cannot be smaller than limit")
            search_kwargs["candidate_pool_size"] = candidate_pool_size
        elif parsed_overrides.candidate_pool_size is not None:
            raise ValueError("candidate_pool_size is valid only for a reranking profile")

        pipeline_info = self.pipeline_info.model_copy(
            update={
                "candidate_pool_size": candidate_pool_size,
                "exact": exact,
                "score_threshold": score_threshold,
            }
        )

        # Transformer inference is serialized per service: concurrent calls to the
        # same local model can otherwise duplicate GPU memory or race in its runtime.
        with self._search_lock:
            frame = self.retriever.search(**search_kwargs)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Retriever.search must return a pandas DataFrame")
        if frame.empty:
            return SearchMoviesResponse(
                status="no_results",
                query=cleaned_query,
                applied_filters=parsed_filters,
                requested_limit=requested_limit,
                returned_count=0,
                pipeline=pipeline_info,
                results=[],
                message="No catalog movies satisfy the query and hard filters.",
            )

        results = [
            self._search_result(row, position)
            for position, (_, row) in enumerate(frame.iterrows(), start=1)
        ]
        return SearchMoviesResponse(
            status="ok",
            query=cleaned_query,
            applied_filters=parsed_filters,
            requested_limit=requested_limit,
            returned_count=len(results),
            pipeline=pipeline_info,
            results=results,
        )

    def _search_result(self, row: pd.Series, fallback_rank: int) -> MovieSearchResult:
        final_rank = _optional_int(_row_value(row, "rank")) or fallback_rank
        dense_rank = _optional_int(_row_value(row, "dense_rank")) or final_rank
        final_score = _required_float(_row_value(row, "score"), "score")
        dense_score = _required_float(
            _row_value(row, "dense_score", final_score), "dense_score"
        )
        reranker_value = _row_value(row, "reranker_score")
        reranker_score = (
            None
            if _optional_text(reranker_value) is None
            else _required_float(reranker_value, "reranker_score")
        )
        movie_id = coerce_point_id(
            _row_value(row, self.profile.dense_config.id_column)
        )
        title = _optional_text(
            _row_value(row, self.profile.dense_config.title_column)
        )
        if title is None:
            raise ValueError(f"Movie {movie_id} has no title in its Qdrant payload")
        return MovieSearchResult(
            rank=final_rank,
            dense_rank=dense_rank,
            movie_id=movie_id,
            title=title,
            overview=_optional_text(_row_value(row, "overview")),
            year=_optional_int(_row_value(row, "release_year")),
            runtime_minutes=_optional_int(_row_value(row, "runtime")),
            genres=_combined_list(row, "genres", "ml_genres", "tmdb_genres"),
            original_language=_optional_text(_row_value(row, "original_language")),
            countries=_combined_list(
                row, "countries", "origin_countries", "production_countries"
            ),
            directors=_combined_list(row, "directors"),
            main_cast=_combined_list(row, "cast_top"),
            scores=RetrievalScores(
                final=final_score,
                dense=dense_score,
                reranker=reranker_score,
            ),
        )

    def get_movie_details(self, movie_ids: Sequence[int]) -> MovieDetailsResponse:
        requested = [coerce_point_id(movie_id) for movie_id in movie_ids]
        if not requested:
            raise ValueError("movie_ids cannot be empty")
        if len(requested) > 50:
            raise ValueError("At most 50 movie IDs can be requested at once")
        if len(set(requested)) != len(requested):
            raise ValueError("movie_ids cannot contain duplicates")

        frame = self.retriever.get_movies(requested)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Retriever.get_movies must return a pandas DataFrame")
        id_column = self.profile.dense_config.id_column
        rows_by_id: dict[int, pd.Series] = {}
        if not frame.empty:
            if id_column not in frame:
                raise KeyError(f"Movie details lack ID column {id_column!r}")
            for _, row in frame.iterrows():
                rows_by_id[coerce_point_id(row[id_column])] = row

        movies = [
            self._movie_details(rows_by_id[movie_id])
            for movie_id in requested
            if movie_id in rows_by_id
        ]
        missing = [movie_id for movie_id in requested if movie_id not in rows_by_id]
        return MovieDetailsResponse(
            requested_movie_ids=requested,
            returned_count=len(movies),
            missing_movie_ids=missing,
            movies=movies,
        )

    def _movie_details(self, row: pd.Series) -> MovieDetails:
        movie_id = coerce_point_id(
            _row_value(row, self.profile.dense_config.id_column)
        )
        title = _optional_text(
            _row_value(row, self.profile.dense_config.title_column)
        )
        if title is None:
            raise ValueError(f"Movie {movie_id} has no title in its Qdrant payload")
        return MovieDetails(
            movie_id=movie_id,
            imdb_id=_optional_int(_row_value(row, "imdbId")),
            tmdb_id=_optional_int(_row_value(row, "tmdbId")),
            title=title,
            original_title=_optional_text(_row_value(row, "tmdb_original_title")),
            overview=_optional_text(_row_value(row, "overview")),
            tagline=_optional_text(_row_value(row, "tagline")),
            release_date=_optional_text(_row_value(row, "release_date")),
            year=_optional_int(_row_value(row, "release_year")),
            runtime_minutes=_optional_int(_row_value(row, "runtime")),
            genres=_combined_list(row, "genres", "ml_genres", "tmdb_genres"),
            original_language=_optional_text(_row_value(row, "original_language")),
            countries=_combined_list(
                row, "countries", "origin_countries", "production_countries"
            ),
            spoken_languages=_combined_list(row, "spoken_languages"),
            directors=_combined_list(row, "directors"),
            writers=_combined_list(row, "writers"),
            composers=_combined_list(row, "composers"),
            main_cast=_combined_list(row, "cast_top"),
            main_characters=_combined_list(row, "cast_characters_top"),
            keywords=_combined_list(row, "tmdb_keywords", "tmdb_keyword_text"),
            user_tags=_combined_list(row, "ml_tags"),
            collection=_optional_text(_row_value(row, "collection_name")),
            production_companies=_combined_list(row, "production_companies"),
        )


def _build_dense_retriever(
    profile: RetrievalProfile,
    client: Any,
    use_fp16: bool | None,
) -> DenseMovieRetriever:
    validate_dense_collection(
        client,
        collection_name=profile.dense_config.collection_name,
        config=profile.dense_config,
    )
    encoder = BGEM3DenseEncoder(config=profile.dense_config, use_fp16=use_fp16)
    return DenseMovieRetriever(
        client=client,
        encoder=encoder,
        collection_name=profile.dense_config.collection_name,
        config=profile.dense_config,
    )


def _build_service_for_profile(
    profile: RetrievalProfile,
    dense_retriever: DenseMovieRetriever,
    reranker: BGEV2M3Reranker | None = None,
) -> MovieSearchService:
    if not profile.reranker_enabled:
        return MovieSearchService(profile=profile, retriever=dense_retriever)

    reranker_config = profile.reranker_config
    if reranker_config is None:  # defensive; profile validation already rejects this
        raise RuntimeError("The selected reranking profile has no reranker config")
    if reranker is None:
        raise RuntimeError("No reranker instance was provided for a reranking profile")
    retriever = RerankedMovieRetriever(
        dense_retriever=dense_retriever,
        reranker=reranker,
        config=reranker_config,
    )
    return MovieSearchService(profile=profile, retriever=retriever)


@dataclass
class MovieSearchServiceRegistry:
    """Lazy cache of whitelisted profile services for one MCP process."""

    catalog: RetrievalProfileCatalog
    client: Any
    default_profile_id: str
    use_fp16: bool | None = None
    _services: dict[str, MovieSearchService] = field(default_factory=dict, init=False)
    _dense_retrievers: dict[Any, DenseMovieRetriever] = field(
        default_factory=dict, init=False, repr=False
    )
    _rerankers: dict[Any, BGEV2M3Reranker] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: Any = field(default_factory=RLock, init=False, repr=False)

    @classmethod
    def from_environment(
        cls,
        profile_id: str | None = None,
        profiles_path: str | None = None,
    ) -> "MovieSearchServiceRegistry":
        catalog = load_profile_catalog(profiles_path)
        default_profile_id = (
            selected_profile_id(profile_id) or catalog.default_profile_id
        )
        catalog.get(default_profile_id)  # fail before opening a Qdrant connection
        url = os.getenv(QDRANT_URL_ENV_VAR, "http://localhost:6333")
        api_key = os.getenv(QDRANT_API_KEY_ENV_VAR) or None
        try:
            timeout = float(os.getenv(QDRANT_TIMEOUT_ENV_VAR, "30"))
        except ValueError as exc:
            raise ValueError(f"{QDRANT_TIMEOUT_ENV_VAR} must be numeric") from exc
        if timeout <= 0:
            raise ValueError(f"{QDRANT_TIMEOUT_ENV_VAR} must be positive")
        client = connect_qdrant(url=url, api_key=api_key, timeout_seconds=timeout)
        return cls(
            catalog=catalog,
            client=client,
            default_profile_id=default_profile_id,
            use_fp16=_parse_bool_env(os.getenv(BGE_USE_FP16_ENV_VAR)),
        )

    @property
    def available_profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.catalog.profiles))

    def get(self, profile_id: str | None = None) -> MovieSearchService:
        selected = profile_id or self.default_profile_id
        profile = self.catalog.get(selected)
        with self._lock:
            if selected in self._services:
                return self._services[selected]

            dense_key = profile.dense_config
            if dense_key not in self._dense_retrievers:
                self._dense_retrievers[dense_key] = _build_dense_retriever(
                    profile=profile,
                    client=self.client,
                    use_fp16=self.use_fp16,
                )
            dense_retriever = self._dense_retrievers[dense_key]

            reranker: BGEV2M3Reranker | None = None
            if profile.reranker_enabled:
                reranker_config = profile.reranker_config
                if reranker_config is None:
                    raise RuntimeError("The selected profile has no reranker config")
                if reranker_config not in self._rerankers:
                    self._rerankers[reranker_config] = BGEV2M3Reranker(
                        config=reranker_config,
                        use_fp16=self.use_fp16,
                    )
                reranker = self._rerankers[reranker_config]

            self._services[selected] = _build_service_for_profile(
                profile=profile,
                dense_retriever=dense_retriever,
                reranker=reranker,
            )
            return self._services[selected]


def build_movie_search_service(
    profile_id: str | None = None,
    profiles_path: str | None = None,
) -> MovieSearchService:
    """Build the active service once from server-side configuration.

    This function connects to an existing collection. It never creates,
    recreates or writes Qdrant data.
    """

    return MovieSearchServiceRegistry.from_environment(
        profile_id=profile_id,
        profiles_path=profiles_path,
    ).get()
