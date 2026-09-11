"""Equivalence checks between the notebook's NumPy ranking and Qdrant."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import BASELINE, DenseBaselineConfig
from .documents import coerce_point_id
from .embeddings import l2_normalize_vector
from .filters import MovieFilters, dataframe_filter_mask


def numpy_dense_search(
    prepared: pd.DataFrame,
    vectors: np.ndarray,
    query_vector: Sequence[float] | np.ndarray,
    filters: Mapping[str, Any] | MovieFilters | None = None,
    top_k: int = 10,
    config: DenseBaselineConfig = BASELINE,
) -> pd.DataFrame:
    matrix = np.asarray(vectors, dtype=np.float32)
    query = l2_normalize_vector(np.asarray(query_vector, dtype=np.float32))
    if matrix.ndim != 2 or matrix.shape[0] != len(prepared):
        raise ValueError("Catalog and embedding matrix shapes are inconsistent")
    if matrix.shape[1] != len(query):
        raise ValueError("Document and query dimensions differ")

    scores = matrix @ query
    eligible = np.flatnonzero(dataframe_filter_mask(prepared, filters).to_numpy())
    ordered = eligible[np.argsort(-scores[eligible], kind="stable")][:top_k]
    columns = [
        column
        for column in (
            config.id_column,
            config.title_column,
            "release_year",
            "runtime",
            "ml_genres",
            "tmdb_genres",
            "original_language",
        )
        if column in prepared.columns
    ]
    result = prepared.iloc[ordered][columns].copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["score"] = scores[ordered]
    return result


def compare_query_rankings(
    prepared: pd.DataFrame,
    vectors: np.ndarray,
    retriever: Any,
    query: str,
    filters: Mapping[str, Any] | MovieFilters | None = None,
    top_k: int = 10,
    query_id: str | None = None,
    config: DenseBaselineConfig = BASELINE,
) -> dict[str, Any]:
    query_vector = retriever.encoder.encode_query(query)
    local = numpy_dense_search(
        prepared,
        vectors,
        query_vector,
        filters=filters,
        top_k=top_k,
        config=config,
    )
    remote = retriever.search_by_vector(
        query_vector,
        filters=filters,
        top_k=top_k,
        exact=True,
    )
    local_ids = [coerce_point_id(value) for value in local.get(config.id_column, [])]
    remote_ids = [coerce_point_id(value) for value in remote.get(config.id_column, [])]
    denominator = min(top_k, max(len(local_ids), len(remote_ids)))
    overlap = 1.0 if denominator == 0 else len(set(local_ids) & set(remote_ids)) / denominator

    local_scores = dict(zip(local_ids, local.get("score", [])))
    remote_scores = dict(zip(remote_ids, remote.get("score", [])))
    common_ids = set(local_scores) & set(remote_scores)
    max_delta = (
        max(abs(float(local_scores[item]) - float(remote_scores[item])) for item in common_ids)
        if common_ids
        else np.nan
    )
    return {
        "query_id": query_id,
        "query": query,
        "filters": dict(filters or {}) if not isinstance(filters, MovieFilters) else str(filters),
        "local_results": len(local_ids),
        "qdrant_results": len(remote_ids),
        "same_order": local_ids == remote_ids,
        f"overlap@{top_k}": overlap,
        "max_score_delta": max_delta,
        "local_ids": local_ids,
        "qdrant_ids": remote_ids,
    }


def validate_query_set(
    prepared: pd.DataFrame,
    vectors: np.ndarray,
    retriever: Any,
    queries: pd.DataFrame | Sequence[Mapping[str, Any]],
    top_k: int = 10,
    config: DenseBaselineConfig = BASELINE,
) -> pd.DataFrame:
    rows = queries.to_dict("records") if isinstance(queries, pd.DataFrame) else list(queries)
    comparisons = [
        compare_query_rankings(
            prepared=prepared,
            vectors=vectors,
            retriever=retriever,
            query=str(row["query"]),
            filters=row.get("filters", {}),
            top_k=top_k,
            query_id=row.get("query_id"),
            config=config,
        )
        for row in rows
    ]
    return pd.DataFrame(comparisons)

