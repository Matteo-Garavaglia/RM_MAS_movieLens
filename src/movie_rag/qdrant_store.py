"""Qdrant collection lifecycle, indexing and dense movie retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import BASELINE, DenseBaselineConfig
from .documents import catalog_fingerprint, coerce_point_id, row_to_payload
from .filters import MovieFilters, build_qdrant_filter


@dataclass(frozen=True)
class CollectionResult:
    collection_name: str
    created: bool
    vector_name: str
    vector_size: int


@dataclass(frozen=True)
class IndexResult:
    collection_name: str
    indexed_points: int
    collection_points: int
    dataset_points: int
    dataset_fingerprint: str


def connect_qdrant(
    url: str = "http://localhost:6333",
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError("qdrant-client is required to connect to Qdrant") from exc
    return QdrantClient(url=url, api_key=api_key or None, timeout=timeout_seconds)


def _models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise ImportError("qdrant-client is required for Qdrant operations") from exc
    return models


def _validate_existing_collection(
    client: Any,
    collection_name: str,
    config: DenseBaselineConfig,
) -> None:
    models = _models()
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, Mapping):
        raise ValueError(
            f"Collection {collection_name!r} does not use named vectors; expected {config.vector_name!r}"
        )
    params = vectors.get(config.vector_name)
    if params is None:
        raise ValueError(
            f"Collection {collection_name!r} lacks named vector {config.vector_name!r}"
        )
    if int(params.size) != config.embedding_dimension:
        raise ValueError(
            f"Collection vector size is {params.size}; expected {config.embedding_dimension}"
        )
    if params.distance != models.Distance.COSINE:
        raise ValueError(f"Collection distance is {params.distance}; expected Cosine")


def validate_dense_collection(
    client: Any,
    collection_name: str = BASELINE.collection_name,
    config: DenseBaselineConfig = BASELINE,
) -> None:
    """Validate an existing collection without creating, deleting or updating it."""

    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Qdrant collection {collection_name!r} does not exist. "
            "Run the indexing notebook before starting the retrieval service."
        )
    _validate_existing_collection(client, collection_name, config)

    models = _models()
    expected_provenance = models.Filter(
        must=[
            models.FieldCondition(
                key="embedding_model",
                match=models.MatchValue(value=config.embedding_model),
            ),
            models.FieldCondition(
                key="representation_version",
                match=models.MatchValue(value=config.representation_version),
            ),
            models.FieldCondition(
                key="payload_schema_version",
                match=models.MatchValue(value=config.payload_schema_version),
            ),
        ]
    )
    collection_points = int(
        client.count(collection_name=collection_name, exact=True).count
    )
    if collection_points == 0:
        raise RuntimeError(f"Qdrant collection {collection_name!r} is empty")
    matching_points = int(
        client.count(
            collection_name=collection_name,
            count_filter=expected_provenance,
            exact=True,
        ).count
    )
    if matching_points != collection_points:
        raise ValueError(
            f"Collection {collection_name!r} contains {collection_points} points, but "
            f"only {matching_points} match embedding model {config.embedding_model!r}, "
            f"representation {config.representation_version!r} and payload schema "
            f"{config.payload_schema_version!r}"
        )


def ensure_dense_collection(
    client: Any,
    collection_name: str = BASELINE.collection_name,
    config: DenseBaselineConfig = BASELINE,
    recreate: bool = False,
) -> CollectionResult:
    """Create the named-vector collection or validate an existing one.

    Setting ``recreate=True`` deletes the existing collection first. It is false by
    default so a notebook rerun cannot silently destroy persisted data.
    """

    models = _models()
    exists = bool(client.collection_exists(collection_name))
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                config.vector_name: models.VectorParams(
                    size=config.embedding_dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )
        created = True
    else:
        _validate_existing_collection(client, collection_name, config)
        created = False

    return CollectionResult(
        collection_name=collection_name,
        created=created,
        vector_name=config.vector_name,
        vector_size=config.embedding_dimension,
    )


PAYLOAD_INDEXES: dict[str, str] = {
    "release_year": "integer",
    "runtime": "integer",
    "original_language_norm": "keyword",
    "countries_norm": "keyword",
    "genres_norm": "keyword",
    "adult": "bool",
    "representation_version": "keyword",
    "dataset_fingerprint": "keyword",
}


def ensure_payload_indexes(
    client: Any,
    collection_name: str = BASELINE.collection_name,
) -> list[str]:
    """Create indexes only for fields currently used in filters or provenance checks."""

    models = _models()
    schemas = {
        "integer": models.PayloadSchemaType.INTEGER,
        "keyword": models.PayloadSchemaType.KEYWORD,
        "bool": models.PayloadSchemaType.BOOL,
    }
    created_or_confirmed: list[str] = []
    for field_name, schema_name in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schemas[schema_name],
            wait=True,
        )
        created_or_confirmed.append(field_name)
    return created_or_confirmed


def index_catalog(
    client: Any,
    prepared: pd.DataFrame,
    vectors: np.ndarray,
    collection_name: str = BASELINE.collection_name,
    config: DenseBaselineConfig = BASELINE,
    batch_size: int = 64,
    allow_extra_points: bool = False,
) -> IndexResult:
    """Idempotently upsert one dense vector and one payload per movie ID."""

    models = _models()
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a vector matrix, got shape {matrix.shape}")
    if len(prepared) != matrix.shape[0]:
        raise ValueError("Catalog rows and vector rows differ")
    if matrix.shape[1] != config.embedding_dimension:
        raise ValueError(
            f"Expected {config.embedding_dimension} vector dimensions, got {matrix.shape[1]}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    fingerprint = catalog_fingerprint(prepared, config)
    points: list[Any] = []
    for position, (_, row) in enumerate(prepared.iterrows()):
        point_id = coerce_point_id(row[config.id_column])
        payload = row_to_payload(
            row,
            document=str(row[config.document_column]),
            dataset_fingerprint=fingerprint,
            config=config,
        )
        points.append(
            models.PointStruct(
                id=point_id,
                vector={config.vector_name: matrix[position].tolist()},
                payload=payload,
            )
        )

    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[start : start + batch_size],
            wait=True,
        )

    fingerprint_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="dataset_fingerprint",
                match=models.MatchValue(value=fingerprint),
            )
        ]
    )
    dataset_points = int(
        client.count(
            collection_name=collection_name,
            count_filter=fingerprint_filter,
            exact=True,
        ).count
    )
    collection_points = int(client.count(collection_name=collection_name, exact=True).count)
    if dataset_points != len(prepared):
        raise RuntimeError(
            f"Qdrant contains {dataset_points} points for this dataset; expected {len(prepared)}"
        )
    if not allow_extra_points and collection_points != dataset_points:
        raise RuntimeError(
            f"Collection contains {collection_points} points but only {dataset_points} belong "
            "to the current catalog fingerprint. Use a new collection name or explicitly "
            "recreate the collection."
        )

    return IndexResult(
        collection_name=collection_name,
        indexed_points=len(points),
        collection_points=collection_points,
        dataset_points=dataset_points,
        dataset_fingerprint=fingerprint,
    )


def _hits_to_frame(hits: Sequence[Any], config: DenseBaselineConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        payload = dict(hit.payload or {})
        record = {
            "rank": rank,
            config.id_column: payload.get(config.id_column, hit.id),
            "score": float(hit.score),
            **payload,
        }
        records.append(record)
    return pd.DataFrame(records)


@dataclass
class DenseMovieRetriever:
    """Thin application service that can later be exposed through MCP."""

    client: Any
    encoder: Any
    collection_name: str = BASELINE.collection_name
    config: DenseBaselineConfig = BASELINE

    def search_by_vector(
        self,
        query_vector: Sequence[float] | np.ndarray,
        filters: Mapping[str, Any] | MovieFilters | None = None,
        top_k: int = 10,
        exact: bool = False,
        score_threshold: float | None = None,
    ) -> pd.DataFrame:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if len(vector) != self.config.embedding_dimension:
            raise ValueError(
                f"Query vector has {len(vector)} dimensions; expected {self.config.embedding_dimension}"
            )
        models = _models()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            using=self.config.vector_name,
            query_filter=build_qdrant_filter(filters),
            search_params=models.SearchParams(exact=exact),
            with_payload=True,
            with_vectors=False,
            limit=top_k,
            score_threshold=score_threshold,
        )
        return _hits_to_frame(response.points, self.config)

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | MovieFilters | None = None,
        top_k: int = 10,
        exact: bool = False,
        score_threshold: float | None = None,
    ) -> pd.DataFrame:
        query_vector = self.encoder.encode_query(query)
        return self.search_by_vector(
            query_vector=query_vector,
            filters=filters,
            top_k=top_k,
            exact=exact,
            score_threshold=score_threshold,
        )

    def get_movies(self, movie_ids: Sequence[int]) -> pd.DataFrame:
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[coerce_point_id(value) for value in movie_ids],
            with_payload=True,
            with_vectors=False,
        )
        records = []
        for point in points:
            payload = dict(point.payload or {})
            records.append(
                {
                    self.config.id_column: payload.get(self.config.id_column, point.id),
                    **payload,
                }
            )
        return pd.DataFrame(records)
