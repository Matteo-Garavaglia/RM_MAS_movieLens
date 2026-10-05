from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .resource_embeddings import SentenceTransformerResourceEncoder
from .resource_models import ResourceDocument, ResourceFilters, ResourceSearchHit

RESOURCE_NAMESPACE = uuid.UUID("26a5b500-2b37-4b04-b8ea-6b39cb21b231")


def qdrant_point_id(resource_id: str | int) -> int | str:
    """Map arbitrary domain IDs to Qdrant-compatible uint64 / UUID IDs."""

    if isinstance(resource_id, (int, np.integer)) and int(resource_id) >= 0:
        return int(resource_id)
    text = str(resource_id)
    if text.isdigit():
        value = int(text)
        if value >= 0:
            return value
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.uuid5(RESOURCE_NAMESPACE, text))


@dataclass(frozen=True)
class QdrantCollectionConfig:
    collection_name: str
    dimension: int
    vector_name: str = "dense"
    distance: str = "Cosine"
    schema_version: str = "resource_payload_v1"


def connect_qdrant(url: str = "http://localhost:6333", api_key: str | None = None, timeout: float = 30.0):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("qdrant-client is required") from exc
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def ensure_collection(client: Any, config: QdrantCollectionConfig, recreate: bool = False) -> None:
    from qdrant_client import models

    existing = {c.name for c in client.get_collections().collections}
    if config.collection_name in existing and recreate:
        client.delete_collection(config.collection_name)
        existing.remove(config.collection_name)
    if config.collection_name not in existing:
        distance = getattr(models.Distance, config.distance.upper())
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config={
                config.vector_name: models.VectorParams(
                    size=config.dimension,
                    distance=distance,
                )
            },
        )


def resource_payload(resource: ResourceDocument, config: QdrantCollectionConfig, embedding_model: str) -> dict[str, Any]:
    return {
        "resource_id": resource.resource_id,
        "title": resource.title,
        "text": resource.text,
        "metadata": resource.metadata,
        "domain": resource.domain,
        "source": resource.source,
        "schema_version": config.schema_version,
        "embedding_model": embedding_model,
    }


def upsert_resources(
    client: Any,
    config: QdrantCollectionConfig,
    resources: Sequence[ResourceDocument],
    vectors: np.ndarray,
    embedding_model: str,
    batch_size: int = 128,
) -> None:
    from qdrant_client import models

    if vectors.shape != (len(resources), config.dimension):
        raise ValueError(
            f"vectors shape {vectors.shape} does not match "
            f"({len(resources)}, {config.dimension})"
        )
    for start in range(0, len(resources), batch_size):
        batch_resources = resources[start:start + batch_size]
        batch_vectors = vectors[start:start + batch_size]
        points = [
            models.PointStruct(
                id=qdrant_point_id(resource.resource_id),
                vector={config.vector_name: vector.tolist()},
                payload=resource_payload(resource, config, embedding_model),
            )
            for resource, vector in zip(batch_resources, batch_vectors, strict=True)
        ]
        client.upsert(
            collection_name=config.collection_name,
            points=points,
            wait=True,
        )


def _metadata_key(key: str) -> str:
    if not key or key.startswith(".") or key.endswith("."):
        raise ValueError(f"invalid metadata key: {key!r}")
    return f"metadata.{key}"


def to_qdrant_filter(filters: ResourceFilters | None):
    if filters is None or filters.is_empty():
        return None
    from qdrant_client import models

    must: list[Any] = []
    must_not: list[Any] = []

    for key, value in filters.equals.items():
        must.append(models.FieldCondition(
            key=_metadata_key(key),
            match=models.MatchValue(value=value),
        ))
    for key, values in filters.any_of.items():
        if values:
            must.append(models.FieldCondition(
                key=_metadata_key(key),
                match=models.MatchAny(any=list(values)),
            ))
    for key, values in filters.none_of.items():
        if values:
            must_not.append(models.FieldCondition(
                key=_metadata_key(key),
                match=models.MatchAny(any=list(values)),
            ))
    for key, constraint in filters.ranges.items():
        if not constraint.is_empty():
            must.append(models.FieldCondition(
                key=_metadata_key(key),
                range=models.Range(
                    gt=constraint.gt,
                    gte=constraint.gte,
                    lt=constraint.lt,
                    lte=constraint.lte,
                ),
            ))

    return models.Filter(must=must or None, must_not=must_not or None)


class QdrantResourceRetriever:
    def __init__(
        self,
        client: Any,
        encoder: SentenceTransformerResourceEncoder,
        config: QdrantCollectionConfig,
    ) -> None:
        self.client = client
        self.encoder = encoder
        self.config = config

    @property
    def name(self) -> str:
        return self.encoder.spec.name

    def search(
        self,
        query: str,
        *,
        filters: ResourceFilters | None = None,
        top_k: int = 10,
        exact: bool = False,
    ) -> list[ResourceSearchHit]:
        if top_k <= 0:
            return []
        vector = self.encoder.encode_query(query)
        query_filter = to_qdrant_filter(filters)
        hits = self._query(vector, query_filter, top_k, exact)
        output: list[ResourceSearchHit] = []
        for rank, hit in enumerate(hits, start=1):
            payload = dict(hit.payload or {})
            output.append(ResourceSearchHit(
                rank=rank,
                resource_id=payload.get("resource_id", hit.id),
                title=str(payload.get("title", "")),
                text=str(payload.get("text", "")),
                metadata=dict(payload.get("metadata") or {}),
                domain=str(payload.get("domain", "generic")),
                score=float(hit.score),
                retrieval_score=float(hit.score),
                first_stage_rank=rank,
            ))
        return output

    def get(self, resource_ids: Sequence[str | int]) -> list[ResourceDocument]:
        if not resource_ids:
            return []
        stored = self.client.retrieve(
            collection_name=self.config.collection_name,
            ids=[qdrant_point_id(value) for value in resource_ids],
            with_payload=True,
            with_vectors=False,
        )
        by_id: dict[str, ResourceDocument] = {}
        for point in stored:
            payload = dict(point.payload or {})
            resource = ResourceDocument(
                resource_id=payload.get("resource_id", point.id),
                title=str(payload.get("title", "")),
                text=str(payload.get("text", "")),
                metadata=dict(payload.get("metadata") or {}),
                domain=str(payload.get("domain", "generic")),
                source=payload.get("source"),
            )
            by_id[str(resource.resource_id)] = resource
        return [by_id[str(value)] for value in resource_ids if str(value) in by_id]

    def _query(self, vector: np.ndarray, query_filter: Any, top_k: int, exact: bool):
        from qdrant_client import models

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.config.collection_name,
                query=vector.tolist(),
                using=self.config.vector_name,
                query_filter=query_filter,
                search_params=models.SearchParams(exact=exact),
                with_payload=True,
                limit=top_k,
            )
            return response.points
        # Compatibility with older qdrant-client releases used by the original project.
        return self.client.search(
            collection_name=self.config.collection_name,
            query_vector=(self.config.vector_name, vector.tolist()),
            query_filter=query_filter,
            search_params=models.SearchParams(exact=exact),
            with_payload=True,
            limit=top_k,
        )
