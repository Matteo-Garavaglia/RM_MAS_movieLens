from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .resource_adapters import MovieLensV2Adapter, movie_filters_to_resource_filters
from .resource_embeddings import GTE_MULTILINGUAL_BASE, SentenceTransformerResourceEncoder
from .resource_qdrant import QdrantCollectionConfig, QdrantResourceRetriever
from .resource_service import RetrievalService, ResourceReranker


@dataclass(frozen=True)
class MovieLensV2RuntimeConfig:
    collection_name: str = "movies_d_full_enriched_v2_gte_multilingual_base_v1"
    vector_name: str = "dense"
    schema_version: str = "movielens_v2_resource_payload_v1"

    @property
    def qdrant(self) -> QdrantCollectionConfig:
        return QdrantCollectionConfig(
            collection_name=self.collection_name,
            dimension=GTE_MULTILINGUAL_BASE.dimension,
            vector_name=self.vector_name,
            schema_version=self.schema_version,
        )


def build_movielens_v2_service(
    client: Any,
    *,
    reranker: ResourceReranker | None = None,
    encoder: SentenceTransformerResourceEncoder | None = None,
    config: MovieLensV2RuntimeConfig | None = None,
) -> RetrievalService:
    config = config or MovieLensV2RuntimeConfig()
    encoder = encoder or SentenceTransformerResourceEncoder(GTE_MULTILINGUAL_BASE)
    retriever = QdrantResourceRetriever(client, encoder, config.qdrant)
    return RetrievalService(retriever, reranker=reranker)


__all__ = [
    "MovieLensV2Adapter",
    "MovieLensV2RuntimeConfig",
    "build_movielens_v2_service",
    "movie_filters_to_resource_filters",
]
