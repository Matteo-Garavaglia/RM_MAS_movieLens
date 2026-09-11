"""Reusable dense-retrieval components for the MovieLens/WhoTeach prototype."""

from .config import BASELINE, D_FULL_ENRICHED_FIELDS, DenseBaselineConfig
from .documents import catalog_fingerprint, prepare_catalog
from .embeddings import BGEM3DenseEncoder, load_or_encode_documents
from .evaluation import dcg_at_k, evaluate_results, ndcg_at_k
from .filters import MovieFilters
from .qdrant_store import (
    DenseMovieRetriever,
    connect_qdrant,
    ensure_dense_collection,
    ensure_payload_indexes,
    index_catalog,
)
from .validation import validate_query_set

__all__ = [
    "BASELINE",
    "D_FULL_ENRICHED_FIELDS",
    "DenseBaselineConfig",
    "prepare_catalog",
    "catalog_fingerprint",
    "BGEM3DenseEncoder",
    "load_or_encode_documents",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_results",
    "MovieFilters",
    "DenseMovieRetriever",
    "connect_qdrant",
    "ensure_dense_collection",
    "ensure_payload_indexes",
    "index_catalog",
    "validate_query_set",
]
