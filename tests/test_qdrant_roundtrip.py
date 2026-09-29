from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from movie_rag import BASELINE
from movie_rag.documents import prepare_catalog
from movie_rag.qdrant_store import (
    DenseMovieRetriever,
    ensure_dense_collection,
    index_catalog,
    validate_dense_collection,
)
from movie_rag.validation import validate_query_set


qdrant_client = pytest.importorskip("qdrant_client")


class StaticEncoder:
    def __init__(self, query_vector):
        self.query_vector = query_vector

    def encode_query(self, query):
        return self.query_vector.copy()


def unit_vector(position: int) -> np.ndarray:
    vector = np.zeros(BASELINE.embedding_dimension, dtype=np.float32)
    vector[position] = 1.0
    return vector


def test_in_memory_qdrant_roundtrip_and_equivalence(sample_catalog):
    client = qdrant_client.QdrantClient(":memory:")
    prepared = prepare_catalog(sample_catalog)
    second = unit_vector(1)
    second[0] = 0.5
    second /= np.linalg.norm(second)
    third = unit_vector(2)
    third[0] = 0.1
    third /= np.linalg.norm(third)
    vectors = np.vstack([unit_vector(0), second, third])

    collection = ensure_dense_collection(client)
    assert collection.created is True
    result = index_catalog(client, prepared, vectors)
    assert result.dataset_points == 3
    assert result.collection_points == 3
    validate_dense_collection(client)

    wrong_model = replace(BASELINE, embedding_model="BAAI/not-the-indexed-model")
    with pytest.raises(ValueError, match="only 0 match embedding model"):
        validate_dense_collection(client, config=wrong_model)

    encoder = StaticEncoder(unit_vector(0))
    retriever = DenseMovieRetriever(client=client, encoder=encoder)
    hits = retriever.search("memory", top_k=2, exact=True)
    assert hits.iloc[0]["movieId"] == 1

    filtered = retriever.search(
        "memory",
        filters={"exclude_genres": ["Drama"]},
        top_k=3,
        exact=True,
    )
    assert filtered["movieId"].tolist() == [2]

    comparison = validate_query_set(
        prepared,
        vectors,
        retriever,
        [{"query_id": "Q1", "query": "memory", "filters": {}}],
        top_k=3,
    )
    assert bool(comparison.iloc[0]["same_order"])
    assert comparison.iloc[0]["overlap@3"] == 1.0
