from __future__ import annotations

import numpy as np

from movie_rag import BASELINE
from movie_rag.embeddings import BGEM3DenseEncoder, load_or_encode_documents


class FakeBGEModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size, max_length):
        self.calls += 1
        vectors = np.zeros((len(texts), BASELINE.embedding_dimension), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index] = 2.0
        return {"dense_vecs": vectors}


def test_encoder_normalizes_and_caches_queries():
    encoder = BGEM3DenseEncoder(use_fp16=False)
    fake_model = FakeBGEModel()
    encoder._model = fake_model

    documents = encoder.encode_documents(["a", "b"])
    assert documents.shape == (2, BASELINE.embedding_dimension)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)

    first = encoder.encode_query("memory")
    second = encoder.encode_query("memory")
    assert np.array_equal(first, second)
    assert fake_model.calls == 2  # one document batch plus one cached query batch


def test_embedding_cache_is_guarded_by_fingerprint(tmp_path):
    encoder = BGEM3DenseEncoder(use_fp16=False)
    fake_model = FakeBGEModel()
    encoder._model = fake_model
    cache_path = tmp_path / "vectors.npz"

    first = load_or_encode_documents(
        documents=["a", "b"],
        movie_ids=[1, 2],
        dataset_fingerprint="first",
        encoder=encoder,
        cache_path=cache_path,
    )
    second = load_or_encode_documents(
        documents=["a", "b"],
        movie_ids=[1, 2],
        dataset_fingerprint="first",
        encoder=encoder,
        cache_path=cache_path,
    )
    third = load_or_encode_documents(
        documents=["a", "b"],
        movie_ids=[1, 2],
        dataset_fingerprint="changed",
        encoder=encoder,
        cache_path=cache_path,
    )

    assert first.source == "computed"
    assert second.source == "cache"
    assert third.source == "computed"
    assert fake_model.calls == 2

