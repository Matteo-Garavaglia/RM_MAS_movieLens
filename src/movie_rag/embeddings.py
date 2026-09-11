"""BGE-M3 dense encoding and a validated local embedding cache."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import BASELINE, DenseBaselineConfig


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Return float32 row-normalized vectors and reject zero vectors."""

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize zero-length vectors")
    return array / norms


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return array / norm


@dataclass
class BGEM3DenseEncoder:
    """Lazy FlagEmbedding adapter matching the winning notebook settings."""

    config: DenseBaselineConfig = BASELINE
    use_fp16: bool | None = None
    _model: Any = field(default=None, init=False, repr=False)
    _query_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.config.embedding_model

    def load(self) -> "BGEM3DenseEncoder":
        if self._model is not None:
            return self
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required. Install the project with the 'bge' extra."
            ) from exc

        use_fp16 = self.use_fp16
        if use_fp16 is None:
            try:
                import torch

                use_fp16 = bool(torch.cuda.is_available())
            except ImportError:
                use_fp16 = False

        self._model = BGEM3FlagModel(self.model_name, use_fp16=bool(use_fp16))
        return self

    def _encode(self, texts: Sequence[str], max_length: int, batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, self.config.embedding_dimension), dtype=np.float32)
        self.load()
        output = self._model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=max_length,
        )
        if "dense_vecs" not in output:
            raise KeyError("BGE-M3 output does not contain 'dense_vecs'")
        vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
        if vectors.shape[1] != self.config.embedding_dimension:
            raise ValueError(
                f"Expected {self.config.embedding_dimension} dimensions, got {vectors.shape[1]}"
            )
        return l2_normalize_rows(vectors) if self.config.normalize_embeddings else vectors

    def encode_documents(self, documents: Sequence[str]) -> np.ndarray:
        return self._encode(
            documents,
            max_length=self.config.document_max_length,
            batch_size=self.config.embedding_batch_size,
        )

    def encode_query(self, query: str) -> np.ndarray:
        query = str(query).strip()
        if not query:
            raise ValueError("The query cannot be empty")
        if query not in self._query_cache:
            self._query_cache[query] = self._encode(
                [query],
                max_length=self.config.query_max_length,
                batch_size=1,
            )[0]
        return self._query_cache[query].copy()


@dataclass(frozen=True)
class EmbeddingCacheResult:
    vectors: np.ndarray
    source: str
    metadata: dict[str, Any]


def save_embedding_cache(
    path: str | Path,
    movie_ids: Sequence[int],
    vectors: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(vectors, dtype=np.float32)
    ids = np.asarray(movie_ids, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError("Embedding rows and movie IDs must have the same length")
    np.savez_compressed(
        path,
        movie_ids=ids,
        vectors=matrix,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_embedding_cache(
    path: str | Path,
    expected_movie_ids: Sequence[int],
    expected_metadata: dict[str, Any],
) -> EmbeddingCacheResult | None:
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as artifact:
        cached_ids = artifact["movie_ids"].astype(np.int64)
        vectors = artifact["vectors"].astype(np.float32)
        metadata = json.loads(str(artifact["metadata"].item()))
    expected_ids = np.asarray(expected_movie_ids, dtype=np.int64)
    if not np.array_equal(cached_ids, expected_ids):
        return None
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            return None
    return EmbeddingCacheResult(vectors=vectors, source="cache", metadata=metadata)


def load_or_encode_documents(
    documents: Sequence[str],
    movie_ids: Sequence[int],
    dataset_fingerprint: str,
    encoder: BGEM3DenseEncoder,
    cache_path: str | Path | None = None,
) -> EmbeddingCacheResult:
    metadata = {
        "dataset_fingerprint": dataset_fingerprint,
        "embedding_model": encoder.config.embedding_model,
        "embedding_dimension": encoder.config.embedding_dimension,
        "document_max_length": encoder.config.document_max_length,
        "normalized": encoder.config.normalize_embeddings,
        "representation_version": encoder.config.representation_version,
    }
    if cache_path is not None:
        cached = load_embedding_cache(cache_path, movie_ids, metadata)
        if cached is not None:
            return cached

    vectors = encoder.encode_documents(documents)
    if cache_path is not None:
        save_embedding_cache(cache_path, movie_ids, vectors, metadata)
    return EmbeddingCacheResult(vectors=vectors, source="computed", metadata=metadata)
