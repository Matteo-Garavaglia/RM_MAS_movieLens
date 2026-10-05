from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .resource_models import ResourceDocument


@dataclass(frozen=True)
class DenseEncoderSpec:
    name: str
    model_name: str
    dimension: int
    trust_remote_code: bool = False
    batch_size: int = 16
    normalize_embeddings: bool = True
    max_seq_length: int | None = None
    query_prompt_name: str | None = None


GTE_MULTILINGUAL_BASE = DenseEncoderSpec(
    name="gte_multilingual_base_dense",
    model_name="Alibaba-NLP/gte-multilingual-base",
    dimension=768,
    trust_remote_code=True,
    batch_size=16,
    normalize_embeddings=True,
    # Keep native sequence length to match the retrieval ablation configuration.
    max_seq_length=None,
    query_prompt_name=None,
)


@dataclass
class SentenceTransformerResourceEncoder:
    spec: DenseEncoderSpec = GTE_MULTILINGUAL_BASE
    device: str | None = None
    _model: Any = field(default=None, init=False, repr=False)

    def load(self) -> "SentenceTransformerResourceEncoder":
        if self._model is not None:
            return self
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "sentence-transformers is required for resource embeddings"
            ) from exc
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.spec.trust_remote_code,
        }
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.spec.model_name, **kwargs)
        if self.spec.max_seq_length is not None:
            self._model.max_seq_length = int(self.spec.max_seq_length)
        return self

    def unload(self) -> None:
        self._model = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    @property
    def model_name(self) -> str:
        return self.spec.model_name

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    def encode_documents(self, documents: Sequence[str]) -> np.ndarray:
        self.load()
        vectors = self._model.encode(
            list(map(str, documents)),
            batch_size=self.spec.batch_size,
            normalize_embeddings=self.spec.normalize_embeddings,
            show_progress_bar=len(documents) >= 64,
        )
        result = np.asarray(vectors, dtype=np.float32)
        self._validate(result, expected_rows=len(documents))
        return result

    def encode_query(self, query: str) -> np.ndarray:
        self.load()
        kwargs: dict[str, Any] = {
            "batch_size": 1,
            "normalize_embeddings": self.spec.normalize_embeddings,
            "show_progress_bar": False,
        }
        if self.spec.query_prompt_name:
            kwargs["prompt_name"] = self.spec.query_prompt_name
        vector = self._model.encode([str(query)], **kwargs)
        result = np.asarray(vector, dtype=np.float32)
        self._validate(result, expected_rows=1)
        return result[0]

    def _validate(self, vectors: np.ndarray, expected_rows: int) -> None:
        if vectors.shape != (expected_rows, self.spec.dimension):
            raise ValueError(
                f"unexpected embedding shape {vectors.shape}; "
                f"expected {(expected_rows, self.spec.dimension)}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("embedding contains non-finite values")
        if self.spec.normalize_embeddings and expected_rows:
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                raise ValueError("embeddings are expected to be L2-normalized")


def resource_fingerprint(resources: Sequence[ResourceDocument], spec: DenseEncoderSpec) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "model": spec.model_name,
        "dimension": spec.dimension,
        "normalize": spec.normalize_embeddings,
        "max_seq_length": spec.max_seq_length,
    }, sort_keys=True).encode("utf-8"))
    for resource in resources:
        digest.update(str(resource.resource_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resource.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_or_encode_resources(
    resources: Sequence[ResourceDocument],
    encoder: SentenceTransformerResourceEncoder,
    cache_path: str | Path,
) -> tuple[np.ndarray, str]:
    """Return vectors and their source ('cache' or 'encoded')."""

    cache_path = Path(cache_path)
    fingerprint = resource_fingerprint(resources, encoder.spec)
    ids = np.asarray([str(r.resource_id) for r in resources], dtype=str)

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as artifact:
            cached_fingerprint = str(artifact["fingerprint"].item())
            cached_ids = artifact["ids"].astype(str)
            vectors = artifact["vectors"].astype(np.float32)
        if (
            cached_fingerprint == fingerprint
            and np.array_equal(cached_ids, ids)
            and vectors.shape == (len(resources), encoder.dimension)
        ):
            return vectors, "cache"

    vectors = encoder.encode_documents([r.text for r in resources])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fingerprint=np.asarray(fingerprint),
        ids=ids,
        vectors=vectors,
    )
    return vectors, "encoded"
