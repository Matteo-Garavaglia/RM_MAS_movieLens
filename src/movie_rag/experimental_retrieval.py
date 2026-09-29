"""Experimental in-memory retrieval components for the 2k-film ablation study.

This module deliberately stays separate from the Qdrant/MCP runtime.  It is used to
compare retrieval models under a common protocol before promoting a new baseline.
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .filters import dataframe_filter_mask


TOKEN_PATTERN = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+(?:[-'][\wÀ-ÖØ-öø-ÿ]+)*", re.UNICODE)


def bm25_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


@dataclass
class BM25Index:
    """Small adapter around Okapi BM25 used only as a lexical baseline."""

    documents: Sequence[str]
    _model: Any = field(default=None, init=False, repr=False)

    def build(self) -> "BM25Index":
        if self._model is not None:
            return self
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank-bm25 to run the lexical baseline.") from exc
        self._model = BM25Okapi([bm25_tokenize(doc) for doc in self.documents])
        return self

    def scores(self, query: str) -> np.ndarray:
        self.build()
        return np.asarray(self._model.get_scores(bm25_tokenize(query)), dtype=np.float64)


@dataclass(frozen=True)
class DenseModelSpec:
    name: str
    model_name: str
    dimension: int
    trust_remote_code: bool = False
    query_prompt_name: str | None = None
    max_seq_length: int | None = None


@dataclass
class SentenceTransformerDenseEncoder:
    """Lazy adapter for multilingual dense models exposed via SentenceTransformers."""

    spec: DenseModelSpec
    device: str | None = None
    batch_size: int = 16
    _model: Any = field(default=None, init=False, repr=False)

    def load(self) -> "SentenceTransformerDenseEncoder":
        if self._model is not None:
            return self
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for Qwen3/GTE experiments."
            ) from exc
        kwargs: dict[str, Any] = {"trust_remote_code": self.spec.trust_remote_code}
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.spec.model_name, **kwargs)
        if self.spec.max_seq_length is not None:
            self._model.max_seq_length = int(self.spec.max_seq_length)
        return self

    def encode_documents(self, documents: Sequence[str]) -> np.ndarray:
        self.load()
        vectors = self._model.encode(
            list(documents),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        self._validate(vectors)
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        self.load()
        kwargs: dict[str, Any] = {
            "batch_size": 1,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        if self.spec.query_prompt_name:
            kwargs["prompt_name"] = self.spec.query_prompt_name
        vector = np.asarray(self._model.encode([query], **kwargs), dtype=np.float32)[0]
        if vector.shape != (self.spec.dimension,):
            raise ValueError(
                f"{self.spec.name}: expected {self.spec.dimension} dimensions, got {vector.shape}"
            )
        return vector

    def _validate(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.spec.dimension:
            raise ValueError(
                f"{self.spec.name}: expected (*, {self.spec.dimension}), got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(f"{self.spec.name}: non-finite embeddings")

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


@dataclass
class BGEM3MultiEncoder:
    """BGE-M3 adapter exposing dense and learned sparse representations together."""

    model_name: str = "BAAI/bge-m3"
    document_max_length: int = 1024
    query_max_length: int = 512
    batch_size: int = 16
    use_fp16: bool | None = None
    _model: Any = field(default=None, init=False, repr=False)

    def load(self) -> "BGEM3MultiEncoder":
        if self._model is not None:
            return self
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError("FlagEmbedding is required for BGE-M3 experiments.") from exc
        use_fp16 = self.use_fp16
        if use_fp16 is None:
            try:
                import torch

                use_fp16 = bool(torch.cuda.is_available())
            except ImportError:
                use_fp16 = False
        self._model = BGEM3FlagModel(self.model_name, use_fp16=bool(use_fp16))
        return self

    @staticmethod
    def _normalize_dense(vectors: Any) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("BGE-M3 returned a zero dense vector")
        return matrix / norms

    def encode_documents(self, documents: Sequence[str]) -> tuple[np.ndarray, list[dict[Any, float]]]:
        self.load()
        out = self._model.encode(
            list(documents),
            batch_size=self.batch_size,
            max_length=self.document_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = self._normalize_dense(out["dense_vecs"])
        sparse = [dict(weights) for weights in out["lexical_weights"]]
        return dense, sparse

    def encode_query(self, query: str) -> tuple[np.ndarray, dict[Any, float]]:
        self.load()
        out = self._model.encode(
            [str(query)],
            batch_size=1,
            max_length=self.query_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = self._normalize_dense(out["dense_vecs"])[0]
        sparse = dict(out["lexical_weights"][0])
        return dense, sparse

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def dense_scores(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    docs = np.asarray(document_vectors, dtype=np.float32)
    if docs.ndim != 2 or docs.shape[1] != len(q):
        raise ValueError(f"Incompatible dense shapes: query={q.shape}, docs={docs.shape}")
    return docs @ q


def sparse_dot(query_weights: Mapping[Any, float], document_weights: Mapping[Any, float]) -> float:
    """Dot product between two learned sparse lexical-weight dictionaries."""

    if len(query_weights) > len(document_weights):
        query_weights, document_weights = document_weights, query_weights
    return float(
        sum(float(weight) * float(document_weights.get(token, 0.0)) for token, weight in query_weights.items())
    )


def sparse_scores(
    query_weights: Mapping[Any, float],
    document_weights: Sequence[Mapping[Any, float]],
) -> np.ndarray:
    return np.asarray(
        [sparse_dot(query_weights, weights) for weights in document_weights],
        dtype=np.float64,
    )


def _rank_positions(scores: np.ndarray, eligible_positions: np.ndarray, depth: int) -> np.ndarray:
    if depth <= 0:
        raise ValueError("depth must be positive")
    eligible_positions = np.asarray(eligible_positions, dtype=int)
    if len(eligible_positions) == 0:
        return eligible_positions
    order = np.argsort(-np.asarray(scores)[eligible_positions], kind="stable")
    return eligible_positions[order[:depth]]


def retrieve_from_scores(
    dataframe: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
    filters: Mapping[str, Any] | None = None,
    id_column: str = "movieId",
    title_column: str = "tmdb_title",
    document_column: str = "document__D_full_enriched_v2",
) -> pd.DataFrame:
    """Apply the shared structured filters and return a ranked frame."""

    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(score_array) != len(dataframe):
        raise ValueError("scores and dataframe must have the same number of rows")
    mask = dataframe_filter_mask(dataframe, filters)
    eligible = np.flatnonzero(mask.to_numpy())
    positions = _rank_positions(score_array, eligible, top_k)
    columns = [c for c in (id_column, title_column, document_column) if c in dataframe.columns]
    result = dataframe.iloc[positions][columns].copy().reset_index(drop=True)
    result.insert(0, "_position", positions)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=int))
    result["score"] = score_array[positions]
    return result


def weighted_rrf(
    rankings: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    top_k: int = 20,
    id_column: str = "movieId",
) -> pd.DataFrame:
    """Fuse ranking positions rather than incomparable raw retrieval scores."""

    if rrf_k < 0 or top_k <= 0:
        raise ValueError("rrf_k must be non-negative and top_k positive")
    if not rankings:
        return pd.DataFrame()
    effective_weights = {name: 1.0 for name in rankings}
    if weights:
        effective_weights.update({name: float(value) for name, value in weights.items()})

    scores: dict[Any, float] = {}
    provenance: dict[Any, dict[str, Any]] = {}
    for system, frame in rankings.items():
        weight = effective_weights[system]
        if weight < 0:
            raise ValueError("RRF weights cannot be negative")
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            movie_id = row_dict[id_column]
            rank = int(row_dict["rank"])
            scores[movie_id] = scores.get(movie_id, 0.0) + weight / (rrf_k + rank)
            if movie_id not in provenance:
                provenance[movie_id] = dict(row_dict)
            provenance[movie_id][f"{system}_rank"] = rank
            provenance[movie_id][f"{system}_score"] = float(row_dict.get("score", np.nan))

    ordered_ids = sorted(scores, key=lambda mid: (-scores[mid], str(mid)))[:top_k]
    rows = []
    for final_rank, movie_id in enumerate(ordered_ids, start=1):
        row = provenance[movie_id].copy()
        row["rank"] = final_rank
        row["score"] = scores[movie_id]
        row["rrf_score"] = scores[movie_id]
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class SentenceTransformerCrossEncoderReranker:
    """Generic adapter for Qwen3 and GTE rerankers exposed as CrossEncoder models."""

    model_name: str
    trust_remote_code: bool = False
    device: str | None = None
    batch_size: int = 8
    _model: Any = field(default=None, init=False, repr=False)
    max_length: int = 1024

    def load(self) -> "SentenceTransformerCrossEncoderReranker":
        if self._model is not None:
            return self
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for this reranker.") from exc
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.device:
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, max_length=self.max_length, **kwargs)
        return self

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        self.load()
        pairs = [(str(query), str(document)) for document in documents]
        if not pairs:
            return np.empty(0, dtype=np.float64)
        values = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(scores) != len(documents):
            raise ValueError("Reranker returned a score count different from the candidate count")
        return scores

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
