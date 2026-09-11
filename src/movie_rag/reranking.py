"""BGE cross-encoder reranking on top of Qdrant dense candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import RERANKER, RerankerConfig
from .filters import MovieFilters


@dataclass
class BGEV2M3Reranker:
    """Lazy adapter for ``BAAI/bge-reranker-v2-m3``.

    The model is deliberately kept outside Qdrant.  It receives the raw query
    and the document stored in each candidate payload, then returns one
    query-dependent score per pair.
    """

    config: RerankerConfig = RERANKER
    use_fp16: bool | None = None
    _model: Any = field(default=None, init=False, repr=False)

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def load(self) -> "BGEV2M3Reranker":
        if self._model is not None:
            return self
        try:
            from FlagEmbedding import FlagReranker
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

        self._model = FlagReranker(self.model_name, use_fp16=bool(use_fp16))
        return self

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """Return one finite relevance score per query-document pair."""

        cleaned_query = str(query).strip()
        if not cleaned_query:
            raise ValueError("The query cannot be empty")

        cleaned_documents = [str(document).strip() for document in documents]
        if not cleaned_documents:
            return np.empty(0, dtype=np.float64)
        empty_positions = [
            position for position, document in enumerate(cleaned_documents) if not document
        ]
        if empty_positions:
            raise ValueError(f"Candidate documents are empty at positions: {empty_positions}")

        self.load()
        pairs = [[cleaned_query, document] for document in cleaned_documents]
        raw_scores = self._model.compute_score(
            pairs,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            normalize=self.config.normalize_scores,
        )
        scores = np.atleast_1d(np.asarray(raw_scores, dtype=np.float64)).reshape(-1)
        if len(scores) != len(cleaned_documents):
            raise ValueError(
                f"Reranker returned {len(scores)} scores for "
                f"{len(cleaned_documents)} documents"
            )
        if not np.isfinite(scores).all():
            raise ValueError("Reranker returned non-finite scores")
        return scores


def _empty_reranked_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    for column, dtype in (
        ("dense_rank", "int64"),
        ("dense_score", "float64"),
        ("reranker_score", "float64"),
        ("score", "float64"),
    ):
        if column not in result:
            result[column] = pd.Series(dtype=dtype)
    return result


def rerank_candidates(
    query: str,
    candidates: pd.DataFrame,
    reranker: Any,
    top_k: int = RERANKER.top_k,
    document_column: str = RERANKER.document_payload_field,
) -> pd.DataFrame:
    """Rerank a dense candidate frame without mutating it.

    ``score`` in the returned frame is the final reranker score.  The original
    Qdrant values remain available as ``dense_rank`` and ``dense_score``.
    Ties are resolved using the dense candidate order, making the result stable.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if candidates.empty:
        return _empty_reranked_frame(candidates)

    required_columns = {"rank", "score", document_column}
    missing = sorted(required_columns - set(candidates.columns))
    if missing:
        raise KeyError(f"Candidate frame lacks required columns: {missing}")

    working = candidates.copy().reset_index(drop=True)
    working["dense_rank"] = working["rank"].astype(int)
    working["dense_score"] = working["score"].astype(float)
    working["_candidate_order"] = np.arange(len(working), dtype=int)

    scores = np.asarray(
        reranker.score(query, working[document_column].astype(str).tolist()),
        dtype=np.float64,
    ).reshape(-1)
    if len(scores) != len(working):
        raise ValueError(
            f"Reranker returned {len(scores)} scores for {len(working)} candidates"
        )
    if not np.isfinite(scores).all():
        raise ValueError("Reranker returned non-finite scores")

    working["reranker_score"] = scores
    working = working.sort_values(
        ["reranker_score", "_candidate_order"],
        ascending=[False, True],
        kind="stable",
    ).head(top_k)
    working = working.drop(columns="_candidate_order").reset_index(drop=True)
    working["rank"] = np.arange(1, len(working) + 1, dtype=int)
    working["score"] = working["reranker_score"]
    return working


@dataclass
class RerankedMovieRetriever:
    """Application boundary for ``dense Qdrant -> BGE reranker``."""

    dense_retriever: Any
    reranker: Any
    config: RerankerConfig = RERANKER

    def retrieve_candidates(
        self,
        query: str,
        filters: Mapping[str, Any] | MovieFilters | None = None,
        candidate_pool_size: int | None = None,
        exact: bool = False,
        score_threshold: float | None = None,
    ) -> pd.DataFrame:
        pool_size = (
            self.config.candidate_pool_size
            if candidate_pool_size is None
            else int(candidate_pool_size)
        )
        if pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        return self.dense_retriever.search(
            query=query,
            filters=filters,
            top_k=pool_size,
            exact=exact,
            score_threshold=score_threshold,
        )

    def search(
        self,
        query: str,
        filters: Mapping[str, Any] | MovieFilters | None = None,
        candidate_pool_size: int | None = None,
        top_k: int | None = None,
        exact: bool = False,
        score_threshold: float | None = None,
    ) -> pd.DataFrame:
        pool_size = (
            self.config.candidate_pool_size
            if candidate_pool_size is None
            else int(candidate_pool_size)
        )
        final_size = self.config.top_k if top_k is None else int(top_k)
        if pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if final_size <= 0:
            raise ValueError("top_k must be positive")
        if final_size > pool_size:
            raise ValueError("top_k cannot exceed candidate_pool_size")

        candidates = self.retrieve_candidates(
            query=query,
            filters=filters,
            candidate_pool_size=pool_size,
            exact=exact,
            score_threshold=score_threshold,
        )
        return rerank_candidates(
            query=query,
            candidates=candidates,
            reranker=self.reranker,
            top_k=final_size,
            document_column=self.config.document_payload_field,
        )

    def get_movies(self, movie_ids: Sequence[int]) -> pd.DataFrame:
        return self.dense_retriever.get_movies(movie_ids)
