from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .resource_models import (
    PipelineInfo,
    ResourceDocument,
    ResourceFilters,
    ResourceSearchHit,
    ResourceSearchResponse,
)


@dataclass(frozen=True)
class SearchProfile:
    profile_id: str
    candidate_k: int
    final_k: int
    rerank: bool = False


FAST_PROFILE = SearchProfile("fast", candidate_k=10, final_k=10, rerank=False)
QUALITY_PROFILE = SearchProfile("quality", candidate_k=20, final_k=10, rerank=True)


class ResourceRetriever(Protocol):
    @property
    def name(self) -> str: ...

    def search(
        self,
        query: str,
        *,
        filters: ResourceFilters | None = None,
        top_k: int = 10,
        exact: bool = False,
    ) -> list[ResourceSearchHit]: ...

    def get(self, resource_ids: Sequence[str | int]) -> list[ResourceDocument]: ...


class ResourceReranker(Protocol):
    @property
    def name(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[ResourceSearchHit],
        *,
        top_k: int,
    ) -> list[ResourceSearchHit]: ...


class ScoreRerankerAdapter:
    """Adapter for the existing reranker classes exposing ``score(query, docs)``."""

    def __init__(self, scorer: Any, name: str | None = None) -> None:
        self.scorer = scorer
        self._name = name or getattr(scorer, "model_name", scorer.__class__.__name__)

    @property
    def name(self) -> str:
        return self._name

    def rerank(
        self,
        query: str,
        candidates: Sequence[ResourceSearchHit],
        *,
        top_k: int,
    ) -> list[ResourceSearchHit]:
        if not candidates:
            return []
        scores = np.asarray(
            self.scorer.score(query, [candidate.text for candidate in candidates]),
            dtype=np.float64,
        ).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError("reranker returned a different number of scores")
        order = np.argsort(-scores, kind="stable")[:top_k]
        output: list[ResourceSearchHit] = []
        for rank, idx in enumerate(order, start=1):
            candidate = candidates[int(idx)]
            output.append(candidate.model_copy(update={
                "rank": rank,
                "score": float(scores[idx]),
                "reranker_score": float(scores[idx]),
                "first_stage_rank": candidate.first_stage_rank or candidate.rank,
            }))
        return output


class RetrievalService:
    """Stable application boundary used by notebooks, MCP and future agents."""

    def __init__(
        self,
        retriever: ResourceRetriever,
        *,
        reranker: ResourceReranker | None = None,
        profiles: Mapping[str, SearchProfile] | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.profiles = dict(profiles or {
            FAST_PROFILE.profile_id: FAST_PROFILE,
            QUALITY_PROFILE.profile_id: QUALITY_PROFILE,
        })

    def search(
        self,
        query: str,
        *,
        filters: ResourceFilters | None = None,
        profile_id: str = "fast",
        limit: int | None = None,
        exact: bool = False,
    ) -> ResourceSearchResponse:
        if profile_id not in self.profiles:
            raise KeyError(f"unknown search profile: {profile_id}")
        profile = self.profiles[profile_id]
        final_k = min(limit, profile.final_k) if limit is not None else profile.final_k
        candidate_k = max(profile.candidate_k, final_k)

        candidates = self.retriever.search(
            query,
            filters=filters,
            top_k=candidate_k,
            exact=exact,
        )

        if profile.rerank:
            if self.reranker is None:
                raise RuntimeError(
                    f"profile {profile_id!r} requires a reranker, but none is configured"
                )
            results = self.reranker.rerank(query, candidates, top_k=final_k)
            reranker_name = self.reranker.name
        else:
            results = [
                hit.model_copy(update={"rank": rank})
                for rank, hit in enumerate(candidates[:final_k], start=1)
            ]
            reranker_name = None

        pipeline = PipelineInfo(
            profile_id=profile.profile_id,
            retriever=self.retriever.name,
            reranker=reranker_name,
            candidate_k=candidate_k,
            final_k=final_k,
            rerank_enabled=profile.rerank,
        )
        return ResourceSearchResponse(
            status="ok" if results else "no_results",
            query=query,
            filters=filters or ResourceFilters(),
            pipeline=pipeline,
            returned_count=len(results),
            results=results,
        )

    def get_resources(self, resource_ids: Sequence[str | int]) -> list[ResourceDocument]:
        return self.retriever.get(resource_ids)
