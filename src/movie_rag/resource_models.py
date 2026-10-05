from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator

Scalar: TypeAlias = str | int | float | bool


class ResourceDocument(BaseModel):
    """Domain-agnostic searchable unit.

    MovieLens currently maps one movie to one ResourceDocument.  WhoTeach can map
    one semantic chunk / learning object to one ResourceDocument without changing
    the retrieval contract.
    """

    resource_id: str | int
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    domain: str = "generic"
    source: str | None = None

    @field_validator("title", "text", "domain")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class RangeConstraint(BaseModel):
    gt: float | None = None
    gte: float | None = None
    lt: float | None = None
    lte: float | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in (self.gt, self.gte, self.lt, self.lte))


class ResourceFilters(BaseModel):
    """Generic metadata constraints understood by the Qdrant layer.

    Keys refer to entries of ``ResourceDocument.metadata``.  This prevents the
    infrastructure from knowing domain concepts such as movie genres or school
    levels. Domain adapters translate user/domain filters into this schema.
    """

    equals: dict[str, Scalar] = Field(default_factory=dict)
    any_of: dict[str, list[Scalar]] = Field(default_factory=dict)
    none_of: dict[str, list[Scalar]] = Field(default_factory=dict)
    ranges: dict[str, RangeConstraint] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.equals or self.any_of or self.none_of or self.ranges)


class ResourceSearchHit(BaseModel):
    rank: int
    resource_id: str | int
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    domain: str = "generic"
    score: float
    retrieval_score: float | None = None
    reranker_score: float | None = None
    first_stage_rank: int | None = None


class PipelineInfo(BaseModel):
    profile_id: str
    retriever: str
    reranker: str | None = None
    candidate_k: int
    final_k: int
    rerank_enabled: bool


class ResourceSearchResponse(BaseModel):
    status: Literal["ok", "no_results"]
    query: str
    filters: ResourceFilters = Field(default_factory=ResourceFilters)
    pipeline: PipelineInfo
    returned_count: int
    results: list[ResourceSearchHit] = Field(default_factory=list)
