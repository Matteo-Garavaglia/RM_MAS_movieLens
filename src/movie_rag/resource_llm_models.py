from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .resource_mcp_gateway import MCPToolTrace
from .resource_models import ResourceSearchResponse


class ResourceRetrievalPlan(BaseModel):
    """LLM planner output; deliberately excludes technical retrieval configuration."""

    semantic_query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    clarification_required: bool = False
    clarification_question: str | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def _clarification_consistency(self):
        self.semantic_query = str(self.semantic_query).strip()
        if not self.semantic_query and not self.clarification_required:
            raise ValueError("semantic_query cannot be blank unless clarification is required")
        if self.clarification_required:
            if not (self.clarification_question or "").strip():
                raise ValueError("clarification_question is required when clarification_required=True")
        else:
            self.clarification_question = None
        return self


class GroundedRecommendation(BaseModel):
    resource_id: str | int
    title: str
    reason: str


class GroundedAnswer(BaseModel):
    status: Literal["ok", "no_results", "needs_clarification"]
    answer: str
    recommendations: list[GroundedRecommendation] = Field(default_factory=list)


class LLMCallRecord(BaseModel):
    role: str
    prompt_version: str
    provider_trace: dict[str, Any] = Field(default_factory=dict)


class ResourceConversationState(BaseModel):
    turn_id: str = Field(default_factory=lambda: uuid4().hex)
    workflow_id: str
    domain_id: str
    original_query: str
    status: Literal[
        "created",
        "needs_clarification",
        "searched",
        "completed",
        "no_results",
        "error",
    ] = "created"
    plan: ResourceRetrievalPlan | None = None
    search_response: ResourceSearchResponse | None = None
    final_answer: GroundedAnswer | None = None
    llm_traces: list[LLMCallRecord] = Field(default_factory=list)
    tool_traces: list[MCPToolTrace] = Field(default_factory=list)
    grounding_violations: list[str] = Field(default_factory=list)

    @property
    def latest_search(self) -> ResourceSearchResponse | None:
        return self.search_response


__all__ = [
    "ResourceRetrievalPlan",
    "GroundedRecommendation",
    "GroundedAnswer",
    "LLMCallRecord",
    "ResourceConversationState",
]
