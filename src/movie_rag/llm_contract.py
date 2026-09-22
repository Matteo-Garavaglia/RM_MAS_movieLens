"""Provider-neutral contracts for LLM agents and workflow state.

The models in this module are deliberately independent from the OpenAI and
Gemini SDKs.  They form the stable boundary between prompts, provider adapters,
the deterministic orchestrator and the existing MCP contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .mcp_contract import (
    MovieDetailsResponse,
    MovieFilterInput,
    SearchMoviesResponse,
)


LLM_CONTRACT_VERSION = "movie-llm/v1"
QueryRewriteStrategy = Literal["none", "translate_to_catalog_language"]


class LLMContractModel(BaseModel):
    """Strict base model used for LLM-generated and orchestration data."""

    model_config = ConfigDict(extra="forbid")


def _clean_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise TypeError("Expected a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = str(item).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


class LLMUsage(LLMContractModel):
    """Normalized usage information; providers may omit unavailable fields."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMCallTrace(LLMContractModel):
    """Audit record for one provider call without storing prompts or secrets."""

    agent: str
    provider: Literal["gemini", "openai"]
    profile_id: str
    model: str
    prompt_version: str
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: float = Field(ge=0)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    request_id: str | None = None


class ToolCallTrace(LLMContractModel):
    """Audit record for one application-side MCP invocation."""

    tool: str
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: float = Field(ge=0)
    status: Literal["ok", "no_results", "error"]
    returned_count: int | None = Field(default=None, ge=0)
    error_type: str | None = None


class RetrievalPlan(LLMContractModel):
    """Structured output of the Planner/Elicitation agent.

    The plan contains user semantics and catalog constraints, never technical
    retrieval choices such as collection names, embedding models or reranking.
    """

    action: Literal["search", "clarify"]
    detected_language: str = Field(min_length=2, max_length=50)
    semantic_query: str | None = Field(default=None, max_length=2000)
    semantic_query_language: str | None = Field(default=None, max_length=50)
    rewrite_strategy: QueryRewriteStrategy
    filters: MovieFilterInput = Field(default_factory=MovieFilterInput)
    ambiguities: list[str] = Field(default_factory=list)
    clarification_question: str | None = Field(default=None, max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("ambiguities", mode="before")
    @classmethod
    def _clean_ambiguities(cls, value: Any) -> list[str]:
        return _clean_text_list(value)

    @field_validator(
        "detected_language",
        "semantic_query",
        "semantic_query_language",
        "clarification_question",
        "reason",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_action_payload(self) -> "RetrievalPlan":
        if self.action == "search":
            if not self.semantic_query:
                raise ValueError("semantic_query is required when action='search'")
            if not self.semantic_query_language:
                raise ValueError(
                    "semantic_query_language is required when action='search'"
                )
            if self.clarification_question is not None:
                raise ValueError(
                    "clarification_question must be null when action='search'"
                )
        else:
            if not self.clarification_question:
                raise ValueError(
                    "clarification_question is required when action='clarify'"
                )
            if self.semantic_query is not None:
                raise ValueError("semantic_query must be null when action='clarify'")
            if self.semantic_query_language is not None:
                raise ValueError(
                    "semantic_query_language must be null when action='clarify'"
                )
        return self


class CriticDecision(LLMContractModel):
    """Structured judgment produced by the optional Critic agent."""

    decision: Literal["accept", "retry", "clarify"]
    reason: str = Field(min_length=1, max_length=1500)
    constraint_violations: list[str] = Field(default_factory=list)
    semantic_weaknesses: list[str] = Field(default_factory=list)
    suggested_query: str | None = Field(default=None, max_length=2000)
    clarification_question: str | None = Field(default=None, max_length=1000)

    @field_validator("constraint_violations", "semantic_weaknesses", mode="before")
    @classmethod
    def _clean_lists(cls, value: Any) -> list[str]:
        return _clean_text_list(value)

    @model_validator(mode="after")
    def _validate_decision_payload(self) -> "CriticDecision":
        if self.decision == "retry":
            if not self.suggested_query:
                raise ValueError("suggested_query is required for decision='retry'")
            if self.clarification_question is not None:
                raise ValueError(
                    "clarification_question must be null for decision='retry'"
                )
        elif self.decision == "clarify":
            if not self.clarification_question:
                raise ValueError(
                    "clarification_question is required for decision='clarify'"
                )
            if self.suggested_query is not None:
                raise ValueError("suggested_query must be null for decision='clarify'")
        else:
            if self.suggested_query is not None:
                raise ValueError("suggested_query must be null for decision='accept'")
            if self.clarification_question is not None:
                raise ValueError(
                    "clarification_question must be null for decision='accept'"
                )
        return self


class GroundedRecommendation(LLMContractModel):
    """One recommendation anchored to an ID returned by the MCP search."""

    movie_id: int = Field(ge=0)
    title: str = Field(min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=2000)


class GroundedAnswer(LLMContractModel):
    """Structured final answer generated only from MCP-provided catalog data."""

    status: Literal["ok", "no_results"]
    response_language: str = Field(min_length=2, max_length=50)
    answer: str = Field(min_length=1, max_length=8000)
    recommendations: list[GroundedRecommendation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("limitations", mode="before")
    @classmethod
    def _clean_limitations(cls, value: Any) -> list[str]:
        return _clean_text_list(value)

    @model_validator(mode="after")
    def _validate_status(self) -> "GroundedAnswer":
        if self.status == "no_results" and self.recommendations:
            raise ValueError("A no_results answer cannot contain recommendations")
        return self


WorkflowStatus = Literal[
    "new",
    "planned",
    "clarification_required",
    "retrieved",
    "criticized",
    "completed",
    "failed",
]


class ConversationState(LLMContractModel):
    """Application-owned state shared selectively with the logical agents.

    Providers remain stateless from the workflow's perspective.  This object is
    the canonical, serializable trace of one user turn and enables replay,
    evaluation and later multi-agent extensions.
    """

    contract_version: Literal["movie-llm/v1"] = LLM_CONTRACT_VERSION
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    original_query: str = Field(min_length=1, max_length=4000)
    catalog_language: str
    query_rewrite_strategy: QueryRewriteStrategy
    status: WorkflowStatus = "new"
    plan: RetrievalPlan | None = None
    search_history: list[SearchMoviesResponse] = Field(default_factory=list)
    details_response: MovieDetailsResponse | None = None
    critique: CriticDecision | None = None
    final_answer: GroundedAnswer | None = None
    retry_count: int = Field(default=0, ge=0)
    llm_traces: list[LLMCallTrace] = Field(default_factory=list)
    tool_traces: list[ToolCallTrace] = Field(default_factory=list)
    error: str | None = None

    @property
    def latest_search(self) -> SearchMoviesResponse | None:
        return self.search_history[-1] if self.search_history else None

    @property
    def user_visible_text(self) -> str | None:
        if self.status == "clarification_required":
            if self.critique and self.critique.clarification_question:
                return self.critique.clarification_question
            if self.plan:
                return self.plan.clarification_question
        return None if self.final_answer is None else self.final_answer.answer
