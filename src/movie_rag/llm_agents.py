"""Logical LLM agents with provider-neutral, validated inputs and outputs.

An agent is intentionally small: it combines one versioned system prompt, one
provider profile and one Pydantic output contract.  It does not own workflow
state and cannot invoke application tools.  The deterministic orchestrator in
``llm_workflow.py`` remains responsible for transitions and MCP calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generic, Mapping, TypeVar

from pydantic import BaseModel

from .llm_config import (
    AgentProfile,
    LLMConfigCatalog,
    WorkflowProfile,
    load_prompt_resource,
)
from .llm_contract import (
    ConversationState,
    CriticDecision,
    GroundedAnswer,
    LLMCallTrace,
    RetrievalPlan,
)
from .llm_providers import LLMProvider, LLMProviderRegistry


T = TypeVar("T", bound=BaseModel)


class AgentConfigurationError(ValueError):
    """Raised when an agent/profile binding is incomplete or inconsistent."""


class AgentOutputError(ValueError):
    """Raised when valid JSON still violates an application grounding rule."""


def _json_prompt(payload: Mapping[str, Any]) -> str:
    """Serialize a bounded application context without SDK-specific messages."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        default=str,
    )


class StructuredAgent(Generic[T]):
    """Base class for one structured logical role.

    Parameters are fixed at construction time.  A later workflow can therefore
    use a different model for each role without changing prompts or call sites.
    """

    output_schema: type[T]

    def __init__(
        self,
        *,
        profile: AgentProfile,
        provider: LLMProvider,
        output_schema: type[T],
    ) -> None:
        if not profile.enabled:
            raise AgentConfigurationError(
                f"Agent {profile.agent_id!r} is disabled in the selected catalog"
            )
        self.profile = profile
        self.provider = provider
        self.output_schema = output_schema
        self.system_prompt = load_prompt_resource(profile.prompt_resource)

    async def _generate(self, payload: Mapping[str, Any]) -> tuple[T, LLMCallTrace]:
        generation = await self.provider.generate_structured(
            agent=self.profile.agent_id,
            prompt_version=self.profile.prompt_version,
            system_prompt=self.system_prompt,
            user_prompt=_json_prompt(payload),
            output_schema=self.output_schema,
            temperature=self.profile.temperature,
            max_output_tokens=self.profile.max_output_tokens,
        )
        return generation.value, generation.trace


class PlannerAgent(StructuredAgent[RetrievalPlan]):
    """Extract hard filters and a semantic query, or ask one clarification."""

    def __init__(self, *, profile: AgentProfile, provider: LLMProvider) -> None:
        super().__init__(
            profile=profile,
            provider=provider,
            output_schema=RetrievalPlan,
        )

    async def run(
        self,
        state: ConversationState,
        workflow: WorkflowProfile,
    ) -> tuple[RetrievalPlan, LLMCallTrace]:
        plan, trace = await self._generate(
            {
                "task": "plan_movie_search",
                "user_request": state.original_query,
                "catalog": {
                    "catalog_language": workflow.catalog_language,
                    "supported_hard_filters": [
                        "min_runtime",
                        "max_runtime",
                        "min_year",
                        "max_year",
                        "languages",
                        "countries",
                        "include_genres",
                        "exclude_genres",
                        "adult",
                    ],
                },
                "policy": {
                    "query_rewrite_strategy": workflow.query_rewrite_strategy,
                    "clarify_only_when_search_is_materially_blocked": True,
                },
            }
        )
        if plan.rewrite_strategy != workflow.query_rewrite_strategy:
            raise AgentOutputError(
                "Planner changed the application-owned query rewrite strategy: "
                f"expected {workflow.query_rewrite_strategy!r}, got "
                f"{plan.rewrite_strategy!r}"
            )
        return plan, trace


class CriticAgent(StructuredAgent[CriticDecision]):
    """Optionally inspect a retrieval result without changing hard filters."""

    def __init__(self, *, profile: AgentProfile, provider: LLMProvider) -> None:
        super().__init__(
            profile=profile,
            provider=provider,
            output_schema=CriticDecision,
        )

    async def run(
        self,
        state: ConversationState,
    ) -> tuple[CriticDecision, LLMCallTrace]:
        if state.plan is None or state.latest_search is None:
            raise AgentOutputError("Critic requires both a plan and search results")
        decision, trace = await self._generate(
            {
                "task": "review_retrieval",
                "user_request": state.original_query,
                "plan": state.plan.model_dump(mode="json"),
                "search_response": state.latest_search.model_dump(mode="json"),
                "retry_policy": {
                    "retry_count": state.retry_count,
                    "hard_filters_are_immutable": True,
                },
            }
        )
        return decision, trace


class AnswerAgent(StructuredAgent[GroundedAnswer]):
    """Compose a response constrained to IDs and titles returned by MCP."""

    def __init__(self, *, profile: AgentProfile, provider: LLMProvider) -> None:
        super().__init__(
            profile=profile,
            provider=provider,
            output_schema=GroundedAnswer,
        )

    async def run(
        self,
        state: ConversationState,
    ) -> tuple[GroundedAnswer, LLMCallTrace]:
        if state.plan is None or state.latest_search is None:
            raise AgentOutputError("Answer agent requires a plan and search response")
        answer, trace = await self._generate(
            {
                "task": "write_grounded_movie_answer",
                "user_request": state.original_query,
                "plan": state.plan.model_dump(mode="json"),
                "search_response": state.latest_search.model_dump(mode="json"),
                "movie_details": (
                    None
                    if state.details_response is None
                    else state.details_response.model_dump(mode="json")
                ),
                "critic_decision": (
                    None
                    if state.critique is None
                    else state.critique.model_dump(mode="json")
                ),
            }
        )
        self._validate_grounding(answer, state)
        return answer, trace

    @staticmethod
    def _validate_grounding(
        answer: GroundedAnswer,
        state: ConversationState,
    ) -> None:
        search = state.latest_search
        if search is None:  # pragma: no cover - guarded by run
            raise AgentOutputError("Missing search response")
        if search.status == "no_results":
            if answer.status != "no_results" or answer.recommendations:
                raise AgentOutputError(
                    "Answer must be no_results when MCP returned no candidates"
                )
            return

        allowed = {result.movie_id: result.title for result in search.results}
        seen: set[int] = set()
        for recommendation in answer.recommendations:
            expected_title = allowed.get(recommendation.movie_id)
            if expected_title is None:
                raise AgentOutputError(
                    "Answer recommended a movie_id absent from MCP results: "
                    f"{recommendation.movie_id}"
                )
            if recommendation.movie_id in seen:
                raise AgentOutputError(
                    f"Duplicate recommendation for movie_id {recommendation.movie_id}"
                )
            if recommendation.title.casefold() != expected_title.casefold():
                raise AgentOutputError(
                    "Answer changed the catalog title for movie_id "
                    f"{recommendation.movie_id}: expected {expected_title!r}, "
                    f"got {recommendation.title!r}"
                )
            seen.add(recommendation.movie_id)


@dataclass(frozen=True)
class AgentSet:
    """The logical roles bound to one workflow execution."""

    planner: PlannerAgent
    answer: AnswerAgent
    critic: CriticAgent | None = None


def _agent_with_profile_override(
    catalog: LLMConfigCatalog,
    agent_id: str,
    role: str,
    overrides: Mapping[str, str],
) -> AgentProfile:
    profile = catalog.get_agent(agent_id)
    override = overrides.get(role)
    if override is None:
        return profile
    catalog.get_llm_profile(override)
    return profile.model_copy(update={"llm_profile": override})


def build_agent_set(
    *,
    catalog: LLMConfigCatalog,
    workflow: WorkflowProfile,
    provider_registry: LLMProviderRegistry,
    role_profile_overrides: Mapping[str, str] | None = None,
) -> AgentSet:
    """Bind configured prompts to providers for planner, answer and critic.

    ``role_profile_overrides`` selects only predeclared LLM profiles.  It never
    accepts an arbitrary model or credential from a user request.
    """

    overrides = dict(role_profile_overrides or {})
    unknown_roles = sorted(set(overrides) - {"planner", "critic", "answer"})
    if unknown_roles:
        raise AgentConfigurationError(
            f"Unknown role profile overrides: {unknown_roles}"
        )

    planner_profile = _agent_with_profile_override(
        catalog, workflow.planner_agent, "planner", overrides
    )
    answer_profile = _agent_with_profile_override(
        catalog, workflow.answer_agent, "answer", overrides
    )
    planner = PlannerAgent(
        profile=planner_profile,
        provider=provider_registry.get(planner_profile.llm_profile),
    )
    answer = AnswerAgent(
        profile=answer_profile,
        provider=provider_registry.get(answer_profile.llm_profile),
    )

    critic: CriticAgent | None = None
    if workflow.enable_critic:
        if workflow.critic_agent is None:  # defensive; config validation rejects it
            raise AgentConfigurationError("Critic is enabled but not configured")
        critic_profile = _agent_with_profile_override(
            catalog, workflow.critic_agent, "critic", overrides
        )
        critic = CriticAgent(
            profile=critic_profile,
            provider=provider_registry.get(critic_profile.llm_profile),
        )
    return AgentSet(planner=planner, answer=answer, critic=critic)
