"""Deterministic orchestration for the provider-neutral Movie RAG agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .llm_agents import AgentSet, build_agent_set
from .llm_config import (
    LLMConfigCatalog,
    WorkflowProfile,
    load_llm_config,
    role_profile_overrides_from_environment,
    selected_workflow_id,
)
from .llm_contract import ConversationState, QueryRewriteStrategy
from .llm_mcp import MovieToolGateway, ToolGatewayError
from .llm_providers import LLMProviderRegistry


class WorkflowExecutionError(RuntimeError):
    """A failed turn together with its serializable partial state."""

    def __init__(self, state: ConversationState) -> None:
        super().__init__(state.error or "Movie assistant workflow failed")
        self.state = state


@dataclass
class MovieAssistantWorkflow:
    """Application-owned state machine for one grounded recommendation turn.

    Initial flow::

        Planner -> search_movies -> [Critic/retry] -> get_movie_details -> Answer

    LLMs propose structured values.  Python owns every branch, retry bound and
    tool call.  In particular, a critic retry may replace only the semantic
    query; the validated hard filters are reused verbatim.
    """

    agents: AgentSet
    gateway: MovieToolGateway
    profile: WorkflowProfile

    async def run(self, user_query: str) -> ConversationState:
        cleaned_query = str(user_query).strip()
        if not cleaned_query:
            raise ValueError("user_query cannot be empty")
        state = ConversationState(
            workflow_id=self.profile.workflow_id,
            original_query=cleaned_query,
            catalog_language=self.profile.catalog_language,
            query_rewrite_strategy=self.profile.query_rewrite_strategy,
        )
        try:
            plan, planner_trace = await self.agents.planner.run(state, self.profile)
            state.plan = plan
            state.llm_traces.append(planner_trace)
            state.status = "planned"

            if plan.action == "clarify":
                state.status = "clarification_required"
                return state

            await self._search(state, plan.semantic_query or "")

            if self.profile.enable_critic:
                await self._run_critic_loop(state)
                if state.status == "clarification_required":
                    return state

            search = state.latest_search
            if search is None:  # pragma: no cover - _search guarantees it
                raise RuntimeError("Search completed without a response")
            if search.results:
                detail_ids = [
                    result.movie_id
                    for result in search.results[: self.profile.details_limit]
                ]
                details = await self.gateway.get_movie_details(movie_ids=detail_ids)
                state.details_response = details.value
                state.tool_traces.append(details.trace)

            answer, answer_trace = await self.agents.answer.run(state)
            state.final_answer = answer
            state.llm_traces.append(answer_trace)
            state.status = "completed"
            return state
        except Exception as exc:
            if isinstance(exc, ToolGatewayError):
                state.tool_traces.append(exc.trace)
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            raise WorkflowExecutionError(state) from exc

    async def _search(self, state: ConversationState, semantic_query: str) -> None:
        if state.plan is None:  # pragma: no cover - guarded by run
            raise RuntimeError("Cannot search without a plan")
        execution = await self.gateway.search_movies(
            query=semantic_query,
            filters=state.plan.filters,
            limit=self.profile.search_limit,
        )
        state.search_history.append(execution.value)
        state.tool_traces.append(execution.trace)
        state.status = "retrieved"

    async def _run_critic_loop(self, state: ConversationState) -> None:
        critic = self.agents.critic
        if critic is None:
            raise RuntimeError("Workflow enables a critic but AgentSet has none")

        while True:
            decision, trace = await critic.run(state)
            state.critique = decision
            state.llm_traces.append(trace)
            state.status = "criticized"

            if decision.decision == "accept":
                return
            if decision.decision == "clarify":
                state.status = "clarification_required"
                return
            if state.retry_count >= self.profile.max_retries:
                # The bounded state machine proceeds with the latest grounded
                # candidates; it never opens an unbounded LLM retry loop.
                return

            state.retry_count += 1
            await self._search(state, decision.suggested_query or "")


def build_movie_assistant_workflow(
    *,
    gateway: MovieToolGateway,
    catalog: LLMConfigCatalog | None = None,
    config_path: str | Path | None = None,
    workflow_id: str | None = None,
    provider_registry: LLMProviderRegistry | None = None,
    role_profile_overrides: Mapping[str, str] | None = None,
    enable_critic: bool | None = None,
    query_rewrite_strategy: QueryRewriteStrategy | None = None,
) -> MovieAssistantWorkflow:
    """Build a workflow from versioned config plus bounded local overrides.

    Provider/model selection occurs here, not in the user prompt or public MCP
    tool.  Passing ``provider_registry`` is useful for tests and custom clients.
    """

    resolved_catalog = catalog or load_llm_config(config_path)
    selected = selected_workflow_id(workflow_id)
    profile = resolved_catalog.get_workflow(selected)
    updates: dict[str, object] = {}
    if enable_critic is not None:
        updates["enable_critic"] = enable_critic
    if query_rewrite_strategy is not None:
        updates["query_rewrite_strategy"] = query_rewrite_strategy
    if updates:
        profile = WorkflowProfile.model_validate(
            {**profile.model_dump(), **updates}
        )

    registry = provider_registry or LLMProviderRegistry(resolved_catalog)
    environment_overrides = role_profile_overrides_from_environment()
    explicit_overrides = dict(role_profile_overrides or {})
    overrides = {**environment_overrides, **explicit_overrides}
    agents = build_agent_set(
        catalog=resolved_catalog,
        workflow=profile,
        provider_registry=registry,
        role_profile_overrides=overrides,
    )
    return MovieAssistantWorkflow(
        agents=agents,
        gateway=gateway,
        profile=profile,
    )
