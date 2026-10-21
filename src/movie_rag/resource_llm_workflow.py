from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .resource_domain import ResourceDomainProfile
from .resource_llm_agents import ResourceAnswerAgent, ResourcePlannerAgent
from .resource_llm_models import (
    GroundedAnswer,
    GroundedRecommendation,
    ResourceConversationState,
)
from .resource_mcp_gateway import MCPResourceToolGateway


class ResourceGateway(Protocol):
    async def search_resources(self, *, query: str, filters=None, limit: int = 10): ...


@dataclass(frozen=True)
class ResourceWorkflowPolicy:
    workflow_id: str = "grounded_resource_v1"
    result_limit: int = 10
    sanitize_grounding: bool = True


class ResourceAssistantWorkflow:
    """Minimal fixed baseline: Planner -> MCP retrieval -> grounded Answer.

    This is intentionally not the final dynamic multi-agent orchestrator.  It is the
    stable baseline we can later compare against a conditional orchestrator.
    """

    def __init__(
        self,
        *,
        domain: ResourceDomainProfile,
        planner: ResourcePlannerAgent,
        answer: ResourceAnswerAgent,
        gateway: ResourceGateway,
        policy: ResourceWorkflowPolicy | None = None,
    ) -> None:
        self.domain = domain
        self.planner = planner
        self.answer = answer
        self.gateway = gateway
        self.policy = policy or ResourceWorkflowPolicy()

    async def run(self, request: str) -> ResourceConversationState:
        state = ResourceConversationState(
            workflow_id=self.policy.workflow_id,
            domain_id=self.domain.domain_id,
            original_query=str(request),
        )

        plan, planner_trace = await self.planner.run(request)
        state.plan = plan
        state.llm_traces.append(planner_trace)

        if plan.clarification_required:
            state.status = "needs_clarification"
            state.final_answer = GroundedAnswer(
                status="needs_clarification",
                answer=plan.clarification_question or "Please clarify your request.",
                recommendations=[],
            )
            return state

        search, tool_trace = await self.gateway.search_resources(
            query=plan.semantic_query,
            filters=plan.filters,
            limit=self.policy.result_limit,
        )
        state.search_response = search
        state.tool_traces.append(tool_trace)
        state.status = "searched"

        if search.status == "no_results" or not search.results:
            state.status = "no_results"
            state.final_answer = GroundedAnswer(
                status="no_results",
                answer=(
                    "No grounded resources matched the current request and constraints. "
                    "Try reformulating the request or explicitly relaxing a constraint."
                ),
                recommendations=[],
            )
            return state

        answer, answer_trace = await self.answer.run(request, search)
        state.llm_traces.append(answer_trace)
        if self.policy.sanitize_grounding:
            answer, violations = _sanitize_answer(answer, search.results)
            state.grounding_violations.extend(violations)
        state.final_answer = answer
        state.status = "completed"
        return state


def _sanitize_answer(answer: GroundedAnswer, hits):
    """Guarantee that final recommendation IDs/titles come from MCP retrieval.

    Violations are retained in state so evaluation can still measure LLM grounding
    errors rather than silently losing the evidence that they occurred.
    """

    by_id = {str(hit.resource_id): hit for hit in hits}
    cleaned: list[GroundedRecommendation] = []
    violations: list[str] = []
    seen: set[str] = set()

    for rec in answer.recommendations:
        key = str(rec.resource_id)
        hit = by_id.get(key)
        if hit is None:
            violations.append(f"recommendation outside retrieved set: {rec.resource_id}")
            continue
        if key in seen:
            violations.append(f"duplicate recommendation id: {rec.resource_id}")
            continue
        seen.add(key)
        if rec.title.strip() != hit.title.strip():
            violations.append(
                f"title corrected for {rec.resource_id}: {rec.title!r} -> {hit.title!r}"
            )
        cleaned.append(
            GroundedRecommendation(
                resource_id=hit.resource_id,
                title=hit.title,
                reason=rec.reason,
            )
        )

    if answer.status == "ok" and not cleaned:
        violations.append("answer status was ok but no valid grounded recommendations remained")

    status = "ok" if cleaned else answer.status
    return answer.model_copy(update={"status": status, "recommendations": cleaned}), violations


def build_resource_assistant_workflow(
    *,
    domain: ResourceDomainProfile,
    planner_provider,
    answer_provider,
    gateway: MCPResourceToolGateway,
    workflow_id: str = "grounded_resource_v1",
    result_limit: int = 10,
    planner_temperature: float = 0.0,
    answer_temperature: float = 0.2,
    query_rewrite_strategy: str = "none",
) -> ResourceAssistantWorkflow:
    return ResourceAssistantWorkflow(
        domain=domain,
        planner=ResourcePlannerAgent(
            provider=planner_provider,
            domain=domain,
            temperature=planner_temperature,
            query_rewrite_strategy=query_rewrite_strategy,
        ),
        answer=ResourceAnswerAgent(
            provider=answer_provider,
            domain=domain,
            temperature=answer_temperature,
        ),
        gateway=gateway,
        policy=ResourceWorkflowPolicy(
            workflow_id=workflow_id,
            result_limit=result_limit,
        ),
    )


__all__ = [
    "ResourceWorkflowPolicy",
    "ResourceAssistantWorkflow",
    "build_resource_assistant_workflow",
]
