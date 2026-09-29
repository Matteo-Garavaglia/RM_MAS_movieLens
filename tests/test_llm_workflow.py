from __future__ import annotations

from dataclasses import dataclass

import pytest

from movie_rag.llm_agents import AgentSet, AnswerAgent, CriticAgent, PlannerAgent
from movie_rag.llm_config import load_llm_config
from movie_rag.llm_contract import (
    CriticDecision,
    GroundedAnswer,
    GroundedRecommendation,
    LLMCallTrace,
    LLMUsage,
    RetrievalPlan,
    ToolCallTrace,
)
from movie_rag.llm_mcp import ToolExecution
from movie_rag.llm_providers import StructuredGeneration
from movie_rag.llm_workflow import MovieAssistantWorkflow, WorkflowExecutionError
from movie_rag.mcp_contract import (
    MovieDetails,
    MovieDetailsResponse,
    MovieFilterInput,
    MovieSearchResult,
    RetrievalPipelineInfo,
    RetrievalScores,
    SearchMoviesResponse,
)
from movie_rag.mcp_profiles import load_profile_catalog


class QueueProvider:
    def __init__(self, profile, outputs):
        self.profile = profile
        self.outputs = list(outputs)
        self.calls = []

    async def generate_structured(self, *, output_schema, agent, **kwargs):
        self.calls.append({"agent": agent, "schema": output_schema, **kwargs})
        output = self.outputs.pop(0)
        value = output_schema.model_validate(output)
        return StructuredGeneration(
            value=value,
            trace=LLMCallTrace(
                agent=agent,
                provider=self.profile.provider,
                profile_id=self.profile.profile_id,
                model=self.profile.model,
                prompt_version=kwargs["prompt_version"],
                duration_ms=1.0,
                usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )


def plan(**updates):
    payload = {
        "action": "search",
        "detected_language": "it",
        "semantic_query": "film malinconico sulla memoria",
        "semantic_query_language": "it",
        "rewrite_strategy": "none",
        "filters": {"max_runtime": 120},
        "ambiguities": [],
        "clarification_question": None,
        "reason": "Richiesta ricercabile.",
    }
    payload.update(updates)
    return payload


def answer(movie_id=1, title="Memory Film (1999)"):
    return GroundedAnswer(
        status="ok",
        response_language="it",
        answer="Ti consiglio Memory Film.",
        recommendations=[
            GroundedRecommendation(
                movie_id=movie_id,
                title=title,
                explanation="Affronta memoria e identità.",
            )
        ],
    )


def pipeline_info():
    return RetrievalPipelineInfo.from_profile(
        load_profile_catalog().get("dense_v1")
    )


def search_response(query, filters, status="ok"):
    results = []
    if status == "ok":
        results = [
            MovieSearchResult(
                rank=1,
                dense_rank=1,
                movie_id=1,
                title="Memory Film (1999)",
                overview="A story about memory and identity.",
                year=1999,
                runtime_minutes=110,
                genres=["Drama"],
                original_language="en",
                countries=["US"],
                scores=RetrievalScores(final=0.8, dense=0.8),
            )
        ]
    return SearchMoviesResponse(
        status=status,
        query=query,
        applied_filters=filters,
        requested_limit=10,
        returned_count=len(results),
        pipeline=pipeline_info(),
        results=results,
        message="No results" if status == "no_results" else None,
    )


class FakeGateway:
    def __init__(self, no_results=False):
        self.no_results = no_results
        self.search_calls = []
        self.detail_calls = []

    async def search_movies(self, *, query, filters, limit):
        self.search_calls.append(
            {"query": query, "filters": filters.model_dump(), "limit": limit}
        )
        response = search_response(
            query,
            filters,
            status="no_results" if self.no_results else "ok",
        )
        return ToolExecution(
            value=response,
            trace=ToolCallTrace(
                tool="search_movies",
                duration_ms=2.0,
                status=response.status,
                returned_count=response.returned_count,
            ),
        )

    async def get_movie_details(self, *, movie_ids):
        self.detail_calls.append(movie_ids)
        response = MovieDetailsResponse(
            requested_movie_ids=movie_ids,
            returned_count=1,
            missing_movie_ids=[],
            movies=[
                MovieDetails(
                    movie_id=1,
                    title="Memory Film (1999)",
                    overview="A story about memory and identity.",
                )
            ],
        )
        return ToolExecution(
            value=response,
            trace=ToolCallTrace(
                tool="get_movie_details",
                duration_ms=1.0,
                status="ok",
                returned_count=1,
            ),
        )


@dataclass
class BuiltAgents:
    agents: AgentSet
    planner_provider: QueueProvider
    answer_provider: QueueProvider
    critic_provider: QueueProvider | None


def agents(planner_outputs, answer_outputs, critic_outputs=None):
    catalog = load_llm_config()
    llm_profile = catalog.get_llm_profile("gemini_free")
    planner_provider = QueueProvider(llm_profile, planner_outputs)
    answer_provider = QueueProvider(llm_profile, answer_outputs)
    critic_provider = (
        None if critic_outputs is None else QueueProvider(llm_profile, critic_outputs)
    )
    return BuiltAgents(
        agents=AgentSet(
            planner=PlannerAgent(
                profile=catalog.get_agent("planner"), provider=planner_provider
            ),
            answer=AnswerAgent(
                profile=catalog.get_agent("answer"), provider=answer_provider
            ),
            critic=(
                None
                if critic_provider is None
                else CriticAgent(
                    profile=catalog.get_agent("critic"), provider=critic_provider
                )
            ),
        ),
        planner_provider=planner_provider,
        answer_provider=answer_provider,
        critic_provider=critic_provider,
    )


@pytest.mark.anyio
async def test_grounded_flow_calls_mcp_and_preserves_traces():
    built = agents([plan()], [answer()])
    gateway = FakeGateway()
    profile = load_llm_config().get_workflow()
    workflow = MovieAssistantWorkflow(built.agents, gateway, profile)

    state = await workflow.run("Vorrei un film malinconico sulla memoria")

    assert state.status == "completed"
    assert state.final_answer.recommendations[0].movie_id == 1
    assert gateway.search_calls[0]["filters"]["max_runtime"] == 120.0
    assert gateway.detail_calls == [[1]]
    assert [trace.agent for trace in state.llm_traces] == ["planner", "answer"]
    assert [trace.tool for trace in state.tool_traces] == [
        "search_movies",
        "get_movie_details",
    ]


@pytest.mark.anyio
async def test_critic_retry_changes_only_query_and_is_bounded():
    decisions = [
        CriticDecision(
            decision="retry",
            reason="Query troppo generica.",
            suggested_query="memoria identità e perdita",
        ),
        CriticDecision(decision="accept", reason="Risultati plausibili."),
    ]
    built = agents([plan()], [answer()], decisions)
    gateway = FakeGateway()
    profile = load_llm_config().get_workflow().model_copy(
        update={"enable_critic": True, "max_retries": 1}
    )
    workflow = MovieAssistantWorkflow(built.agents, gateway, profile)

    state = await workflow.run("Un film sulla memoria sotto le due ore")

    assert state.retry_count == 1
    assert [call["query"] for call in gateway.search_calls] == [
        "film malinconico sulla memoria",
        "memoria identità e perdita",
    ]
    assert gateway.search_calls[0]["filters"] == gateway.search_calls[1]["filters"]
    assert len(state.search_history) == 2
    assert state.critique.decision == "accept"


@pytest.mark.anyio
async def test_planner_clarification_stops_before_tools():
    clarify = plan(
        action="clarify",
        semantic_query=None,
        semantic_query_language=None,
        clarification_question="Che tipo di atmosfera preferisci?",
        reason="La richiesta è troppo generica.",
    )
    built = agents([clarify], [])
    gateway = FakeGateway()
    workflow = MovieAssistantWorkflow(
        built.agents, gateway, load_llm_config().get_workflow()
    )

    state = await workflow.run("Consigliami qualcosa")

    assert state.status == "clarification_required"
    assert state.user_visible_text == "Che tipo di atmosfera preferisci?"
    assert gateway.search_calls == []
    assert built.answer_provider.calls == []


@pytest.mark.anyio
async def test_no_results_is_answered_without_detail_lookup():
    no_result_answer = GroundedAnswer(
        status="no_results",
        response_language="it",
        answer="Il catalogo non contiene film che rispettino tutti i vincoli.",
        recommendations=[],
        limitations=["Puoi scegliere esplicitamente di rilassare un vincolo."],
    )
    built = agents([plan()], [no_result_answer])
    gateway = FakeGateway(no_results=True)
    workflow = MovieAssistantWorkflow(
        built.agents, gateway, load_llm_config().get_workflow()
    )

    state = await workflow.run("Film italiano sotto le due ore")

    assert state.status == "completed"
    assert state.final_answer.status == "no_results"
    assert gateway.detail_calls == []


@pytest.mark.anyio
async def test_answer_cannot_invent_a_movie_outside_mcp_results():
    built = agents([plan()], [answer(movie_id=999, title="Invented Film")])
    workflow = MovieAssistantWorkflow(
        built.agents, FakeGateway(), load_llm_config().get_workflow()
    )

    with pytest.raises(WorkflowExecutionError) as caught:
        await workflow.run("Un film sulla memoria")

    assert caught.value.state.status == "failed"
    assert "absent from MCP results" in caught.value.state.error
