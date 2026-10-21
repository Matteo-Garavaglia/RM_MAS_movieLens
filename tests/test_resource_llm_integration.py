from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from movie_rag.resource_domain import ResourceDomainProfile
from movie_rag.resource_llm_agents import ResourceAnswerAgent, ResourcePlannerAgent
from movie_rag.resource_llm_workflow import ResourceAssistantWorkflow
from movie_rag.resource_mcp_gateway import MCPToolTrace
from movie_rag.resource_models import (
    PipelineInfo,
    ResourceFilters,
    ResourceSearchHit,
    ResourceSearchResponse,
)


def movie_domain() -> ResourceDomainProfile:
    return ResourceDomainProfile(
        domain_id="movies",
        display_name="movies",
        resource_label="movie",
        user_role="viewer",
        task_description="find relevant movies",
        catalog_description="movie catalog",
        allowed_filter_keys=["max_runtime", "exclude_genres"],
    )


class FakeProvider:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        trace = SimpleNamespace(model_dump=lambda mode="json": {"provider": "fake"})
        return SimpleNamespace(text=json.dumps(payload), trace=trace)


@pytest.mark.asyncio
async def test_planner_keeps_technical_retrieval_outside_llm_contract():
    provider = FakeProvider([
        {
            "semantic_query": "melancholic film about memory",
            "filters": {"max_runtime": 120},
            "clarification_required": False,
            "clarification_question": None,
            "rationale": "runtime was explicit",
        }
    ])
    planner = ResourcePlannerAgent(provider, movie_domain())
    plan, trace = await planner.run("a melancholic film under two hours")
    assert plan.semantic_query == "melancholic film about memory"
    assert plan.filters == {"max_runtime": 120}
    system = provider.calls[0]["system_prompt"]
    assert "do not choose an embedding model" in system.lower()
    assert trace.role == "planner"


@pytest.mark.asyncio
async def test_planner_rejects_domain_filter_not_in_profile():
    provider = FakeProvider([
        {
            "semantic_query": "x",
            "filters": {"made_up_filter": "y"},
            "clarification_required": False,
            "clarification_question": None,
            "rationale": "",
        }
    ])
    planner = ResourcePlannerAgent(provider, movie_domain())
    with pytest.raises(ValueError, match="unsupported filters"):
        await planner.run("x")


class FakeGateway:
    async def search_resources(self, *, query, filters=None, limit=10):
        response = ResourceSearchResponse(
            status="ok",
            query=query,
            filters=ResourceFilters(),
            pipeline=PipelineInfo(
                profile_id="fast",
                retriever="fake",
                candidate_k=10,
                final_k=10,
                rerank_enabled=False,
            ),
            returned_count=2,
            results=[
                ResourceSearchHit(
                    rank=1,
                    resource_id=1,
                    title="One",
                    text="memory story",
                    metadata={},
                    score=0.9,
                ),
                ResourceSearchHit(
                    rank=2,
                    resource_id=2,
                    title="Two",
                    text="identity story",
                    metadata={},
                    score=0.8,
                ),
            ],
        )
        return response, MCPToolTrace(
            tool="search_resources",
            latency_ms=1.0,
            arguments={"query": query, "filters": filters or {}, "limit": limit},
        )


@pytest.mark.asyncio
async def test_workflow_sanitizes_hallucinated_recommendation_ids():
    planner_provider = FakeProvider([
        {
            "semantic_query": "memory and identity",
            "filters": {},
            "clarification_required": False,
            "clarification_question": None,
            "rationale": "",
        }
    ])
    answer_provider = FakeProvider([
        {
            "status": "ok",
            "answer": "Try these.",
            "recommendations": [
                {"resource_id": 1, "title": "WRONG TITLE", "reason": "relevant"},
                {"resource_id": 999, "title": "Hallucinated", "reason": "not grounded"},
            ],
        }
    ])
    domain = movie_domain()
    workflow = ResourceAssistantWorkflow(
        domain=domain,
        planner=ResourcePlannerAgent(planner_provider, domain),
        answer=ResourceAnswerAgent(answer_provider, domain),
        gateway=FakeGateway(),
    )
    state = await workflow.run("something about memory")
    assert state.status == "completed"
    assert [r.resource_id for r in state.final_answer.recommendations] == [1]
    assert state.final_answer.recommendations[0].title == "One"
    assert any("outside retrieved set" in x for x in state.grounding_violations)
    assert any("title corrected" in x for x in state.grounding_violations)


@pytest.mark.asyncio
async def test_clarification_stops_before_retrieval():
    provider = FakeProvider([
        {
            "semantic_query": "",
            "filters": {},
            "clarification_required": True,
            "clarification_question": "What topic do you mean?",
            "rationale": "missing topic",
        }
    ])
    domain = movie_domain()

    class NeverGateway:
        async def search_resources(self, **kwargs):
            raise AssertionError("retrieval must not be called")

    workflow = ResourceAssistantWorkflow(
        domain=domain,
        planner=ResourcePlannerAgent(provider, domain),
        answer=ResourceAnswerAgent(provider, domain),
        gateway=NeverGateway(),
    )
    state = await workflow.run("help")
    assert state.status == "needs_clarification"
    assert state.final_answer.status == "needs_clarification"


def test_domain_config_and_mcp_profile_config_parse():
    from pathlib import Path
    from movie_rag.resource_domain import load_resource_domain_catalog
    from movie_rag.resource_mcp_profiles import load_resource_mcp_profiles

    root = Path(__file__).resolve().parents[1]
    domains = load_resource_domain_catalog(root / "configs" / "resource_domains_v1.json")
    assert domains.get("movielens_v2").template_only is False
    assert domains.get("whoteach_template").template_only is True

    profiles = load_resource_mcp_profiles(root / "configs" / "resource_mcp_profiles_v1.json")
    profile = profiles.get("movielens_gte_fast_v2")
    assert profile.encoder.model_name == "Alibaba-NLP/gte-multilingual-base"
    assert profile.search_profile_id == "fast"
    assert profile.build_search_profiles()["quality"].candidate_k == 20
