from __future__ import annotations

from types import SimpleNamespace

import pytest


mcp = pytest.importorskip("mcp")

from mcp import Client

from movie_rag.llm_config import load_llm_config
from movie_rag.llm_contract import (
    GroundedAnswer,
    GroundedRecommendation,
    LLMCallTrace,
    RetrievalPlan,
)
from movie_rag.llm_mcp import MCPMovieToolGateway
from movie_rag.llm_providers import StructuredGeneration
from movie_rag.llm_workflow import build_movie_assistant_workflow
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
from movie_rag.mcp_server import create_mcp_server


class SchemaDrivenProvider:
    def __init__(self, profile):
        self.profile = profile

    async def generate_structured(self, *, output_schema, agent, **kwargs):
        if output_schema is RetrievalPlan:
            value = RetrievalPlan(
                action="search",
                detected_language="it",
                semantic_query="memoria e identità",
                semantic_query_language="it",
                rewrite_strategy="none",
                filters=MovieFilterInput(max_runtime=120),
                reason="Richiesta ricercabile.",
            )
        elif output_schema is GroundedAnswer:
            value = GroundedAnswer(
                status="ok",
                response_language="it",
                answer="Ti consiglio Memory Film.",
                recommendations=[
                    GroundedRecommendation(
                        movie_id=1,
                        title="Memory Film (1999)",
                        explanation="Il catalogo lo descrive come una storia sulla memoria.",
                    )
                ],
            )
        else:  # pragma: no cover - critic is disabled in this test
            raise AssertionError(output_schema)
        return StructuredGeneration(
            value=value,
            trace=LLMCallTrace(
                agent=agent,
                provider="gemini",
                profile_id=self.profile.profile_id,
                model=self.profile.model,
                prompt_version=kwargs["prompt_version"],
                duration_ms=1,
            ),
        )


class ProviderRegistry:
    def __init__(self, provider):
        self.provider = provider

    def get(self, profile_id):
        assert profile_id == self.provider.profile.profile_id
        return self.provider


class FakeMovieService:
    def __init__(self):
        self.search_calls = []

    def search_movies(self, query, filters=None, limit=10):
        filters = MovieFilterInput.model_validate(filters or {})
        self.search_calls.append((query, filters, limit))
        return SearchMoviesResponse(
            status="ok",
            query=query,
            applied_filters=filters,
            requested_limit=limit,
            returned_count=1,
            pipeline=RetrievalPipelineInfo.from_profile(
                load_profile_catalog().get("dense_v1")
            ),
            results=[
                MovieSearchResult(
                    rank=1,
                    dense_rank=1,
                    movie_id=1,
                    title="Memory Film (1999)",
                    overview="A story about memory and identity.",
                    runtime_minutes=110,
                    scores=RetrievalScores(final=0.8, dense=0.8),
                )
            ],
        )

    def get_movie_details(self, movie_ids):
        return MovieDetailsResponse(
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


@pytest.mark.anyio
async def test_provider_to_orchestrator_to_real_mcp_roundtrip():
    service = FakeMovieService()
    registry = SimpleNamespace(get=lambda profile_id=None: service)
    server = create_mcp_server(registry, enable_experimental=False)
    catalog = load_llm_config()
    provider = SchemaDrivenProvider(catalog.get_llm_profile("gemini_free"))

    async with Client(server, raise_exceptions=True) as client:
        workflow = build_movie_assistant_workflow(
            gateway=MCPMovieToolGateway(client),
            catalog=catalog,
            provider_registry=ProviderRegistry(provider),
        )
        state = await workflow.run("Un film sulla memoria sotto le due ore")

    assert state.status == "completed"
    assert state.final_answer.recommendations[0].movie_id == 1
    assert service.search_calls[0][0] == "memoria e identità"
    assert service.search_calls[0][1].max_runtime == 120
    assert [trace.tool for trace in state.tool_traces] == [
        "search_movies",
        "get_movie_details",
    ]
