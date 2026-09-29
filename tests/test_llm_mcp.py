from __future__ import annotations

from types import SimpleNamespace

import pytest

from movie_rag.llm_mcp import MCPMovieToolGateway, ToolGatewayError
from movie_rag.mcp_contract import (
    MovieDetailsResponse,
    MovieFilterInput,
    RetrievalPipelineInfo,
    SearchMoviesResponse,
)
from movie_rag.mcp_profiles import load_profile_catalog


def pipeline_info() -> RetrievalPipelineInfo:
    return RetrievalPipelineInfo.from_profile(
        load_profile_catalog().get("dense_v1")
    )


class FakeClient:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "search_movies":
            payload = SearchMoviesResponse(
                status="no_results",
                query=arguments["query"],
                applied_filters=MovieFilterInput.model_validate(
                    arguments["filters"]
                ),
                requested_limit=arguments["limit"],
                returned_count=0,
                pipeline=pipeline_info(),
                results=[],
                message="No catalog results.",
            ).model_dump(mode="json")
        else:
            payload = MovieDetailsResponse(
                requested_movie_ids=arguments["movie_ids"],
                returned_count=0,
                missing_movie_ids=arguments["movie_ids"],
                movies=[],
            ).model_dump(mode="json")
        return SimpleNamespace(structured_content=payload, is_error=False)


@pytest.mark.anyio
async def test_mcp_gateway_uses_stable_public_contract():
    client = FakeClient()
    gateway = MCPMovieToolGateway(client)

    execution = await gateway.search_movies(
        query="film italiano",
        filters=MovieFilterInput(languages=["it"]),
        limit=10,
    )

    assert execution.value.status == "no_results"
    assert execution.trace.status == "no_results"
    assert client.calls == [
        (
            "search_movies",
            {
                "query": "film italiano",
                "filters": {"languages": ["it"]},
                "limit": 10,
            },
        )
    ]


@pytest.mark.anyio
async def test_mcp_gateway_turns_transport_errors_into_auditable_failure():
    class BrokenClient:
        async def call_tool(self, name, arguments):
            raise ConnectionError("sensitive internal detail")

    gateway = MCPMovieToolGateway(BrokenClient())
    with pytest.raises(ToolGatewayError, match="ConnectionError") as caught:
        await gateway.search_movies(
            query="memory",
            filters=MovieFilterInput(),
            limit=5,
        )

    assert caught.value.trace.status == "error"
    assert caught.value.trace.error_type == "ConnectionError"
    assert "sensitive internal detail" not in str(caught.value)
