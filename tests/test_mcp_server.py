from __future__ import annotations

import pytest


mcp = pytest.importorskip("mcp")

from mcp import Client

from movie_rag.mcp_server import create_mcp_server
from movie_rag.mcp_service import MovieSearchService
from test_mcp_service import FakeRetriever, profile


class FakeRegistry:
    def __init__(self):
        self.services = {
            "dense_v1": MovieSearchService(profile("dense_v1"), FakeRetriever()),
            "dense_rerank_v1": MovieSearchService(
                profile("dense_rerank_v1"), FakeRetriever()
            ),
        }

    def get(self, profile_id=None):
        return self.services[profile_id or "dense_v1"]


FORBIDDEN_PRODUCTION_ARGUMENTS = {
    "profile_id",
    "model",
    "rerank",
    "candidate_pool_size",
    "exact",
    "score_threshold",
    "collection_name",
}


@pytest.mark.anyio
async def test_production_contract_hides_technical_selection():
    server = create_mcp_server(FakeRegistry(), enable_experimental=False)

    async with Client(server, raise_exceptions=True) as client:
        listing = await client.list_tools()
        tools = {tool.name: tool for tool in listing.tools}

    assert set(tools) == {"search_movies", "get_movie_details"}
    search_properties = set(tools["search_movies"].input_schema["properties"])
    assert search_properties == {"query", "filters", "limit"}
    assert not search_properties & FORBIDDEN_PRODUCTION_ARGUMENTS
    assert tools["search_movies"].output_schema is not None
    assert tools["search_movies"].annotations.read_only_hint is True


@pytest.mark.anyio
async def test_structured_search_and_details_roundtrip():
    server = create_mcp_server(FakeRegistry(), enable_experimental=False)

    async with Client(server, raise_exceptions=True) as client:
        search = await client.call_tool(
            "search_movies",
            {
                "query": "memory and identity",
                "filters": {"max_runtime": 120},
                "limit": 5,
            },
        )
        details = await client.call_tool("get_movie_details", {"movie_ids": [1, 2]})

    assert search.structured_content["status"] == "ok"
    assert search.structured_content["pipeline"]["profile_id"] == "dense_v1"
    assert search.structured_content["results"][0]["movie_id"] == 1
    assert details.structured_content["missing_movie_ids"] == [2]


@pytest.mark.anyio
async def test_experimental_profile_override_is_per_call_and_opt_in():
    registry = FakeRegistry()
    server = create_mcp_server(registry, enable_experimental=True)

    async with Client(server, raise_exceptions=True) as client:
        listing = await client.list_tools()
        names = {tool.name for tool in listing.tools}
        experiment = await client.call_tool(
            "search_movies_experimental",
            {
                "query": "memory",
                "profile_id": "dense_rerank_v1",
                "execution_overrides": {"candidate_pool_size": 30},
            },
        )
        production = await client.call_tool(
            "search_movies", {"query": "memory", "limit": 10}
        )

    assert "search_movies_experimental" in names
    assert experiment.structured_content["pipeline"]["profile_id"] == "dense_rerank_v1"
    assert experiment.structured_content["pipeline"]["candidate_pool_size"] == 30
    assert production.structured_content["pipeline"]["profile_id"] == "dense_v1"
