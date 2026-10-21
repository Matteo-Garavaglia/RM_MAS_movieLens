from __future__ import annotations

from typing import Any, Callable, Mapping

from .resource_models import ResourceFilters
from .resource_service import RetrievalService


def create_resource_mcp_server(
    service: RetrievalService,
    name: str = "Resource Retrieval MCP",
    filter_adapter: Callable[[Mapping[str, Any] | None], ResourceFilters] | None = None,
    *,
    default_profile_id: str = "fast",
    expose_profile_to_client: bool = False,
    exact: bool = False,
):
    """Expose domain resources without leaking vector-store/model details.

    By default the retrieval profile is **server-side configuration**.  This keeps
    the LLM from choosing implementation parameters such as candidate-pool size or
    whether a local reranker is enabled.  ``expose_profile_to_client=True`` is an
    explicit future opt-in for experiments where the orchestrator is allowed to
    select a retrieval mode.
    """

    # try:
    #     from mcp.server.fastmcp import FastMCP
    # except ImportError:  # pragma: no cover - depends on installed MCP stack
    #     try:
    #         from fastmcp import FastMCP
    #     except ImportError as exc:
    #         raise ImportError("Install the MCP/FastMCP dependency to expose tools") from exc

    # print("FastMCP initialized.")
    # if default_profile_id not in service.profiles:
    #     raise KeyError(
    #         f"unknown default search profile {default_profile_id!r}; "
    #         f"available={sorted(service.profiles)}"
    #     )
    
    # mcp = FastMCP(name)
    
    from mcp.server import MCPServer

    mcp = MCPServer(name=name, title=name, version="0.4.0")



    def _normalize(filters: dict[str, Any] | None) -> ResourceFilters:
        return (
            filter_adapter(filters)
            if filter_adapter is not None
            else ResourceFilters.model_validate(filters or {})
        )

    if expose_profile_to_client:

        @mcp.tool()
        def search_resources(
            query: str,
            filters: dict[str, Any] | None = None,
            limit: int = 10,
            profile: str = default_profile_id,
        ) -> dict[str, Any]:
            """Search grounded resources with optional domain metadata constraints."""

            response = service.search(
                query,
                filters=_normalize(filters),
                profile_id=profile,
                limit=limit,
                exact=exact,
            )
            return response.model_dump(mode="json")

    else:

        @mcp.tool()
        def search_resources(
            query: str,
            filters: dict[str, Any] | None = None,
            limit: int = 10,
        ) -> dict[str, Any]:
            """Search grounded resources with optional domain metadata constraints."""

            response = service.search(
                query,
                filters=_normalize(filters),
                profile_id=default_profile_id,
                limit=limit,
                exact=exact,
            )
            return response.model_dump(mode="json")

    @mcp.tool()
    def get_resources(resource_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch complete grounded resources by their stable domain identifiers."""

        return [
            resource.model_dump(mode="json")
            for resource in service.get_resources(resource_ids)
        ]

    return mcp


__all__ = ["create_resource_mcp_server"]
