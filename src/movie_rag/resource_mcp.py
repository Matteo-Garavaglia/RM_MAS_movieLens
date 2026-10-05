from __future__ import annotations

from typing import Any, Callable, Mapping

from .resource_models import ResourceFilters
from .resource_service import RetrievalService


def create_resource_mcp_server(
    service: RetrievalService,
    name: str = "Resource Retrieval MCP",
    filter_adapter: Callable[[Mapping[str, Any] | None], ResourceFilters] | None = None,
):
    """Expose the stable resource contract, not Qdrant/model implementation details.

    ``filter_adapter`` is the only domain-specific seam. MovieLens can keep the
    existing ``max_runtime/include_genres/...`` contract, while WhoTeach can later
    translate ``course/topic/resource_type/...`` constraints into ResourceFilters.
    """

    from mcp.server import MCPServer

    mcp = MCPServer(name=name, title=name, version="0.4.0")

    @mcp.tool()
    def search_resources(
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        profile: str = "fast",
    ) -> dict[str, Any]:
        """Search grounded resources by semantic query and optional metadata constraints."""

        normalized_filters = (
            filter_adapter(filters)
            if filter_adapter is not None
            else ResourceFilters.model_validate(filters or {})
        )
        response = service.search(
            query,
            filters=normalized_filters,
            profile_id=profile,
            limit=limit,
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
