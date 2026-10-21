from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from .resource_models import ResourceDocument, ResourceSearchResponse


class MCPToolTrace(BaseModel):
    tool: str
    latency_ms: float
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True


def _maybe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _extract_mcp_payload(result: Any) -> Any:
    """Normalize FastMCP/mcp client result shapes across library versions."""

    if result is None:
        return None
    if isinstance(result, (dict, list, str, int, float, bool)):
        return result

    for attr in ("structured_content", "data"):
        value = getattr(result, attr, None)
        if value not in (None, {}):
            return _maybe_model_dump(value)

    content = getattr(result, "content", None)
    if content:
        texts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(str(text))
        if texts:
            joined = "\n".join(texts).strip()
            try:
                return json.loads(joined)
            except json.JSONDecodeError:
                return joined

    dumped = _maybe_model_dump(result)
    if dumped is not result:
        return dumped
    raise TypeError(f"unsupported MCP result type: {type(result)!r}")


@dataclass
class MCPResourceToolGateway:
    """Typed client boundary between orchestration and generic MCP tools."""

    client: Any

    async def search_resources(
        self,
        *,
        query: str,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> tuple[ResourceSearchResponse, MCPToolTrace]:
        arguments = {
            "query": str(query),
            "filters": dict(filters or {}),
            "limit": int(limit),
        }
        t0 = perf_counter()
        result = await self.client.call_tool("search_resources", arguments)
        elapsed = (perf_counter() - t0) * 1000.0
        payload = _extract_mcp_payload(result)
        response = ResourceSearchResponse.model_validate(payload)
        return response, MCPToolTrace(
            tool="search_resources",
            latency_ms=elapsed,
            arguments=arguments,
        )

    async def get_resources(
        self,
        resource_ids: Sequence[str | int],
    ) -> tuple[list[ResourceDocument], MCPToolTrace]:
        arguments = {"resource_ids": [str(value) for value in resource_ids]}
        t0 = perf_counter()
        result = await self.client.call_tool("get_resources", arguments)
        elapsed = (perf_counter() - t0) * 1000.0
        payload = _extract_mcp_payload(result)
        resources = [ResourceDocument.model_validate(item) for item in (payload or [])]
        return resources, MCPToolTrace(
            tool="get_resources",
            latency_ms=elapsed,
            arguments=arguments,
        )


__all__ = ["MCPResourceToolGateway", "MCPToolTrace"]
