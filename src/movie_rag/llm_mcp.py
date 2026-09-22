"""Typed gateways between the LLM workflow and Movie RAG MCP tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from .llm_contract import ToolCallTrace
from .mcp_contract import (
    MovieDetailsResponse,
    MovieFilterInput,
    SearchMoviesResponse,
)


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ToolExecution(Generic[T]):
    """Validated tool value plus a transport-neutral audit record."""

    value: T
    trace: ToolCallTrace


class MovieToolGateway(Protocol):
    """Capabilities required by the orchestrator, independent of transport."""

    async def search_movies(
        self,
        *,
        query: str,
        filters: MovieFilterInput,
        limit: int,
    ) -> ToolExecution[SearchMoviesResponse]: ...

    async def get_movie_details(
        self,
        *,
        movie_ids: list[int],
    ) -> ToolExecution[MovieDetailsResponse]: ...


class ToolGatewayError(RuntimeError):
    """Sanitized MCP/service failure carrying a trace for the workflow state."""

    def __init__(self, message: str, trace: ToolCallTrace) -> None:
        super().__init__(message)
        self.trace = trace


def _trace(
    *,
    tool: str,
    started_at: datetime,
    started_counter: float,
    status: str,
    returned_count: int | None = None,
    error_type: str | None = None,
) -> ToolCallTrace:
    return ToolCallTrace(
        tool=tool,
        started_at=started_at,
        duration_ms=(perf_counter() - started_counter) * 1000.0,
        status=status,  # type: ignore[arg-type]
        returned_count=returned_count,
        error_type=error_type,
    )


def _structured_payload(result: Any) -> Any:
    payload = getattr(result, "structured_content", None)
    if payload is not None:
        return payload

    # Compatibility fallback for MCP clients that expose only text content.
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    raise ValueError("MCP response has no structured JSON content")


class MCPMovieToolGateway:
    """Invoke the stable public tools through an initialized MCP client.

    The client lifecycle remains outside this class.  For the in-process
    notebook this means constructing it inside ``async with Client(server)``;
    the same gateway also works with a separately launched stdio client.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    async def _call(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        schema: type[T],
    ) -> ToolExecution[T]:
        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        try:
            result = await self.client.call_tool(tool, arguments)
            if bool(getattr(result, "is_error", False)):
                raise RuntimeError("MCP tool returned an error result")
            value = schema.model_validate(_structured_payload(result))
        except Exception as exc:
            trace = _trace(
                tool=tool,
                started_at=started_at,
                started_counter=started_counter,
                status="error",
                error_type=type(exc).__name__,
            )
            raise ToolGatewayError(
                f"Tool {tool!r} failed with {type(exc).__name__}", trace
            ) from exc

        returned_count = getattr(value, "returned_count", None)
        status = (
            "no_results"
            if isinstance(value, SearchMoviesResponse) and value.status == "no_results"
            else "ok"
        )
        return ToolExecution(
            value=value,
            trace=_trace(
                tool=tool,
                started_at=started_at,
                started_counter=started_counter,
                status=status,
                returned_count=returned_count,
            ),
        )

    async def search_movies(
        self,
        *,
        query: str,
        filters: MovieFilterInput,
        limit: int,
    ) -> ToolExecution[SearchMoviesResponse]:
        return await self._call(
            tool="search_movies",
            arguments={
                "query": query,
                "filters": filters.model_dump(
                    mode="json", exclude_none=True, exclude_defaults=True
                ),
                "limit": limit,
            },
            schema=SearchMoviesResponse,
        )

    async def get_movie_details(
        self,
        *,
        movie_ids: list[int],
    ) -> ToolExecution[MovieDetailsResponse]:
        return await self._call(
            tool="get_movie_details",
            arguments={"movie_ids": movie_ids},
            schema=MovieDetailsResponse,
        )


class ServiceMovieToolGateway:
    """Direct adapter for tests or applications that do not need MCP transport.

    It exposes exactly the same capability boundary as ``MCPMovieToolGateway``.
    Production experiments intended to measure MCP should use the MCP gateway.
    """

    def __init__(self, service: Any) -> None:
        self.service = service

    async def search_movies(
        self,
        *,
        query: str,
        filters: MovieFilterInput,
        limit: int,
    ) -> ToolExecution[SearchMoviesResponse]:
        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        try:
            value = self.service.search_movies(
                query=query,
                filters=filters,
                limit=limit,
            )
            value = SearchMoviesResponse.model_validate(value)
        except Exception as exc:
            trace = _trace(
                tool="search_movies",
                started_at=started_at,
                started_counter=started_counter,
                status="error",
                error_type=type(exc).__name__,
            )
            raise ToolGatewayError(
                f"Tool 'search_movies' failed with {type(exc).__name__}", trace
            ) from exc
        return ToolExecution(
            value=value,
            trace=_trace(
                tool="search_movies",
                started_at=started_at,
                started_counter=started_counter,
                status="no_results" if value.status == "no_results" else "ok",
                returned_count=value.returned_count,
            ),
        )

    async def get_movie_details(
        self,
        *,
        movie_ids: list[int],
    ) -> ToolExecution[MovieDetailsResponse]:
        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        try:
            value = self.service.get_movie_details(movie_ids)
            value = MovieDetailsResponse.model_validate(value)
        except Exception as exc:
            trace = _trace(
                tool="get_movie_details",
                started_at=started_at,
                started_counter=started_counter,
                status="error",
                error_type=type(exc).__name__,
            )
            raise ToolGatewayError(
                f"Tool 'get_movie_details' failed with {type(exc).__name__}", trace
            ) from exc
        return ToolExecution(
            value=value,
            trace=_trace(
                tool="get_movie_details",
                started_at=started_at,
                started_counter=started_counter,
                status="ok",
                returned_count=value.returned_count,
            ),
        )
