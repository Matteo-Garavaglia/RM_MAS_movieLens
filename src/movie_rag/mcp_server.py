"""MCP transport adapter for versioned Movie RAG retrieval capabilities."""

from __future__ import annotations

import logging
import os
from threading import RLock
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from .mcp_contract import (
    MovieDetailsResponse,
    MovieFilterInput,
    SearchExecutionOverrides,
    SearchMoviesResponse,
)
from .mcp_service import MovieSearchServiceRegistry


SERVER_VERSION = "0.4.0"
EXPERIMENTAL_TOOLS_ENV_VAR = "MOVIE_RAG_ENABLE_EXPERIMENTAL_TOOLS"
TRANSPORT_ENV_VAR = "MOVIE_RAG_MCP_TRANSPORT"


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


class _LazyRegistry:
    """Create the Qdrant/model registry on first tool call, never at import time."""

    def __init__(self, registry: Any | None = None) -> None:
        self._registry = registry
        self._lock = RLock()

    def get_registry(self) -> Any:
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    self._registry = MovieSearchServiceRegistry.from_environment()
        return self._registry


READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def create_mcp_server(
    registry: Any | None = None,
    enable_experimental: bool | None = None,
) -> MCPServer:
    """Create a server whose public tools do not expose model implementation details."""

    lazy_registry = _LazyRegistry(registry)
    experimental_enabled = (
        _environment_flag(EXPERIMENTAL_TOOLS_ENV_VAR)
        if enable_experimental is None
        else bool(enable_experimental)
    )
    server = MCPServer(
        name="movie-rag",
        title="Movie RAG",
        version=SERVER_VERSION,
        description=(
            "Read-only retrieval over the indexed movie catalog. Hard constraints "
            "belong in filters; thematic preferences belong in query."
        ),
        instructions=(
            "Use search_movies to find catalog candidates. Use get_movie_details only "
            "for IDs returned by search_movies. Never invent a movie outside the results."
        ),
    )

    @server.tool(
        name="search_movies",
        title="Search movies",
        description=(
            "Search the indexed catalog. Put themes, plot, mood and other soft "
            "preferences in query. Put year, runtime, language, country, genre and "
            "adult constraints in filters. The server selects the retrieval profile."
        ),
        annotations=READ_ONLY_ANNOTATIONS,
        structured_output=True,
    )
    def search_movies(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=2000,
                description="Natural-language semantic movie request.",
            ),
        ],
        filters: MovieFilterInput | None = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=20,
                description="Maximum number of catalog candidates to return.",
            ),
        ] = 10,
    ) -> SearchMoviesResponse:
        return lazy_registry.get_registry().get().search_movies(
            query=query,
            filters=filters,
            limit=limit,
        )

    @server.tool(
        name="get_movie_details",
        title="Get movie details",
        description=(
            "Return grounded catalog metadata for movie IDs already obtained from "
            "search_movies. Missing IDs are reported explicitly."
        ),
        annotations=READ_ONLY_ANNOTATIONS,
        structured_output=True,
    )
    def get_movie_details(
        movie_ids: Annotated[
            list[int],
            Field(
                min_length=1,
                max_length=50,
                description="Unique MovieLens movie IDs returned by search_movies.",
            ),
        ],
    ) -> MovieDetailsResponse:
        return lazy_registry.get_registry().get().get_movie_details(movie_ids)

    if experimental_enabled:

        @server.tool(
            name="search_movies_experimental",
            title="Search movies with an experimental profile",
            description=(
                "Evaluation-only movie search. Select one server-whitelisted profile "
                "for this call and optionally override bounded execution parameters. "
                "It does not mutate the default profile or any later call."
            ),
            annotations=READ_ONLY_ANNOTATIONS,
            structured_output=True,
        )
        def search_movies_experimental(
            query: Annotated[
                str,
                Field(min_length=1, max_length=2000),
            ],
            profile_id: Annotated[
                str,
                Field(
                    min_length=1,
                    max_length=100,
                    description=(
                        "Versioned profile ID allowed by the server configuration; "
                        "never a free-form model name."
                    ),
                ),
            ],
            filters: MovieFilterInput | None = None,
            limit: Annotated[int, Field(ge=1, le=20)] = 10,
            execution_overrides: SearchExecutionOverrides | None = None,
        ) -> SearchMoviesResponse:
            return lazy_registry.get_registry().get(profile_id).search_movies(
                query=query,
                filters=filters,
                limit=limit,
                execution_overrides=execution_overrides,
            )

    return server


mcp = create_mcp_server()


def main() -> None:
    """Run the server; stdio remains the safe local-development default."""

    logging.basicConfig(level=logging.INFO)
    transport = os.getenv(TRANSPORT_ENV_VAR, "stdio").strip().casefold()
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError(
            f"{TRANSPORT_ENV_VAR} must be 'stdio' or 'streamable-http', got {transport!r}"
        )
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
