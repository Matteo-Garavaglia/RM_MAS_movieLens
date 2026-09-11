"""Versioned server-side retrieval profiles for the MCP adapter.

Profiles deliberately bundle the encoder configuration, its compatible Qdrant
collection and the optional reranker.  MCP callers select a capability
(`search_movies`), not an implementation detail such as a model name.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

from .config import BASELINE, RERANKER, DenseBaselineConfig, RerankerConfig


MCP_CONTRACT_VERSION = "movie-search/v1"
DEFAULT_MCP_PROFILE_ID = "dense_v1"
PROFILE_ENV_VAR = "MOVIE_RAG_PROFILE"
PROFILES_PATH_ENV_VAR = "MOVIE_RAG_PROFILES_PATH"

PipelineKind = Literal["dense", "dense_rerank"]


@dataclass(frozen=True)
class RetrievalProfile:
    """One reproducible retrieval pipeline selected when the server starts."""

    profile_id: str
    pipeline: PipelineKind
    encoder_provider: str
    dense_config: DenseBaselineConfig
    reranker_provider: str | None = None
    reranker_config: RerankerConfig | None = None
    default_limit: int = 10
    max_limit: int = 20
    exact: bool = True
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id cannot be empty")
        if self.encoder_provider != "bge_m3":
            raise ValueError(f"Unsupported encoder provider: {self.encoder_provider!r}")
        if self.pipeline not in {"dense", "dense_rerank"}:
            raise ValueError(f"Unsupported pipeline: {self.pipeline!r}")
        if self.default_limit <= 0 or self.max_limit <= 0:
            raise ValueError("default_limit and max_limit must be positive")
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit cannot exceed max_limit")
        if self.pipeline == "dense":
            if self.reranker_provider is not None or self.reranker_config is not None:
                raise ValueError("A dense profile cannot define a reranker")
        else:
            if self.reranker_provider != "flag_embedding":
                raise ValueError(
                    "A dense_rerank profile requires reranker_provider='flag_embedding'"
                )
            if self.reranker_config is None:
                raise ValueError("A dense_rerank profile requires reranker configuration")
            if self.reranker_config.candidate_pool_size < self.max_limit:
                raise ValueError("candidate_pool_size cannot be smaller than max_limit")

    @property
    def reranker_enabled(self) -> bool:
        return self.pipeline == "dense_rerank"


@dataclass(frozen=True)
class RetrievalProfileCatalog:
    """Validated profile registry loaded from one configuration snapshot."""

    contract_version: str
    default_profile_id: str
    profiles: Mapping[str, RetrievalProfile]
    source: str

    def __post_init__(self) -> None:
        if self.contract_version != MCP_CONTRACT_VERSION:
            raise ValueError(
                f"Profile contract is {self.contract_version!r}; "
                f"expected {MCP_CONTRACT_VERSION!r}"
            )
        if self.default_profile_id not in self.profiles:
            raise ValueError(
                f"Default profile {self.default_profile_id!r} is not present in the registry"
            )

    def get(self, profile_id: str | None = None) -> RetrievalProfile:
        selected = profile_id or self.default_profile_id
        try:
            return self.profiles[selected]
        except KeyError as exc:
            choices = ", ".join(sorted(self.profiles))
            raise KeyError(f"Unknown retrieval profile {selected!r}; available: {choices}") from exc


def _default_profile_mapping() -> dict[str, Any]:
    dense = {
        "encoder_provider": "bge_m3",
        "embedding_model": BASELINE.embedding_model,
        "embedding_dimension": BASELINE.embedding_dimension,
        "collection_name": BASELINE.collection_name,
        "vector_name": BASELINE.vector_name,
        "experiment_name": BASELINE.experiment_name,
        "version": BASELINE.version,
        "query_max_length": BASELINE.query_max_length,
        "normalize_embeddings": BASELINE.normalize_embeddings,
        "default_limit": 10,
        "max_limit": 20,
        "exact": True,
        "score_threshold": None,
    }
    return {
        "contract_version": MCP_CONTRACT_VERSION,
        "default_profile": DEFAULT_MCP_PROFILE_ID,
        "profiles": {
            "dense_v1": {"pipeline": "dense", **dense},
            "dense_rerank_v1": {
                "pipeline": "dense_rerank",
                **dense,
                "reranker": {
                    "provider": "flag_embedding",
                    "model_name": RERANKER.model_name,
                    "candidate_pool_size": RERANKER.candidate_pool_size,
                    "batch_size": RERANKER.batch_size,
                    "max_length": RERANKER.max_length,
                    "normalize_scores": RERANKER.normalize_scores,
                    "version": RERANKER.version,
                },
            },
        },
    }


def _project_profiles_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "mcp_profiles_v1.json"


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(f"Unknown {context} fields: {unknown}")


def _parse_profile(profile_id: str, raw: Mapping[str, Any]) -> RetrievalProfile:
    allowed = {
        "pipeline",
        "encoder_provider",
        "embedding_model",
        "embedding_dimension",
        "collection_name",
        "vector_name",
        "experiment_name",
        "version",
        "query_max_length",
        "normalize_embeddings",
        "default_limit",
        "max_limit",
        "exact",
        "score_threshold",
        "reranker",
    }
    _reject_unknown(raw, allowed, f"profile {profile_id!r}")
    pipeline = str(raw.get("pipeline", "")).strip()
    dense_config = replace(
        BASELINE,
        version=str(raw.get("version", BASELINE.version)),
        experiment_name=str(raw.get("experiment_name", BASELINE.experiment_name)),
        embedding_model=str(raw.get("embedding_model", BASELINE.embedding_model)),
        embedding_dimension=int(
            raw.get("embedding_dimension", BASELINE.embedding_dimension)
        ),
        query_max_length=int(raw.get("query_max_length", BASELINE.query_max_length)),
        normalize_embeddings=bool(
            raw.get("normalize_embeddings", BASELINE.normalize_embeddings)
        ),
        vector_name=str(raw.get("vector_name", BASELINE.vector_name)),
        collection_name=str(raw.get("collection_name", BASELINE.collection_name)),
    )

    reranker_raw = raw.get("reranker")
    reranker_provider: str | None = None
    reranker_config: RerankerConfig | None = None
    if reranker_raw is not None:
        if not isinstance(reranker_raw, Mapping):
            raise TypeError(f"Profile {profile_id!r} reranker must be an object")
        reranker_allowed = {
            "provider",
            "model_name",
            "candidate_pool_size",
            "batch_size",
            "max_length",
            "normalize_scores",
            "version",
        }
        _reject_unknown(reranker_raw, reranker_allowed, f"profile {profile_id!r} reranker")
        reranker_provider = str(reranker_raw.get("provider", "")).strip()
        reranker_config = replace(
            RERANKER,
            version=str(reranker_raw.get("version", RERANKER.version)),
            model_name=str(reranker_raw.get("model_name", RERANKER.model_name)),
            candidate_pool_size=int(
                reranker_raw.get("candidate_pool_size", RERANKER.candidate_pool_size)
            ),
            batch_size=int(reranker_raw.get("batch_size", RERANKER.batch_size)),
            max_length=int(reranker_raw.get("max_length", RERANKER.max_length)),
            normalize_scores=bool(
                reranker_raw.get("normalize_scores", RERANKER.normalize_scores)
            ),
        )

    threshold = raw.get("score_threshold")
    return RetrievalProfile(
        profile_id=profile_id,
        pipeline=pipeline,  # type: ignore[arg-type]
        encoder_provider=str(raw.get("encoder_provider", "bge_m3")),
        dense_config=dense_config,
        reranker_provider=reranker_provider,
        reranker_config=reranker_config,
        default_limit=int(raw.get("default_limit", 10)),
        max_limit=int(raw.get("max_limit", 20)),
        exact=bool(raw.get("exact", True)),
        score_threshold=None if threshold is None else float(threshold),
    )


def load_profile_catalog(path: str | Path | None = None) -> RetrievalProfileCatalog:
    """Load profiles from JSON, falling back to the versioned built-in registry."""

    configured_path = path or os.getenv(PROFILES_PATH_ENV_VAR)
    if configured_path is not None:
        source_path = Path(configured_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"MCP profile configuration not found: {source_path}")
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        source = str(source_path)
    else:
        project_path = _project_profiles_path()
        if project_path.exists():
            raw = json.loads(project_path.read_text(encoding="utf-8"))
            source = str(project_path)
        else:
            raw = _default_profile_mapping()
            source = "built-in"

    if not isinstance(raw, Mapping):
        raise TypeError("MCP profile configuration must be a JSON object")
    _reject_unknown(
        raw,
        {"contract_version", "default_profile", "profiles"},
        "profile catalog",
    )
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ValueError("MCP profile configuration must contain at least one profile")
    profiles = {
        str(profile_id): _parse_profile(str(profile_id), profile)
        for profile_id, profile in raw_profiles.items()
        if isinstance(profile, Mapping)
    }
    if len(profiles) != len(raw_profiles):
        raise TypeError("Every MCP profile must be a JSON object")
    return RetrievalProfileCatalog(
        contract_version=str(raw.get("contract_version", "")),
        default_profile_id=str(raw.get("default_profile", "")),
        profiles=profiles,
        source=source,
    )


def selected_profile_id(explicit_profile_id: str | None = None) -> str | None:
    """Resolve an optional explicit selection before the environment fallback."""

    return explicit_profile_id or os.getenv(PROFILE_ENV_VAR) or None

