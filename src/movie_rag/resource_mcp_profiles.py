from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field, field_validator, model_validator

from .resource_embeddings import DenseEncoderSpec, SentenceTransformerResourceEncoder
from .resource_qdrant import QdrantCollectionConfig, QdrantResourceRetriever, connect_qdrant
from .resource_service import (
    RetrievalService,
    ResourceReranker,
    ScoreRerankerAdapter,
    SearchProfile,
)
from .resource_rerankers import SentenceTransformerCrossEncoderScorer


class EncoderProfile(BaseModel):
    name: str
    model_name: str
    dimension: int
    trust_remote_code: bool = False
    batch_size: int = 16
    normalize_embeddings: bool = True
    max_seq_length: int | None = None
    query_prompt_name: str | None = None

    def to_spec(self) -> DenseEncoderSpec:
        return DenseEncoderSpec(**self.model_dump())




class RerankerProfile(BaseModel):
    """Declarative reranker configuration loaded server-side from JSON.

    ``build()`` creates the generic ``ResourceReranker`` adapter, but the
    underlying CrossEncoder remains lazy: model weights are loaded only on the
    first quality search that actually invokes reranking.
    """

    name: str
    backend: str = "sentence_transformers_cross_encoder"
    model_name: str
    trust_remote_code: bool = False
    batch_size: int = 8
    max_length: int = 1024
    device: str = "auto"
    precision: str = "auto"

    @field_validator("name", "backend", "model_name", "device", "precision")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("batch_size", "max_length")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError("value must be > 0")
        return value

    @field_validator("precision")
    @classmethod
    def _supported_precision(cls, value: str) -> str:
        if value not in {"auto", "float16", "float32"}:
            raise ValueError("precision must be one of: auto, float16, float32")
        return value

    def build(self) -> ResourceReranker:
        if self.backend != "sentence_transformers_cross_encoder":
            raise ValueError(f"unsupported reranker backend: {self.backend!r}")
        scorer = SentenceTransformerCrossEncoderScorer(
            self.model_name,
            name=self.name,
            trust_remote_code=self.trust_remote_code,
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device,
            precision=self.precision,
        )
        return ScoreRerankerAdapter(scorer, name=self.name)


class ResourceMCPProfile(BaseModel):
    """Server-side retrieval profile.

    LLMs never see these implementation details.  They call ``search_resources``;
    the server decides which Qdrant collection, encoder and search profile to use.
    """

    profile_id: str
    domain_id: str
    collection_name: str
    search_profile_id: str = "fast"
    vector_name: str = "dense"
    schema_version: str = "resource_payload_v1"
    qdrant_url_env: str = "QDRANT_URL"
    qdrant_url_default: str = "http://localhost:6333"
    qdrant_api_key_env: str | None = None
    exact: bool = False
    encoder: EncoderProfile
    reranker: RerankerProfile | None = None
    search_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("profile_id", "domain_id", "collection_name", "search_profile_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_default_pipeline(self) -> "ResourceMCPProfile":
        # When custom search profiles are declared, fail during configuration
        # loading rather than at the first user query.
        if self.search_profiles:
            if self.search_profile_id not in self.search_profiles:
                raise ValueError(
                    f"default search profile {self.search_profile_id!r} is not defined; "
                    f"available={sorted(self.search_profiles)}"
                )
            selected = self.search_profiles[self.search_profile_id]
            if bool(selected.get("rerank", False)) and self.reranker is None:
                raise ValueError(
                    f"MCP profile {self.profile_id!r} selects reranking but has no "
                    "top-level 'reranker' configuration"
                )
        return self

    def build_search_profiles(self) -> dict[str, SearchProfile] | None:
        if not self.search_profiles:
            return None
        return {
            key: SearchProfile(profile_id=key, **value)
            for key, value in self.search_profiles.items()
        }


class ResourceMCPProfileCatalog(BaseModel):
    version: str = "1"
    profiles: dict[str, ResourceMCPProfile]

    def get(self, profile_id: str) -> ResourceMCPProfile:
        try:
            return self.profiles[profile_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown resource MCP profile {profile_id!r}; "
                f"available={sorted(self.profiles)}"
            ) from exc


def load_resource_mcp_profiles(path: str | Path) -> ResourceMCPProfileCatalog:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    catalog = ResourceMCPProfileCatalog.model_validate(payload)
    # Ensure dictionary key and embedded id cannot silently diverge.
    for key, profile in catalog.profiles.items():
        if key != profile.profile_id:
            raise ValueError(
                f"profile key {key!r} differs from profile_id {profile.profile_id!r}"
            )
    return catalog


class ResourceSearchServiceRegistry:
    """Build the generic retrieval service from server-side configuration."""

    def __init__(
        self,
        profile: ResourceMCPProfile,
        service: RetrievalService,
        *,
        client: Any,
    ) -> None:
        self.profile = profile
        self.service = service
        self.client = client

    @classmethod
    def from_profile(
        cls,
        profile: ResourceMCPProfile,
        *,
        reranker: ResourceReranker | None = None,
        client: Any | None = None,
        device: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "ResourceSearchServiceRegistry":
        env_map = os.environ if env is None else env
        if client is None:
            url = env_map.get(profile.qdrant_url_env, profile.qdrant_url_default)
            api_key = (
                env_map.get(profile.qdrant_api_key_env)
                if profile.qdrant_api_key_env
                else None
            )
            client = connect_qdrant(url=url, api_key=api_key)

        encoder = SentenceTransformerResourceEncoder(
            profile.encoder.to_spec(),
            device=device,
        )
        collection = QdrantCollectionConfig(
            collection_name=profile.collection_name,
            dimension=profile.encoder.dimension,
            vector_name=profile.vector_name,
            schema_version=profile.schema_version,
        )
        retriever = QdrantResourceRetriever(client, encoder, collection)
        # Normal path: the reranker is built from server-side configuration.
        # ``reranker=`` remains only as an explicit dependency-injection override
        # for tests/experiments; notebooks and MCP startup do not need to pass it.
        effective_reranker = reranker
        if effective_reranker is None and profile.reranker is not None:
            effective_reranker = profile.reranker.build()

        service = RetrievalService(
            retriever,
            reranker=effective_reranker,
            profiles=profile.build_search_profiles(),
        )
        if profile.search_profile_id not in service.profiles:
            raise ValueError(
                f"default search profile {profile.search_profile_id!r} is not defined; "
                f"available={sorted(service.profiles)}"
            )
        default_search = service.profiles[profile.search_profile_id]
        if default_search.rerank and service.reranker is None:
            raise ValueError(
                f"MCP profile {profile.profile_id!r} selects reranking but has no "
                "reranker configured. Add a top-level 'reranker' block or pass "
                "reranker= explicitly."
            )
        return cls(profile, service, client=client)

    @classmethod
    def from_environment(
        cls,
        config_path: str | Path,
        *,
        profile_id: str | None = None,
        profile_env: str = "RESOURCE_RAG_PROFILE",
        reranker: ResourceReranker | None = None,
        client: Any | None = None,
        device: str | None = None,
    ) -> "ResourceSearchServiceRegistry":
        catalog = load_resource_mcp_profiles(config_path)
        selected = profile_id or os.getenv(profile_env)
        if not selected:
            raise RuntimeError(
                f"set {profile_env} or pass profile_id; available={sorted(catalog.profiles)}"
            )
        return cls.from_profile(
            catalog.get(selected),
            reranker=reranker,
            client=client,
            device=device,
        )


__all__ = [
    "EncoderProfile",
    "RerankerProfile",
    "ResourceMCPProfile",
    "ResourceMCPProfileCatalog",
    "ResourceSearchServiceRegistry",
    "load_resource_mcp_profiles",
]
