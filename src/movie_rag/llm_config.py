"""Versioned configuration for providers, logical agents and workflows."""

from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .llm_contract import LLM_CONTRACT_VERSION, QueryRewriteStrategy


LLM_CONFIG_ENV_VAR = "MOVIE_RAG_LLM_CONFIG"
LLM_WORKFLOW_ENV_VAR = "MOVIE_RAG_LLM_WORKFLOW"
ROLE_PROFILE_ENV_VARS = {
    "planner": "MOVIE_RAG_PLANNER_LLM_PROFILE",
    "critic": "MOVIE_RAG_CRITIC_LLM_PROFILE",
    "answer": "MOVIE_RAG_ANSWER_LLM_PROFILE",
}


class FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMProviderProfile(FrozenConfigModel):
    """One provider/model configuration referenced by logical agents."""

    profile_id: str
    provider: Literal["gemini", "openai"]
    model: str
    api_key_env: str
    base_url_env: str | None = None
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, ge=1, le=65536)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    @field_validator(
        "profile_id", "model", "api_key_env", "base_url_env", mode="before"
    )
    @classmethod
    def _clean_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

    @model_validator(mode="after")
    def _required_strings(self) -> "LLMProviderProfile":
        for field_name in ("profile_id", "model", "api_key_env"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty")
        return self


class AgentProfile(FrozenConfigModel):
    """Prompt and model binding for one logical agent role."""

    agent_id: str
    llm_profile: str
    prompt_resource: str
    prompt_version: str
    enabled: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=65536)

    @field_validator(
        "agent_id",
        "llm_profile",
        "prompt_resource",
        "prompt_version",
        mode="before",
    )
    @classmethod
    def _clean_strings(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class WorkflowProfile(FrozenConfigModel):
    """Bounded orchestration policy, independent of any provider SDK."""

    workflow_id: str
    planner_agent: str
    critic_agent: str | None = None
    answer_agent: str
    enable_critic: bool = False
    max_retries: int = Field(default=1, ge=0, le=3)
    search_limit: int = Field(default=10, ge=1, le=20)
    details_limit: int = Field(default=5, ge=1, le=20)
    catalog_language: str = Field(default="en", min_length=2, max_length=50)
    query_rewrite_strategy: QueryRewriteStrategy = "none"

    @model_validator(mode="after")
    def _critic_is_configured(self) -> "WorkflowProfile":
        if self.enable_critic and not self.critic_agent:
            raise ValueError("enable_critic=true requires critic_agent")
        if self.details_limit > self.search_limit:
            raise ValueError("details_limit cannot exceed search_limit")
        return self


class LLMConfigCatalog(FrozenConfigModel):
    """Validated snapshot of all model, agent and workflow choices."""

    contract_version: Literal["movie-llm/v1"] = LLM_CONTRACT_VERSION
    default_workflow: str
    llm_profiles: Mapping[str, LLMProviderProfile]
    agents: Mapping[str, AgentProfile]
    workflows: Mapping[str, WorkflowProfile]
    source: str

    @model_validator(mode="after")
    def _validate_references(self) -> "LLMConfigCatalog":
        if self.default_workflow not in self.workflows:
            raise ValueError(
                f"Default workflow {self.default_workflow!r} is not configured"
            )
        for agent_id, agent in self.agents.items():
            if agent.agent_id != agent_id:
                raise ValueError(f"Agent key {agent_id!r} does not match agent_id")
            if agent.llm_profile not in self.llm_profiles:
                raise ValueError(
                    f"Agent {agent_id!r} references unknown LLM profile "
                    f"{agent.llm_profile!r}"
                )
        for workflow_id, workflow in self.workflows.items():
            if workflow.workflow_id != workflow_id:
                raise ValueError(
                    f"Workflow key {workflow_id!r} does not match workflow_id"
                )
            referenced = [workflow.planner_agent, workflow.answer_agent]
            if workflow.critic_agent:
                referenced.append(workflow.critic_agent)
            unknown = [agent for agent in referenced if agent not in self.agents]
            if unknown:
                raise ValueError(
                    f"Workflow {workflow_id!r} references unknown agents: {unknown}"
                )
        return self

    def get_llm_profile(self, profile_id: str) -> LLMProviderProfile:
        try:
            return self.llm_profiles[profile_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self.llm_profiles))
            raise KeyError(
                f"Unknown LLM profile {profile_id!r}; available: {choices}"
            ) from exc

    def get_agent(self, agent_id: str) -> AgentProfile:
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self.agents))
            raise KeyError(f"Unknown agent {agent_id!r}; available: {choices}") from exc

    def get_workflow(self, workflow_id: str | None = None) -> WorkflowProfile:
        selected = workflow_id or self.default_workflow
        try:
            return self.workflows[selected]
        except KeyError as exc:
            choices = ", ".join(sorted(self.workflows))
            raise KeyError(
                f"Unknown workflow {selected!r}; available: {choices}"
            ) from exc


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(f"Unknown {context} fields: {unknown}")


def _project_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "llm_profiles_v1.json"


def _parse_named_models(
    raw: Mapping[str, Any], model_type: type[BaseModel], id_field: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item_id, item in raw.items():
        if not isinstance(item, Mapping):
            raise TypeError(f"Configuration {item_id!r} must be an object")
        result[str(item_id)] = model_type.model_validate(
            {id_field: str(item_id), **dict(item)}
        )
    return result


def load_llm_config(path: str | Path | None = None) -> LLMConfigCatalog:
    """Load and validate one reproducible LLM/orchestration snapshot."""

    configured = path or os.getenv(LLM_CONFIG_ENV_VAR)
    source_path = (
        Path(configured).expanduser().resolve()
        if configured is not None
        else _project_config_path()
    )
    if not source_path.exists():
        raise FileNotFoundError(f"LLM configuration not found: {source_path}")
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("LLM configuration must be a JSON object")
    _reject_unknown(
        raw,
        {
            "contract_version",
            "default_workflow",
            "llm_profiles",
            "agents",
            "workflows",
        },
        "LLM catalog",
    )
    for field_name in ("llm_profiles", "agents", "workflows"):
        if not isinstance(raw.get(field_name), Mapping) or not raw[field_name]:
            raise ValueError(f"{field_name} must contain at least one entry")

    return LLMConfigCatalog(
        contract_version=str(raw.get("contract_version", "")),
        default_workflow=str(raw.get("default_workflow", "")),
        llm_profiles=_parse_named_models(
            raw["llm_profiles"], LLMProviderProfile, "profile_id"
        ),
        agents=_parse_named_models(raw["agents"], AgentProfile, "agent_id"),
        workflows=_parse_named_models(
            raw["workflows"], WorkflowProfile, "workflow_id"
        ),
        source=str(source_path),
    )


def selected_workflow_id(explicit: str | None = None) -> str | None:
    """Return an explicit workflow selection without inventing a default."""

    value = explicit or os.getenv(LLM_WORKFLOW_ENV_VAR)
    return value.strip() if value and value.strip() else None


def role_profile_overrides_from_environment() -> dict[str, str]:
    """Read opt-in per-role provider profiles for one application process."""

    result: dict[str, str] = {}
    for role, env_var in ROLE_PROFILE_ENV_VARS.items():
        value = os.getenv(env_var)
        if value and value.strip():
            result[role] = value.strip()
    return result


def load_prompt_resource(resource: str) -> str:
    """Read a package prompt while rejecting absolute and parent paths."""

    path = PurePosixPath(resource)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe prompt resource path: {resource!r}")
    prompt = files("movie_rag").joinpath(*path.parts).read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"Prompt resource is empty: {resource!r}")
    return prompt.strip()
