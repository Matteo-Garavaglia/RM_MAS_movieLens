from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ResourceDomainProfile(BaseModel):
    """Domain-specific semantics injected into an otherwise generic LLM workflow.

    The orchestration/retrieval code should not know what a movie, a course, or a
    learning object is.  This profile intentionally keeps the domain vocabulary,
    admissible native filters and answer guidance outside the core workflow.
    """

    domain_id: str
    display_name: str
    resource_label: str = "resource"
    user_role: str = "user"
    task_description: str
    catalog_description: str
    catalog_language: str = "en"
    allowed_filter_keys: list[str] = Field(default_factory=list)
    planner_instructions: list[str] = Field(default_factory=list)
    answer_instructions: list[str] = Field(default_factory=list)
    metadata_hints: dict[str, str] = Field(default_factory=dict)
    template_only: bool = False

    @field_validator(
        "domain_id",
        "display_name",
        "resource_label",
        "user_role",
        "task_description",
        "catalog_description",
        "catalog_language",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    def validate_native_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        unsupported = sorted(set(filters) - set(self.allowed_filter_keys))
        if unsupported:
            raise ValueError(
                f"unsupported filters for domain {self.domain_id!r}: {unsupported}; "
                f"allowed={self.allowed_filter_keys}"
            )
        return filters


class ResourceDomainCatalog(BaseModel):
    version: str = "1"
    domains: dict[str, ResourceDomainProfile]

    def get(self, domain_id: str) -> ResourceDomainProfile:
        try:
            return self.domains[domain_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown domain profile {domain_id!r}; available={sorted(self.domains)}"
            ) from exc


def load_resource_domain_catalog(path: str | Path) -> ResourceDomainCatalog:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResourceDomainCatalog.model_validate(payload)


__all__ = [
    "ResourceDomainProfile",
    "ResourceDomainCatalog",
    "load_resource_domain_catalog",
]
