from __future__ import annotations

import json
from pathlib import Path

import pytest

from movie_rag.llm_config import (
    load_llm_config,
    load_prompt_resource,
    role_profile_overrides_from_environment,
)


def test_default_llm_catalog_is_referentially_valid():
    catalog = load_llm_config()

    workflow = catalog.get_workflow()
    assert catalog.contract_version == "movie-llm/v1"
    assert workflow.workflow_id == "grounded_v1"
    assert workflow.enable_critic is False
    assert catalog.get_agent(workflow.planner_agent).prompt_version == "planner_v1"
    assert catalog.get_llm_profile("gemini_free").provider == "gemini"
    assert "grounded movie recommender" in load_prompt_resource(
        "prompts/answer_v1.txt"
    )


def test_role_profile_overrides_are_explicit_environment_choices(monkeypatch):
    monkeypatch.setenv("MOVIE_RAG_PLANNER_LLM_PROFILE", "openai_default")
    monkeypatch.delenv("MOVIE_RAG_CRITIC_LLM_PROFILE", raising=False)

    assert role_profile_overrides_from_environment() == {
        "planner": "openai_default"
    }


def test_invalid_cross_reference_is_rejected(tmp_path):
    source = json.loads(
        Path("configs/llm_profiles_v1.json").read_text(encoding="utf-8")
    )
    source["agents"]["planner"]["llm_profile"] = "missing_profile"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown LLM profile"):
        load_llm_config(path)


def test_prompt_resource_rejects_parent_traversal():
    with pytest.raises(ValueError, match="Unsafe"):
        load_prompt_resource("../secret.txt")
