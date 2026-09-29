from __future__ import annotations

from types import SimpleNamespace

import pytest

from movie_rag.llm_config import load_llm_config
from movie_rag.llm_contract import RetrievalPlan
from movie_rag.llm_providers import (
    GeminiProvider,
    MissingAPIKeyError,
    OpenAIProvider,
)


PLAN_JSON = RetrievalPlan(
    action="search",
    detected_language="it",
    semantic_query="film sulla memoria",
    semantic_query_language="it",
    rewrite_strategy="none",
    reason="Richiesta sufficientemente specifica.",
).model_dump_json()


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=PLAN_JSON,
            response_id="gemini-request",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=8,
                total_token_count=20,
            ),
        )


class FakeOpenAIResponses:
    def __init__(self):
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=RetrievalPlan.model_validate_json(PLAN_JSON),
            output_text=PLAN_JSON,
            id="openai-request",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=7,
                total_tokens=17,
            ),
        )


@pytest.mark.anyio
async def test_gemini_adapter_normalizes_structured_output_and_trace():
    models = FakeGeminiModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    profile = load_llm_config().get_llm_profile("gemini_free")
    provider = GeminiProvider(profile, client=client)

    result = await provider.generate_structured(
        agent="planner",
        prompt_version="planner_v1",
        system_prompt="system",
        user_prompt="user",
        output_schema=RetrievalPlan,
    )

    assert result.value.semantic_query == "film sulla memoria"
    assert result.trace.provider == "gemini"
    assert result.trace.usage.total_tokens == 20
    assert models.calls[0]["model"] == profile.model
    assert models.calls[0]["config"]["response_mime_type"] == "application/json"
    assert "response_json_schema" in models.calls[0]["config"]


@pytest.mark.anyio
async def test_openai_adapter_normalizes_structured_output_and_trace():
    responses = FakeOpenAIResponses()
    client = SimpleNamespace(responses=responses)
    profile = load_llm_config().get_llm_profile("openai_default")
    provider = OpenAIProvider(profile, client=client)

    result = await provider.generate_structured(
        agent="planner",
        prompt_version="planner_v1",
        system_prompt="system",
        user_prompt="user",
        output_schema=RetrievalPlan,
    )

    assert result.value.detected_language == "it"
    assert result.trace.provider == "openai"
    assert result.trace.usage.total_tokens == 17
    assert responses.calls[0]["text_format"] is RetrievalPlan
    assert responses.calls[0]["model"] == profile.model
    assert responses.calls[0]["instructions"] == "system"
    assert responses.calls[0]["input"] == "user"


def test_missing_key_fails_before_importing_optional_sdk(monkeypatch):
    profile = load_llm_config().get_llm_profile("gemini_free")
    monkeypatch.delenv(profile.api_key_env, raising=False)

    with pytest.raises(MissingAPIKeyError, match=profile.api_key_env):
        GeminiProvider(profile)
