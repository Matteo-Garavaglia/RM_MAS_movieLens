"""Provider adapters that normalize Gemini and OpenAI structured generation.

Optional SDKs are imported only when their provider is instantiated.  Importing
``movie_rag`` therefore remains possible for retrieval-only installations.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from .llm_config import LLMConfigCatalog, LLMProviderProfile
from .llm_contract import LLMCallTrace, LLMUsage


T = TypeVar("T", bound=BaseModel)


class LLMProviderError(RuntimeError):
    """Base exception for sanitized provider and response failures."""


class MissingLLMDependencyError(LLMProviderError):
    """Raised when a selected provider SDK was not installed."""


class MissingAPIKeyError(LLMProviderError):
    """Raised when the configured key environment variable is absent."""


class EmptyLLMResponseError(LLMProviderError):
    """Raised when a provider completed without usable model output."""


@dataclass(frozen=True)
class TextGeneration:
    text: str
    trace: LLMCallTrace


@dataclass(frozen=True)
class StructuredGeneration(Generic[T]):
    value: T
    trace: LLMCallTrace


class LLMProvider(Protocol):
    """Common subset used by agents, independent of any vendor SDK."""

    profile: LLMProviderProfile

    async def generate_text(
        self,
        *,
        agent: str,
        prompt_version: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> TextGeneration: ...

    async def generate_structured(
        self,
        *,
        agent: str,
        prompt_version: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredGeneration[T]: ...


async def _await_with_timeout(value: Any, timeout_seconds: float) -> Any:
    async def resolve() -> Any:
        return await value if inspect.isawaitable(value) else value

    return await asyncio.wait_for(resolve(), timeout=timeout_seconds)


def _attribute(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _integer_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _gemini_usage(response: Any) -> LLMUsage:
    usage = getattr(response, "usage_metadata", None)
    return LLMUsage(
        input_tokens=_integer_or_none(
            _attribute(usage, "prompt_token_count", "prompt_tokens")
        ),
        output_tokens=_integer_or_none(
            _attribute(usage, "candidates_token_count", "output_tokens")
        ),
        total_tokens=_integer_or_none(
            _attribute(usage, "total_token_count", "total_tokens")
        ),
    )


def _openai_usage(response: Any) -> LLMUsage:
    usage = getattr(response, "usage", None)
    return LLMUsage(
        input_tokens=_integer_or_none(_attribute(usage, "input_tokens")),
        output_tokens=_integer_or_none(_attribute(usage, "output_tokens")),
        total_tokens=_integer_or_none(_attribute(usage, "total_tokens")),
    )


class _BaseProvider:
    provider_name: str

    def __init__(self, profile: LLMProviderProfile) -> None:
        self.profile = profile

    def _temperature(self, override: float | None) -> float | None:
        return self.profile.temperature if override is None else override

    def _max_tokens(self, override: int | None) -> int:
        return self.profile.max_output_tokens if override is None else override

    def _trace(
        self,
        *,
        agent: str,
        prompt_version: str,
        started_at: datetime,
        started_counter: float,
        usage: LLMUsage,
        request_id: Any,
    ) -> LLMCallTrace:
        return LLMCallTrace(
            agent=agent,
            provider=self.provider_name,  # type: ignore[arg-type]
            profile_id=self.profile.profile_id,
            model=self.profile.model,
            prompt_version=prompt_version,
            started_at=started_at,
            duration_ms=(perf_counter() - started_counter) * 1000.0,
            usage=usage,
            request_id=None if request_id is None else str(request_id),
        )


class GeminiProvider(_BaseProvider):
    """Gemini Developer API adapter using ``google-genai``.

    A client can be injected for tests.  Production construction reads only the
    environment variable named by the selected profile; the key is never
    retained in traces or configuration output.
    """

    provider_name = "gemini"

    def __init__(
        self,
        profile: LLMProviderProfile,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(profile)
        if profile.provider != "gemini":
            raise ValueError("GeminiProvider requires a gemini profile")
        if client is None:
            key = api_key or os.getenv(profile.api_key_env)
            if not key:
                raise MissingAPIKeyError(
                    f"Set {profile.api_key_env} before using profile "
                    f"{profile.profile_id!r}"
                )
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on optional SDK
                raise MissingLLMDependencyError(
                    'Install the Gemini adapter with `pip install -e ".[llm-gemini]"`'
                ) from exc
            client = genai.Client(api_key=key)
        self._client = client

    @property
    def _models(self) -> Any:
        asynchronous = getattr(self._client, "aio", None)
        return getattr(asynchronous, "models", None) or self._client.models

    def _config(
        self,
        *,
        system_prompt: str,
        temperature: float | None,
        max_output_tokens: int,
        output_schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            config["temperature"] = temperature
        if output_schema is not None:
            config.update(
                {
                    "response_mime_type": "application/json",
                    "response_json_schema": output_schema.model_json_schema(),
                }
            )
        return config

    async def generate_text(
        self,
        *,
        agent: str,
        prompt_version: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> TextGeneration:
        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        call = self._models.generate_content(
            model=self.profile.model,
            contents=user_prompt,
            config=self._config(
                system_prompt=system_prompt,
                temperature=self._temperature(temperature),
                max_output_tokens=self._max_tokens(max_output_tokens),
            ),
        )
        response = await _await_with_timeout(call, self.profile.timeout_seconds)
        # await asyncio.sleep(0.10)
        text = str(getattr(response, "text", "") or "").strip()
        # await asyncio.sleep(0.10)
        # print("Gemini model response", response)
        if not text:
            raise EmptyLLMResponseError(
                f"Gemini model {self.profile.model!r} returned no text"
            )
        return TextGeneration(
            text=text,
            trace=self._trace(
                agent=agent,
                prompt_version=prompt_version,
                started_at=started_at,
                started_counter=started_counter,
                usage=_gemini_usage(response),
                request_id=_attribute(response, "response_id", "id"),
            ),
        )

    async def generate_structured(
        self,
        *,
        agent: str,
        prompt_version: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredGeneration[T]:
        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        call = self._models.generate_content(
            model=self.profile.model,
            contents=user_prompt,
            config=self._config(
                system_prompt=system_prompt,
                temperature=self._temperature(temperature),
                max_output_tokens=self._max_tokens(max_output_tokens),
                output_schema=output_schema,
            ),
        )
        response = await _await_with_timeout(call, self.profile.timeout_seconds)
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise EmptyLLMResponseError(
                f"Gemini model {self.profile.model!r} returned no structured output"
            )
        value = output_schema.model_validate_json(text)
        return StructuredGeneration(
            value=value,
            trace=self._trace(
                agent=agent,
                prompt_version=prompt_version,
                started_at=started_at,
                started_counter=started_counter,
                usage=_gemini_usage(response),
                request_id=_attribute(response, "response_id", "id"),
            ),
        )


class OpenAIProvider(_BaseProvider):
    """OpenAI Responses API adapter with native Pydantic parsing."""

    provider_name = "openai"

    def __init__(
        self,
        profile: LLMProviderProfile,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(profile)
        if profile.provider != "openai":
            raise ValueError("OpenAIProvider requires an openai profile")
        if client is None:
            key = api_key or os.getenv(profile.api_key_env)
            if not key:
                raise MissingAPIKeyError(
                    f"Set {profile.api_key_env} before using profile "
                    f"{profile.profile_id!r}"
                )
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional SDK
                raise MissingLLMDependencyError(
                    'Install the OpenAI adapter with `pip install -e ".[llm-openai]"`'
                ) from exc
            base_url = (
                os.getenv(profile.base_url_env)
                if profile.base_url_env is not None
                else None
            )
            kwargs: dict[str, Any] = {
                "api_key": key,
                "timeout": profile.timeout_seconds,
            }
            if base_url:
                kwargs["base_url"] = base_url
            client = AsyncOpenAI(**kwargs)
        self._client = client

    def _request_kwargs(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    async def generate_text(
        self,
        *,
        agent: str,
        prompt_version: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> TextGeneration:
        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        call = self._client.responses.create(
            **self._request_kwargs(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature(temperature),
                max_output_tokens=self._max_tokens(max_output_tokens),
            )
        )
        response = await _await_with_timeout(call, self.profile.timeout_seconds)
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise EmptyLLMResponseError(
                f"OpenAI model {self.profile.model!r} returned no text"
            )
        return TextGeneration(
            text=text,
            trace=self._trace(
                agent=agent,
                prompt_version=prompt_version,
                started_at=started_at,
                started_counter=started_counter,
                usage=_openai_usage(response),
                request_id=getattr(response, "id", None),
            ),
        )

    async def generate_structured(
        self,
        *,
        agent: str,
        prompt_version: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredGeneration[T]:
        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        call = self._client.responses.parse(
            **self._request_kwargs(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature(temperature),
                max_output_tokens=self._max_tokens(max_output_tokens),
            ),
            text_format=output_schema,
        )
        response = await _await_with_timeout(call, self.profile.timeout_seconds)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raw = str(getattr(response, "output_text", "") or "").strip()
            if not raw:
                raise EmptyLLMResponseError(
                    f"OpenAI model {self.profile.model!r} returned no structured output"
                )
            parsed = output_schema.model_validate_json(raw)
        elif not isinstance(parsed, output_schema):
            parsed = output_schema.model_validate(parsed)
        return StructuredGeneration(
            value=parsed,
            trace=self._trace(
                agent=agent,
                prompt_version=prompt_version,
                started_at=started_at,
                started_counter=started_counter,
                usage=_openai_usage(response),
                request_id=getattr(response, "id", None),
            ),
        )


def create_llm_provider(
    profile: LLMProviderProfile,
    *,
    api_key: str | None = None,
    client: Any | None = None,
) -> LLMProvider:
    """Construct the configured adapter without exposing credentials to agents."""

    if profile.provider == "gemini":
        return GeminiProvider(profile, api_key=api_key, client=client)
    if profile.provider == "openai":
        return OpenAIProvider(profile, api_key=api_key, client=client)
    raise ValueError(f"Unsupported provider: {profile.provider!r}")


class LLMProviderRegistry:
    """Lazily build and reuse one provider client per versioned profile."""

    def __init__(
        self,
        catalog: LLMConfigCatalog,
        *,
        client_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.catalog = catalog
        self.client_overrides = dict(client_overrides or {})
        self._providers: dict[str, LLMProvider] = {}

    def get(self, profile_id: str) -> LLMProvider:
        if profile_id not in self._providers:
            profile = self.catalog.get_llm_profile(profile_id)
            self._providers[profile_id] = create_llm_provider(
                profile,
                client=self.client_overrides.get(profile_id),
            )
        return self._providers[profile_id]
