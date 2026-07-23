"""Small OpenAI-compatible client boundary used by all runtime roles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import ProviderSettings


class ChatClient(Protocol):
    def complete(self, settings: ProviderSettings, messages: list[dict[str, str]]) -> str: ...

    def embed(self, settings: ProviderSettings, texts: list[str]) -> list[list[float]]: ...


@dataclass
class OpenAICompatibleClient:
    """No key, prompt, or response body is persisted by this client."""

    def complete(self, settings: ProviderSettings, messages: list[dict[str, str]]) -> str:
        from openai import OpenAI

        response = OpenAI(api_key=settings.api_key, base_url=settings.base_url).chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        return response.choices[0].message.content or ""

    def embed(self, settings: ProviderSettings, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        response = OpenAI(api_key=settings.api_key, base_url=settings.base_url).embeddings.create(
            model=settings.model, input=texts
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]
