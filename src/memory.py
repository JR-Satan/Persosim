"""Memory adapter contract and shared fairness controls."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


MEMORY_TOKEN_BUDGET = 2048
NAIVE_RAG_POLICY_ID = "recent_plus_one_semantic_v1"


@dataclass(frozen=True)
class MemoryWrite:
    persona_id: str
    task_id: str
    turns: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    source_task_id: str
    source_user_id: str
    text: str
    score: float | None
    retrieval_kind: str


class MemoryBackend(Protocol):
    """Minimal contract for any personal-memory implementation.

    An adapter owns exactly one target persona.  It may never return a record
    from another persona, and every candidate must retain source-task provenance.
    """

    system_name: str

    def write_task(self, write: MemoryWrite) -> None: ...

    def retrieve(self, query: str, limit: int) -> list[MemoryCandidate]: ...

    def close(self) -> None: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def render_turns(turns: tuple[dict[str, str], ...] | list[dict[str, str]]) -> str:
    return "\n".join(
        f"{turn['role'].title()}: {turn['content'].strip()}"
        for turn in turns
        if turn.get("role") in {"user", "assistant"} and turn.get("content", "").strip()
    )


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return 0.0 if not left_norm or not right_norm else sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def cap_visible_memory(candidates: list[MemoryCandidate], *, token_budget: int = MEMORY_TOKEN_BUDGET) -> list[MemoryCandidate]:
    """Apply one shared agent-visible budget after native retrieval.

    `tiktoken` is used when available.  The conservative character fallback
    keeps tests dependency-light without silently exceeding the requested budget.
    """
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("o200k_base")
        length = lambda value: len(encoder.encode(value))
    except Exception:  # pragma: no cover - optional fallback
        length = lambda value: max(1, (len(value) + 3) // 4)
    selected: list[MemoryCandidate] = []
    used = 0
    for candidate in candidates:
        size = length(candidate.text)
        if size <= token_budget - used:
            selected.append(candidate)
            used += size
    return selected
