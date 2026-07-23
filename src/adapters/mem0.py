"""Optional Mem0 adapter; no upstream Mem0 code is copied into this repository."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory import MemoryCandidate, MemoryWrite, render_turns


@dataclass
class Mem0Adapter:
    persona_id: str
    api_key: str
    base_url: str
    llm_model: str
    embedding_model: str
    store_dir: Path
    client: Any | None = None
    system_name: str = "mem0"

    def __post_init__(self) -> None:
        self.store_dir = self.store_dir.resolve()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.client = self.client or self._open_client()

    def _open_client(self) -> Any:
        # Mem0 reads this flag while importing its telemetry integration. Set it
        # before importing the package so a benchmark run never depends on an
        # analytics endpoint.
        os.environ["MEM0_TELEMETRY"] = "false"
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover - checked in integration setup
            raise RuntimeError("Mem0 requires: pip install -e '.[mem0]'") from exc
        config = {
            "llm": {"provider": "openai", "config": {
                "model": self.llm_model, "api_key": self.api_key,
                "openai_base_url": self.base_url, "temperature": 0.0,
            }},
            "embedder": {"provider": "openai", "config": {
                "model": self.embedding_model, "api_key": self.api_key,
                "openai_base_url": self.base_url,
            }},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": f"personsim_{self.persona_id}", "embedding_model_dims": 1536,
                "path": str(self.store_dir / "qdrant"), "on_disk": True,
            }},
            "history_db_path": str(self.store_dir / "history.db"),
            "version": "v1.1",
        }
        return Memory.from_config(config)

    def write_task(self, write: MemoryWrite) -> None:
        if write.persona_id != self.persona_id:
            raise ValueError("Mem0 adapter cannot write another persona")
        self.client.add(list(write.turns), user_id=self.persona_id, metadata={
            "source": "personsim", "persona_id": self.persona_id, "task_id": write.task_id,
        })

    def retrieve(self, query: str, limit: int) -> list[MemoryCandidate]:
        response = self.client.search(query=query, top_k=limit, filters={"user_id": self.persona_id})
        rows = response.get("results", []) if isinstance(response, dict) else []
        candidates: list[MemoryCandidate] = []
        for row in rows:
            metadata = row.get("metadata") or {}
            source_user = str(row.get("user_id") or metadata.get("persona_id") or "")
            task_id = str(metadata.get("task_id") or "")
            text = str(row.get("memory") or "").strip()
            if source_user != self.persona_id or not task_id or not text:
                raise ValueError("Mem0 response lacks local user or task provenance")
            score = row.get("score")
            candidates.append(MemoryCandidate(
                memory_id=str(row.get("id") or task_id), source_task_id=task_id,
                source_user_id=source_user, text=text,
                score=float(score) if isinstance(score, (int, float)) else None,
                retrieval_kind="native",
            ))
        return candidates

    def close(self) -> None:
        close = getattr(getattr(getattr(self.client, "vector_store", None), "client", None), "close", None)
        if callable(close):
            close()
