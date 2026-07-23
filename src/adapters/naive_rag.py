"""The public NaiveRAG baseline: newest transcript plus one older semantic hit."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..memory import Embedder, MemoryCandidate, MemoryWrite, cosine, render_turns


@dataclass
class NaiveRAGAdapter:
    persona_id: str
    embedder: Embedder
    system_name: str = "naive_rag"
    _records: list[tuple[MemoryWrite, list[float]]] = field(default_factory=list)

    def write_task(self, write: MemoryWrite) -> None:
        if write.persona_id != self.persona_id:
            raise ValueError("NaiveRAG adapter cannot write another persona")
        text = render_turns(write.turns)
        vector = self.embedder.embed([text])
        if len(vector) != 1 or not vector[0]:
            raise RuntimeError("embedder returned an invalid write vector")
        self._records.append((write, vector[0]))

    def retrieve(self, query: str, limit: int) -> list[MemoryCandidate]:
        if limit < 1 or not self._records:
            return []
        newest_write, _ = self._records[-1]
        result = [MemoryCandidate(
            memory_id=f"{newest_write.task_id}:recent",
            source_task_id=newest_write.task_id,
            source_user_id=self.persona_id,
            text=render_turns(newest_write.turns),
            score=None,
            retrieval_kind="recent",
        )]
        if limit == 1 or len(self._records) == 1:
            return result
        query_vectors = self.embedder.embed([query])
        if len(query_vectors) != 1:
            raise RuntimeError("embedder returned an invalid query vector")
        scored = [
            (cosine(query_vectors[0], vector), index, write)
            for index, (write, vector) in enumerate(self._records[:-1])
        ]
        score, _, write = max(scored, key=lambda row: (row[0], row[1]))
        result.append(MemoryCandidate(
            memory_id=f"{write.task_id}:rag",
            source_task_id=write.task_id,
            source_user_id=self.persona_id,
            text=render_turns(write.turns),
            score=round(score, 6),
            retrieval_kind="semantic",
        ))
        return result

    def close(self) -> None:
        return None
