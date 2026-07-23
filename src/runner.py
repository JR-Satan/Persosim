"""A compact on-policy Lifelong runner shared by NaiveRAG and Mem0."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adapters.mem0 import Mem0Adapter
from .adapters.naive_rag import NaiveRAGAdapter
from .client import ChatClient, OpenAICompatibleClient
from .config import ProviderSettings, load_settings
from .dataset import Persona, Task, load_dataset
from .memory import MEMORY_TOKEN_BUDGET, MemoryBackend, MemoryWrite, cap_visible_memory


SYSTEM_PROMPT = (
    "You are a personal assistant. Current user messages and attachments have highest authority. "
    "Use target-user memory only when relevant. Preserve concrete constraints, prefer the latest "
    "valid value under conflict, do not invent facts, and return only the user-facing response."
)


@dataclass(frozen=True)
class LifelongRunConfig:
    data_path: Path
    output_root: Path
    run_id: str
    persona_id: str
    memory_system: str = "naive_rag"
    task_limit: int | None = None
    config_path: Path = Path("configs/api.md")
    tested_agent_model: str | None = None
    user_simulator_model: str | None = None
    evaluator_model: str | None = None
    embedding_model: str | None = None
    memory_model: str | None = None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _select_persona(data_path: Path, persona_id: str) -> Persona:
    for persona in load_dataset(data_path):
        if persona.persona_id == persona_id:
            return persona
    raise ValueError(f"unknown persona: {persona_id}")


class _ClientEmbedder:
    def __init__(self, client: ChatClient, settings: ProviderSettings) -> None:
        self.client, self.settings = client, settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(self.settings, texts)


def _memory_backend(
    config: LifelongRunConfig,
    *, client: ChatClient,
    embedding: ProviderSettings,
    memory_llm: ProviderSettings | None = None,
) -> MemoryBackend:
    if config.memory_system == "naive_rag":
        return NaiveRAGAdapter(persona_id=config.persona_id, embedder=_ClientEmbedder(client, embedding))
    if config.memory_system == "mem0":
        if memory_llm is None:
            raise RuntimeError("Mem0 requires a memory LLM setting")
        return Mem0Adapter(
            persona_id=config.persona_id,
            api_key=memory_llm.api_key,
            base_url=memory_llm.base_url,
            llm_model=memory_llm.model,
            embedding_model=embedding.model,
            store_dir=config.output_root / config.memory_system / config.run_id / config.persona_id / "native_store",
        )
    raise ValueError("memory_system must be naive_rag or mem0")


def _messages_for_simulator(persona: Persona, task: Task, cue: str, transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You simulate the named PersonSim user. Return one natural user message only; do not mention hidden evaluation or benchmark fields."},
        {"role": "user", "content": json.dumps({
            "persona": persona.profile, "request": task.request, "attachment": task.attachment,
            "cue": cue, "transcript": transcript,
        }, ensure_ascii=False)},
    ]


def _messages_for_agent(task: Task, transcript: list[dict[str, str]], memory: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "current_task_transcript": transcript, "attachment": task.attachment,
            "target_user_memory": memory,
        }, ensure_ascii=False)},
    ]


def _messages_for_evaluator(task: Task, transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a strict evaluator. Return JSON only with boolean pass and a list of short failure_flags."},
        {"role": "user", "content": json.dumps({
            "request": task.request, "evaluation": task.evaluation, "transcript": transcript,
        }, ensure_ascii=False)},
    ]


def _parse_verdict(raw: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"pass": False, "failure_flags": ["invalid_evaluator_json"]}
    passed = parsed.get("pass")
    flags = parsed.get("failure_flags", [])
    if not isinstance(passed, bool) or not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
        return {"pass": False, "failure_flags": ["invalid_evaluator_schema"]}
    return {"pass": passed, "failure_flags": flags}


def run_lifelong(
    config: LifelongRunConfig,
    *,
    client: ChatClient | None = None,
    backend_factory: Callable[[LifelongRunConfig, ChatClient, ProviderSettings], MemoryBackend] | None = None,
) -> dict[str, object]:
    """Run a fresh, sequential path. Existing output roots are never overwritten."""
    client = client or OpenAICompatibleClient()
    persona = _select_persona(config.data_path, config.persona_id)
    tasks = persona.tasks[: config.task_limit] if config.task_limit else persona.tasks
    if not tasks:
        raise ValueError("task_limit selects no tasks")
    run_root = config.output_root / config.memory_system / config.run_id / config.persona_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_root}")
    tested = load_settings(config.config_path, "tested_agent", model=config.tested_agent_model)
    simulator = load_settings(config.config_path, "user_simulator", model=config.user_simulator_model)
    evaluator = load_settings(config.config_path, "evaluator", model=config.evaluator_model)
    embedding = load_settings(config.config_path, "embedding", model=config.embedding_model)
    memory_llm = load_settings(config.config_path, "memory", model=config.memory_model) if config.memory_system == "mem0" else None
    backend = backend_factory(config, client, embedding) if backend_factory else _memory_backend(
        config, client=client, embedding=embedding, memory_llm=memory_llm
    )
    manifest = {
        "schema_version": "personsim_public_run_v1", "created_at": _utc(), "run_id": config.run_id,
        "persona_id": persona.persona_id, "memory_system": config.memory_system,
        "models": {"tested_agent": tested.model, "user_simulator": simulator.model, "evaluator": evaluator.model, "embedding": embedding.model, "memory": memory_llm.model if memory_llm else None},
        "memory_token_budget": MEMORY_TOKEN_BUDGET,
    }
    _write_json(run_root / "run_manifest.json", manifest)
    passed = 0
    try:
        for index, task in enumerate(tasks, start=1):
            transcript: list[dict[str, str]] = []
            retrieved_ids: list[str] = []
            for cue in task.cues:
                user_text = client.complete(simulator, _messages_for_simulator(persona, task, cue, transcript)).strip()
                if not user_text:
                    raise RuntimeError("user simulator returned an empty utterance")
                transcript.append({"role": "user", "content": user_text})
                query = task.request + "\n" + user_text
                candidates = cap_visible_memory(backend.retrieve(query, limit=2))
                if any(candidate.source_user_id != persona.persona_id for candidate in candidates):
                    raise RuntimeError("memory backend returned a cross-user record")
                retrieved_ids.extend(candidate.memory_id for candidate in candidates)
                memory = [{"representation_type": candidate.retrieval_kind, "text": candidate.text} for candidate in candidates]
                answer = client.complete(tested, _messages_for_agent(task, transcript, memory)).strip()
                if not answer:
                    raise RuntimeError("tested agent returned an empty answer")
                transcript.append({"role": "assistant", "content": answer})
            verdict = _parse_verdict(client.complete(evaluator, _messages_for_evaluator(task, transcript)))
            passed += int(bool(verdict["pass"]))
            record = {"task_id": task.task_id, "task_index": index, "transcript": transcript, "retrieved_memory_ids": retrieved_ids}
            _append_jsonl(run_root / "task_results.jsonl", record)
            _append_jsonl(run_root / "evaluation_results.jsonl", {"task_id": task.task_id, "evaluation": verdict})
            _append_jsonl(run_root / "memory_records.jsonl", {"task_id": task.task_id, "source_turns": transcript})
            backend.write_task(MemoryWrite(persona.persona_id, task.task_id, tuple(transcript)))
    finally:
        backend.close()
    summary = {"status": "complete", "completed_at": _utc(), "task_count": len(tasks), "passed": passed, "pass_rate": passed / len(tasks)}
    _write_json(run_root / "summary.json", summary)
    return summary
