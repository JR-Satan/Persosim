"""Public dataset schema for the compact runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    request: str
    attachment: str | None
    cues: tuple[str, str]
    evaluation: str


@dataclass(frozen=True)
class Persona:
    persona_id: str
    profile: str
    tasks: tuple[Task, ...]


def load_dataset(path: Path) -> tuple[Persona, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "personsim_runtime_lifeline_aggregate_v1":
        return _load_research_lifeline(path, payload)
    # Public research projections intentionally contain only ``lifelines``.
    # They are paired with a sibling evaluator file and omit internal campaign
    # and provenance metadata from the original aggregate format.
    if set(payload) == {"lifelines"}:
        return _load_research_lifeline(path, payload)
    if payload.get("schema_version") != "personsim_public_lifeline_v1":
        raise ValueError("unsupported dataset schema_version")
    personas: list[Persona] = []
    seen_personas: set[str] = set()
    for raw_persona in payload.get("personas", []):
        persona_id = str(raw_persona.get("persona_id", "")).strip()
        profile = str(raw_persona.get("profile", "")).strip()
        if not persona_id or not profile or persona_id in seen_personas:
            raise ValueError("each persona needs a unique ID and non-empty profile")
        seen_personas.add(persona_id)
        tasks: list[Task] = []
        seen_tasks: set[str] = set()
        for raw_task in raw_persona.get("tasks", []):
            task_id = str(raw_task.get("task_id", "")).strip()
            cues = raw_task.get("cues")
            if (
                not task_id
                or task_id in seen_tasks
                or not isinstance(cues, list)
                or len(cues) != 2
                or not all(isinstance(cue, str) and cue.strip() for cue in cues)
            ):
                raise ValueError(f"{persona_id}: tasks require unique IDs and two non-empty cues")
            seen_tasks.add(task_id)
            request = str(raw_task.get("request", "")).strip()
            evaluation = str(raw_task.get("evaluation", "")).strip()
            attachment = raw_task.get("attachment")
            if not request or not evaluation or (attachment is not None and not isinstance(attachment, str)):
                raise ValueError(f"{persona_id}/{task_id}: request, evaluation and attachment are invalid")
            tasks.append(Task(task_id, request, attachment, (cues[0], cues[1]), evaluation))
        if not tasks:
            raise ValueError(f"{persona_id}: at least one task is required")
        personas.append(Persona(persona_id, profile, tuple(tasks)))
    if not personas:
        raise ValueError("dataset contains no personas")
    return tuple(personas)


def _load_research_lifeline(path: Path, payload: dict[str, Any]) -> tuple[Persona, ...]:
    """Project the released two-file research aggregate into the public runner.

    The runtime file remains the sole source of user-visible content. The
    sibling evaluator aggregate contributes only post-hoc criterion text.
    """
    if not path.name.endswith("_lifeline.json"):
        raise ValueError("research lifeline filenames must end in _lifeline.json")
    eval_path = path.with_name(path.name.replace("_lifeline.json", "_eval.json"))
    if not eval_path.is_file():
        raise ValueError(f"the released research lifeline requires sibling {eval_path.name}")
    evaluation_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    evaluation_schema = evaluation_payload.get("schema_version")
    if evaluation_schema not in (None, "personsim_eval_aggregate_v1"):
        raise ValueError("unexpected research evaluator aggregate schema")
    criteria: dict[tuple[str, str], str] = {}
    for lifeline in evaluation_payload.get("lifelines", []):
        persona_id = str(lifeline.get("persona_id") or "")
        for task in lifeline.get("tasks", []):
            task_key = task.get("task_key") or {}
            task_id = str(task_key.get("task_id") or "")
            propositions = [str(item.get("proposition") or "").strip() for item in task.get("need_exact", [])]
            if persona_id and task_id and all(propositions):
                criteria[(persona_id, task_id)] = "\n".join(propositions)
    personas: list[Persona] = []
    lifelines = payload.get("lifelines", [])
    if not isinstance(lifelines, list) or not lifelines:
        raise ValueError("released lifeline must contain at least one persona")
    for lifeline in lifelines:
        persona_id = str(lifeline.get("persona_id") or "").strip()
        tasks: list[Task] = []
        for raw_task in lifeline.get("tasks", []):
            task_key = raw_task.get("task_key") or {}
            task_id = str(task_key.get("task_id") or "").strip()
            request = str((raw_task.get("request") or {}).get("text") or "").strip()
            attachment = raw_task.get("attachment") or {}
            attachment_text = attachment.get("content") if attachment.get("type") != "none" else None
            cues = tuple(str(item.get("text") or "").strip() for item in raw_task.get("cues", []))
            evaluation = criteria.get((persona_id, task_id), "")
            if not persona_id or not task_id or not request or len(cues) != 2 or not all(cues) or not evaluation:
                raise ValueError(f"invalid released task projection: {persona_id}/{task_id}")
            tasks.append(Task(task_id, request, str(attachment_text) if attachment_text is not None else None, (cues[0], cues[1]), evaluation))
        if len(tasks) != 50:
            raise ValueError(f"{persona_id}: released lifeline must contain 50 tasks")
        personas.append(Persona(persona_id, f"PersonSim persona {persona_id}", tuple(tasks)))
    if len(personas) != len({persona.persona_id for persona in personas}):
        raise ValueError("released lifeline must contain uniquely identified personas")
    return tuple(personas)
