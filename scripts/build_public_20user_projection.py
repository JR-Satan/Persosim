"""Build the clean 20-user public projection from the clean 100-user release.

The existing 20-user file supplies only the canonical persona order.  Task
content is always selected from the 100-user public projection, which removes
legacy campaign/provenance metadata and includes the repaired sed_003 text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("lifelines"), list):
        raise ValueError(f"{path}: expected an object with a lifelines array")
    return payload


def lifeline_index(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in payload["lifelines"]:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: invalid lifeline")
        persona_id = row.get("persona_id")
        if not isinstance(persona_id, str) or not persona_id or persona_id in indexed:
            raise ValueError(f"{label}: lifelines need unique non-empty persona_id values")
        indexed[persona_id] = row
    return indexed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


parser = argparse.ArgumentParser(description="Regenerate the clean public 20-user aggregate pair.")
parser.add_argument("--persona-order", type=Path, default=Path("data/20user_lifeline.json"))
parser.add_argument("--runtime-source", type=Path, default=Path("data/100user_lifeline.json"))
parser.add_argument("--eval-source", type=Path, default=Path("data/100user_eval.json"))
parser.add_argument("--runtime-output", type=Path, default=Path("data/20user_lifeline.json"))
parser.add_argument("--eval-output", type=Path, default=Path("data/20user_eval.json"))
args = parser.parse_args()

order_payload = load_json(args.persona_order)
runtime_source = load_json(args.runtime_source)
eval_source = load_json(args.eval_source)
persona_order = [row.get("persona_id") for row in order_payload["lifelines"]]
if len(persona_order) != 20 or not all(isinstance(persona_id, str) and persona_id for persona_id in persona_order):
    raise ValueError("persona-order must contain exactly 20 non-empty persona IDs")
if len(set(persona_order)) != 20:
    raise ValueError("persona-order contains duplicate persona IDs")

runtime_index = lifeline_index(runtime_source, "runtime source")
eval_index = lifeline_index(eval_source, "evaluator source")
missing = [persona_id for persona_id in persona_order if persona_id not in runtime_index or persona_id not in eval_index]
if missing:
    raise ValueError(f"source is missing personas: {', '.join(missing)}")

runtime_projection = {"lifelines": [runtime_index[persona_id] for persona_id in persona_order]}
eval_projection = {"lifelines": [eval_index[persona_id] for persona_id in persona_order]}
for label, projection in (("runtime", runtime_projection), ("evaluator", eval_projection)):
    text = json.dumps(projection, ensure_ascii=False)
    if "private private" in text:
        raise ValueError(f"{label} projection still contains malformed 'private private' text")

write_json(args.runtime_output, runtime_projection)
write_json(args.eval_output, eval_projection)
print({"status": "written", "personas": len(persona_order), "runtime_tasks": sum(len(row["tasks"]) for row in runtime_projection["lifelines"]), "eval_tasks": sum(len(row["tasks"]) for row in eval_projection["lifelines"])})
