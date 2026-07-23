from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import load_dataset
from src.runner import LifelongRunConfig, run_lifelong


DEFAULT_MODELS = ("deepseek-v4-flash", "gpt-4o-mini", "gemini-3-flash-preview", "claude-haiku-4-5-20251001")
parser = argparse.ArgumentParser(description="Run the same Lifelong protocol over a tested-model axis.")
parser.add_argument("--data", type=Path, default=Path("data/20user_lifeline.json"))
parser.add_argument("--output-root", type=Path, default=Path("outputs/model_axis"))
parser.add_argument("--run-id", required=True)
parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
parser.add_argument("--persona-id", action="append")
parser.add_argument("--memory-system", choices=("naive_rag", "mem0"), default="naive_rag")
parser.add_argument("--task-limit", type=int)
parser.add_argument("--config", type=Path, default=Path("configs/api.md"))
parser.add_argument("--user-simulator-model", default="gpt-4o-mini")
parser.add_argument("--evaluator-model", default="deepseek-v4-pro")
parser.add_argument("--embedding-model", default="text-embedding-3-small")
parser.add_argument("--memory-model", default="gpt-4o-mini")
args = parser.parse_args()
available = {persona.persona_id for persona in load_dataset(args.data)}
personas = args.persona_id or sorted(available)
if unknown := set(personas) - available:
    raise SystemExit("unknown personas: " + ", ".join(sorted(unknown)))
for model in args.model or DEFAULT_MODELS:
    slug = model.replace("-", "_")
    for persona_id in personas:
        summary = run_lifelong(LifelongRunConfig(
            data_path=args.data, output_root=args.output_root / slug, run_id=args.run_id,
            persona_id=persona_id, memory_system=args.memory_system, task_limit=args.task_limit,
            config_path=args.config, tested_agent_model=model,
            user_simulator_model=args.user_simulator_model, evaluator_model=args.evaluator_model,
            embedding_model=args.embedding_model,
            memory_model=args.memory_model,
        ))
        print({"model": model, "persona_id": persona_id, **summary})
