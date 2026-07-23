from __future__ import annotations

import argparse
from pathlib import Path

from src.runner import LifelongRunConfig, run_lifelong


parser = argparse.ArgumentParser(description="Run one fresh on-policy PersonSim Lifelong path.")
parser.add_argument("--data", type=Path, default=Path("data/20user_lifeline.json"))
parser.add_argument("--output-root", type=Path, default=Path("outputs"))
parser.add_argument("--run-id", required=True)
parser.add_argument("--persona-id", required=True)
parser.add_argument("--memory-system", choices=("naive_rag", "mem0"), default="naive_rag")
parser.add_argument("--task-limit", type=int)
parser.add_argument("--config", type=Path, default=Path("configs/api.md"))
parser.add_argument("--tested-agent-model")
parser.add_argument("--user-simulator-model")
parser.add_argument("--evaluator-model")
parser.add_argument("--embedding-model")
parser.add_argument("--memory-model")
args = parser.parse_args()
summary = run_lifelong(LifelongRunConfig(
    data_path=args.data, output_root=args.output_root, run_id=args.run_id, persona_id=args.persona_id,
    memory_system=args.memory_system, task_limit=args.task_limit, config_path=args.config,
    tested_agent_model=args.tested_agent_model, user_simulator_model=args.user_simulator_model,
    evaluator_model=args.evaluator_model, embedding_model=args.embedding_model,
    memory_model=args.memory_model,
))
print(summary)
