from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import load_dataset


parser = argparse.ArgumentParser(description="Validate a PersonSim public-lifeline dataset without model calls.")
parser.add_argument("--data", type=Path, default=Path("data/20user_lifeline.json"))
args = parser.parse_args()
personas = load_dataset(args.data)
print({"status": "valid", "personas": len(personas), "tasks": sum(len(persona.tasks) for persona in personas)})
