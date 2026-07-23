from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


parser = argparse.ArgumentParser(description="Validate released PersonSim persona-instance metadata without model calls.")
parser.add_argument("--data", type=Path, default=Path("data/persona_instances_5x20.jsonl"))
args = parser.parse_args()
rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
ids = [str(row.get("instance_id") or "") for row in rows]
families = Counter(str(row.get("family_id") or "") for row in rows)
if not rows or any(not value for value in ids) or len(ids) != len(set(ids)):
    raise SystemExit("instance_id values must be non-empty and unique")
if len(families) != 5 or set(families.values()) != {20}:
    raise SystemExit("expected exactly five persona families with 20 instances each")
if any(row.get("schema_version") != "1.0" for row in rows):
    raise SystemExit("unexpected persona-instance schema version")
print({"status": "valid", "records": len(rows), "families": dict(sorted(families.items()))})
