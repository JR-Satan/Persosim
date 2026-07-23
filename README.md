# PersonSim

PersonSim is a compact runtime for evaluating memory-augmented personal
assistants on ordered, on-policy user lifelines. This public repository starts
with two deliberately comparable memory paths:

- `naive_rag` — the reference baseline: the latest completed task transcript
  plus one semantically retrieved older transcript.
- `mem0` — an optional integration of [Mem0](https://github.com/mem0ai/mem0)
  through the same adapter contract, showing how to add another memory system.

The repository includes the released 20-persona, 50-task-per-persona Lifelong
aggregate plus `persona_instances_5x20.jsonl` metadata (100 instances, five
balanced families).

## Design constraints

- A persona has one sequential, on-policy trajectory per condition.
- Every memory backend is isolated to one persona and must emit source-task
  provenance for every retrieval candidate.
- The runner rejects cross-user candidates before the Tested Agent sees them.
- The Evaluator is post-hoc; its output is never inserted into memory.

## Installation

Python 3.10 or newer is required.

```bash
git clone <YOUR_REPOSITORY_URL> persosim
cd persosim
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For Mem0 support, install the optional dependency:

```bash
pip install -e '.[mem0,dev]'
```

## Configure providers

Copy the template locally and fill one OpenAI-compatible API route. The copied
file is ignored by Git.

```bash
cp configs/api.example.md configs/api.md
chmod 600 configs/api.md
```

`[global]` contains the single `base_url` and `api_key` shared by default. The
role sections select models and generation settings:

| Role | Purpose |
|---|---|
| `tested_agent` | Model being compared |
| `user_simulator` | Generates the two natural user turns per task |
| `evaluator` | Produces a strict post-hoc JSON verdict |
| `embedding` | Indexes and queries NaiveRAG transcripts |

`memory` is used only by adapters such as Mem0 and also inherits `[global]` by
default.

If one role needs another provider, add `base_url` and `api_key` only to that
role's section. Never commit `api.md`, `.env`, output folders, or provider
response logs.

## Run NaiveRAG

This runs three tasks for `cap_002`. It creates a fresh run root and refuses to
overwrite an existing one.

```bash
python scripts/run_lifelong.py \
  --run-id demo-naiverag-001 \
  --persona-id cap_002 \
  --task-limit 3 \
  --memory-system naive_rag \
  --tested-agent-model deepseek-v4-flash \
  --user-simulator-model gpt-4o-mini \
  --evaluator-model deepseek-v4-pro \
  --embedding-model text-embedding-3-small
```

## Run Mem0

Mem0 is optional and uses the same task stream, role settings, output schema,
and visible-memory budget.

```bash
pip install -e '.[mem0]'

python scripts/run_lifelong.py \
  --run-id demo-mem0-001 \
  --persona-id cap_002 \
  --memory-system mem0 \
  --task-limit 3 \
  --tested-agent-model deepseek-v4-flash \
  --user-simulator-model gpt-4o-mini \
  --evaluator-model deepseek-v4-pro \
  --embedding-model text-embedding-3-small
```

## Add another memory system

Implement `MemoryBackend` in `src/memory.py`:

```python
class MyMemory:
    system_name = "my_memory"

    def write_task(self, write: MemoryWrite) -> None: ...
    def retrieve(self, query: str, limit: int) -> list[MemoryCandidate]: ...
    def close(self) -> None: ...
```

## Data release and license

The code is MIT licensed. The released Lifelong aggregate and persona-instance
metadata are included.
