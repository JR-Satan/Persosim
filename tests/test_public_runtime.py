from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adapters.mem0 import Mem0Adapter
from src.adapters.naive_rag import NaiveRAGAdapter
from src.dataset import load_dataset
from src.memory import MemoryWrite
from src.runner import LifelongRunConfig, run_lifelong
from src.config import load_settings


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_LIFELINE = REPO_ROOT / "data" / "20user_lifeline.json"


class FakeEmbedder:
    def embed(self, texts):
        return [[float(len(text)), float(sum(map(ord, text)) % 97 + 1)] for text in texts]


class FakeClient:
    def complete(self, settings, messages):
        system = messages[0]["content"]
        if "simulate" in system:
            return "Please follow the visible request."
        if "strict evaluator" in system:
            return '{"pass": true, "failure_flags": []}'
        return "I will provide the requested current plan."

    def embed(self, settings, texts):
        return FakeEmbedder().embed(texts)


def test_released_research_data_is_valid():
    personas = load_dataset(RESEARCH_LIFELINE)
    assert len(personas) == 20
    assert all(len(persona.tasks) == 50 for persona in personas)


def test_naive_rag_returns_recent_and_one_older():
    adapter = NaiveRAGAdapter("demo_001", FakeEmbedder())
    for number in range(1, 4):
        adapter.write_task(MemoryWrite("demo_001", f"task_{number:03d}", ({"role": "user", "content": f"item {number}"},)))
    found = adapter.retrieve("item 1", 2)
    assert [candidate.retrieval_kind for candidate in found] == ["recent", "semantic"]
    assert found[0].source_task_id == "task_003"
    assert found[1].source_task_id != "task_003"


def test_mem0_adapter_requires_local_provenance(tmp_path):
    class Mem0Fake:
        def add(self, *args, **kwargs):
            return {"results": []}

        def search(self, **kwargs):
            return {"results": [{"id": "x", "memory": "memory", "user_id": "other", "metadata": {"task_id": "task_001"}}]}

    adapter = Mem0Adapter("demo_001", "x", "https://example.invalid/v1", "m", "e", tmp_path, client=Mem0Fake())
    with pytest.raises(ValueError, match="provenance"):
        adapter.retrieve("query", 1)


def test_runner_writes_isolated_artifacts(tmp_path):
    config = tmp_path / "api.local.md"
    config.write_text("\n".join(
        f"[{role}]\nmodel = \"fake\"\napi_key = \"x\"\nbase_url = \"https://example.invalid/v1\""
        for role in ("tested_agent", "user_simulator", "evaluator", "embedding")
    ), encoding="utf-8")
    summary = run_lifelong(LifelongRunConfig(
        data_path=RESEARCH_LIFELINE, output_root=tmp_path / "out", run_id="test",
        persona_id="cap_002", task_limit=2, config_path=config,
    ), client=FakeClient())
    assert summary["passed"] == 2
    root = tmp_path / "out" / "naive_rag" / "test" / "cap_002"
    assert len(root.joinpath("task_results.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(root.joinpath("summary.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_server_style_config_routes_pool_and_evaluator_without_exposing_keys(tmp_path):
    config = tmp_path / "api.local.md"
    config.write_text(
        "[generation]\nmodel = \"gpt-4o-mini\"\napi_key = \"generation-key\"\nbase_url = \"https://generation.invalid/v1\"\n"
        "[chatanywhere_pool]\nkey_01 = \"pool-key\"\nbase_url = \"https://pool.invalid/v1\"\n"
        "[deepseek]\napi_key = \"judge-key\"\nbase_url = \"https://judge.invalid/v1\"\n",
        encoding="utf-8",
    )
    assert load_settings(config, "tested_agent", model="deepseek-v4-flash").base_url == "https://pool.invalid/v1"
    assert load_settings(config, "user_simulator", model="gpt-4o-mini").base_url == "https://pool.invalid/v1"
    assert load_settings(config, "evaluator", model="deepseek-v4-pro").base_url == "https://judge.invalid/v1"


def test_simple_global_config_shares_one_key_across_roles(tmp_path):
    config = tmp_path / "api.md"
    config.write_text(
        "[global]\napi_key = \"one-key\"\nbase_url = \"https://provider.invalid/v1\"\n"
        "[tested_agent]\nmodel = \"agent-model\"\n"
        "[user_simulator]\nmodel = \"simulator-model\"\n",
        encoding="utf-8",
    )
    agent = load_settings(config, "tested_agent")
    simulator = load_settings(config, "user_simulator")
    assert agent.api_key == simulator.api_key == "one-key"
    assert agent.base_url == simulator.base_url == "https://provider.invalid/v1"
    assert (agent.model, simulator.model) == ("agent-model", "simulator-model")
