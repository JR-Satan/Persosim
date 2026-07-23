"""Local, non-versioned API configuration loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderSettings:
    model: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int


def _parse(path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"global": {}}
    section = "global"
    if not path.is_file():
        return sections
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"\[([A-Za-z0-9_.-]+)\]", line)
        if match:
            section = match.group(1)
            sections.setdefault(section, {})
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", line)
        if match:
            value = match.group(2).split("#", 1)[0].strip().strip("\"'")
            sections[section][match.group(1)] = value
    return sections


def load_settings(config_path: Path, role: str, *, model: str | None = None) -> ProviderSettings:
    """Load one role without exposing its secret in logs or artifacts."""
    sections = _parse(config_path)
    # The public template uses role sections. The existing server configuration
    # uses a shared `[generation]` route plus a `[deepseek]` route; supporting
    # both permits one private config to remain outside this repository.
    explicit_role = sections.get(role, {})
    values = {
        **sections.get("global", {}),
        **sections.get("generation", {}),
        **explicit_role,
    }
    requested_model = model or values.get("model")
    if not requested_model:
        raise ValueError(f"{role}: a model is required")
    # Existing PersonSim server configs expose a five-key ChatAnywhere relay
    # pool plus an official DeepSeek evaluator route. Use relay slot 1 for this
    # single-user public runner; callers that need concurrent slots should pass
    # an explicit role section or run a scheduler above this package.
    pool = sections.get("chatanywhere_pool", {})
    if not explicit_role and role != "evaluator" and pool.get("key_01") and pool.get("base_url"):
        values.update({"api_key": pool["key_01"], "base_url": pool["base_url"]})
    if role == "evaluator" and requested_model.lower().startswith("deepseek"):
        values.update(sections.get("deepseek", {}))
    values.update(sections.get("model." + requested_model, {}))
    key = values.get("api_key") or os.getenv("PERSONSIM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = values.get("base_url") or os.getenv("PERSONSIM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not key or key.startswith("REPLACE_"):
        raise RuntimeError(f"{role}: missing API key; copy configs/api.example.md to api.md")
    if not base_url or base_url.startswith("REPLACE_"):
        raise RuntimeError(f"{role}: missing base URL")
    return ProviderSettings(
        model=requested_model,
        api_key=key,
        base_url=base_url,
        temperature=float(values.get("temperature", "0")),
        max_tokens=int(values.get("max_tokens", "1024")),
    )
