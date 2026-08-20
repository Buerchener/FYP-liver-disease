from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from .paths import project_config_path, user_config_path


PROFILES: dict[str, dict[str, Any]] = {
    "quality": {
        "execution_mode": "agent-v2",
        "agent_budget_profile": "quality",
        "rule_memory_mode": "active",
        "evidence_entailment_mode": "active",
        "risk_router_mode": "active",
        "pair_classifier_mode": "active",
        "relation_authority": "unified-active",
        "max_workers": 5,
        "extraction_inner_max_workers": 1,
        "extraction_cache_mode": "persistent",
    },
    "balanced": {
        "execution_mode": "agent-v2-shadow",
        "agent_budget_profile": "balanced",
        "rule_memory_mode": "shadow",
        "evidence_entailment_mode": "shadow",
        "risk_router_mode": "shadow",
        "pair_classifier_mode": "shadow",
        "relation_authority": "unified-shadow",
        "max_workers": 5,
        "extraction_inner_max_workers": 1,
        "extraction_cache_mode": "persistent",
    },
    "speed": {
        "execution_mode": "agent-v2-shadow",
        "agent_budget_profile": "speed",
        "rule_memory_mode": "shadow",
        "evidence_entailment_mode": "off",
        "risk_router_mode": "shadow",
        "pair_classifier_mode": "shadow",
        "relation_authority": "legacy",
        "max_workers": 5,
        "extraction_inner_max_workers": 1,
        "extraction_cache_mode": "memory",
    },
    "legacy": {
        "execution_mode": "legacy",
        "agent_budget_profile": "quality",
        "rule_memory_mode": "off",
        "evidence_entailment_mode": "off",
        "risk_router_mode": "off",
        "pair_classifier_mode": "shadow",
        "relation_authority": "legacy",
        "max_workers": 5,
        "extraction_inner_max_workers": 2,
        "extraction_cache_mode": "memory",
    },
}


ENV_MAP = {
    "model_id": "GEMINI_MODEL",
    "api_base": "GEMINI_API_BASE",
    "neo4j_uri": "NEO4J_URI",
    "neo4j_user": "NEO4J_USER",
    "neo4j_database": "NEO4J_DATABASE",
    "default_profile": "LIVERKG_PROFILE",
    "ncbi_email": "NCBI_EMAIL",
    "ncbi_tool": "NCBI_TOOL",
    "second_llm_model_id": "SECOND_LLM_MODEL_ID",
    "second_llm_api_base": "SECOND_LLM_API_BASE",
    "aux_critic_model": "AUX_CRITIC_MODEL",
    "qwen_api_base": "ALIYUN_MAAS_API_BASE",
    "rule_bundle": "RULE_BUNDLE",
    "conformal_calibration": "CONFORMAL_CALIBRATION",
}


@dataclass
class LiverKGConfig:
    model_id: str = "[L]gemini-3-flash-preview"
    api_base: str = "https://bboluo.com/v1"
    default_profile: str = "quality"
    output_dir: str = "extraction_output"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_database: str = "neo4j"
    ncbi_email: str = ""
    ncbi_tool: str = "liverkg"
    second_llm_model_id: str = "deepseek-v4-flash"
    second_llm_api_base: str = "https://api.deepseek.com"
    aux_critic_model: str = "qwen3.6-flash"
    qwen_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rule_bundle: str = ""
    conformal_calibration: str = ""
    language: str = "auto"
    profiles: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(PROFILES))

    def to_toml_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["profiles"] = self.profiles
        return data


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(cli_overrides: dict[str, Any] | None = None) -> LiverKGConfig:
    data = LiverKGConfig().to_toml_dict()
    data = _merge(data, _load_toml(user_config_path()))
    data = _merge(data, _load_toml(project_config_path()))
    env_update = {
        key: os.environ[value]
        for key, value in ENV_MAP.items()
        if os.environ.get(value)
    }
    data = _merge(data, env_update)
    if cli_overrides:
        data = _merge(data, {k: v for k, v in cli_overrides.items() if v not in (None, "")})
    return LiverKGConfig(**{k: v for k, v in data.items() if k in LiverKGConfig.__dataclass_fields__})


def write_user_config(config: LiverKGConfig) -> Path:
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(config.to_toml_dict()), encoding="utf-8")
    return path


def set_config_value(key: str, value: Any) -> Path:
    config = load_config()
    if key not in LiverKGConfig.__dataclass_fields__:
        raise KeyError(key)
    setattr(config, key, value)
    return write_user_config(config)
