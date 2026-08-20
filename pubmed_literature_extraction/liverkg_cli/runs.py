from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import project_root, runs_dir


@dataclass
class RunSpec:
    run_id: str
    profile: str = "quality"
    input_path: str = ""
    normalized_input_path: str = ""
    output_dir: str = "extraction_output"
    limit: int = 50
    dry_run: bool = True
    write_neo4j: bool = False
    max_workers: int = 5
    extraction_inner_max_workers: int = 1
    model_id: str = ""
    api_base: str = ""
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_database: str = ""
    ncbi_email: str = ""
    ncbi_tool: str = "liverkg"
    second_llm_enabled: bool = False
    second_llm_model_id: str = ""
    second_llm_api_base: str = ""
    second_llm_mode: str = "conditional"
    aux_primary_model: str = "deepseek-v4-flash"
    aux_critic_model: str = "qwen3.6-flash"
    rule_bundle: str = ""
    conformal_calibration: str = ""
    agent_args: dict[str, Any] = field(default_factory=dict)
    input_manifest: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    project_root: str = field(default_factory=lambda: str(project_root()))


def run_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def save_run_spec(spec: RunSpec) -> Path:
    path = run_dir(spec.run_id) / "runspec.json"
    atomic_write_json(path, asdict(spec))
    return path


def load_run_spec(run_id: str) -> RunSpec:
    path = run_dir(run_id) / "runspec.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunSpec(**data)


def update_status(run_id: str, **updates: Any) -> None:
    path = run_dir(run_id) / "status.json"
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(updates)
    current["updated_at"] = time.time()
    atomic_write_json(path, current)


def read_status(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / "status.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs() -> list[dict[str, Any]]:
    base = runs_dir()
    if not base.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        status_path = path / "status.json"
        spec_path = path / "runspec.json"
        if not status_path.exists() or not spec_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append({"run_id": path.name, "status": status, "spec": spec})
    return items


def latest_run_id() -> str | None:
    runs = list_runs()
    return runs[0]["run_id"] if runs else None


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_run(run_id: str) -> None:
    status = read_status(run_id)
    pid = int(status.get("pid") or 0)
    if pid and is_pid_running(pid):
        os.kill(pid, signal.SIGTERM)
        update_status(run_id, state="stopping")
