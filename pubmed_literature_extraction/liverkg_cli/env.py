from __future__ import annotations

import os
import shlex
from pathlib import Path

from .paths import project_root


LOCAL_ENV_FILES = (
    ".env",
    "workstreams/literature_hmdb_kegg/.env",
    "workstreams/literature_hmdb_kegg/active-gemini.env",
)


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        try:
            parts = shlex.split(value, comments=False, posix=True)
            value = parts[0] if parts else ""
        except ValueError:
            value = value.strip("'\"")
        values[key] = value
    return values


def load_project_env(root: Path | None = None) -> dict[str, str]:
    base = root or project_root()
    values: dict[str, str] = {}
    for rel in LOCAL_ENV_FILES:
        values.update(parse_env_file(base / rel))
    return values


def merged_runtime_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in load_project_env().items():
        env.setdefault(key, value)
    if extra:
        for key, value in extra.items():
            if value:
                env[key] = value
    return env
