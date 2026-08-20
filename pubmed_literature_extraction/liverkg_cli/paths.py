from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_state_dir


APP_NAME = "LiverKG"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def user_config_path() -> Path:
    override = os.environ.get("LIVERKG_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "config.toml"


def project_config_path(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        candidate = path / ".liverkg.toml"
        if candidate.exists():
            return candidate
    return None


def runs_dir() -> Path:
    override = os.environ.get("LIVERKG_RUNS_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_state_dir(APP_NAME, appauthor=False)) / "runs"


def cache_dir() -> Path:
    override = os.environ.get("LIVERKG_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_cache_dir(APP_NAME, appauthor=False))
