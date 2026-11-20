"""Atomic writes and a process lock; a checkpoint commits a complete transition."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def atomic_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def run_lock(directory):
    with (Path(directory) / '.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('This run is already being processed') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)

def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
