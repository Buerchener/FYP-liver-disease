"""Atomic writes and a process lock; a checkpoint commits a complete transition."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
from ..models import canonical_json
from .graph import digest


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def save_checkpoint(path, state):
    atomic_json(path, {'payload_sha256': digest(state), 'state': state})


def read_checkpoint(path):
    value = read_json(path)
    if digest(value['state']) != value['payload_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    return value['state']


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
