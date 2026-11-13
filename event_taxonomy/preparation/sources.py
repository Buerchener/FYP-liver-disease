from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


@dataclass
class SourceFile:
    path: Path
    content: bytes
    sha256: str

    @classmethod
    def read(cls, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        return cls(path, content, hashlib.sha256(content).hexdigest())

    def json(self):
        return json.loads(self.content.decode("utf-8-sig"))

    def jsonl(self):
        for index, line in enumerate(self.content.decode("utf-8-sig").splitlines()):
            if line.strip():
                yield index + 1, json.loads(line)

    def assert_unchanged(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("Source changed during preparation: " + str(self.path))
