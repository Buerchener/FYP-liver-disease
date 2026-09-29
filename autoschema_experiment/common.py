import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def uid(kind, *values):
    return kind + "_" + digest(values)[:20]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: nonempty string required")
    return value.strip()


def load_articles(path):
    articles = read_jsonl(path)
    seen = set()
    if not articles:
        raise ValueError("no articles")
    for a in articles:
        pmid = require_text(a.get("pmid"), "pmid")
        if not pmid.isdigit() or pmid in seen:
            raise ValueError("PMID must be numeric and unique")
        seen.add(pmid)
        a["source_text"] = require_text(a.get("abstract") or a.get("text"), "abstract/text")
        a["source_hash"] = digest(a["source_text"])
    return articles
