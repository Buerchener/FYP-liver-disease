"""Read-only GO lookup for open placement questions."""

import json
import subprocess
import sys

from ..storage import atomic_json, digest, read_json


GO_TYPES = {"Pathway", "BiologicalProcess", "MolecularFunction", "CellularComponent"}
STATUSES = {"MATCHED", "REFERENCE_ONLY", "NO_MATCH", "UNCERTAIN", "LOOKUP_FAILED"}
MAX_TOOL_CALLS = 5


def reference_evidence(result):
    if result["status"] not in {"MATCHED", "REFERENCE_ONLY"}:
        return None
    if not (result.get("go_definition") or result.get("ontology_relations")):
        return None
    return {key: result[key] for key in (
        "concept_id", "status", "go_id", "go_name", "go_definition",
        "relation_type", "source", "ontology_relations", "term_query_id",
    )}


class GOReferenceLookup:
    def __init__(self, config, directory):
        self.config = config.resolve()
        self.directory = directory.resolve()

    def __call__(self, concept, questions):
        concept_id = concept["id"]
        parents = {
            item["parent"]["id"]: item["parent"]
            for item in questions
            if item["child"]["id"] == concept_id
        }
        request = {
            "concept_id": concept_id,
            "name": concept["name"],
            "definition": concept["definition"],
            "candidate_parents": [
                {"concept_id": ident, "name": row["name"], "definition": row["definition"],
                 **({"go_id": row["go_id"]} if row.get("go_id") else {})}
                for ident, row in list(parents.items())[:50]
            ],
            "taxonomy_context": {
                "entity_type": concept["kind"],
                "evidence_request": (
                    "What direct GO relation, if any, connects the corresponding terms? "
                    "Preserve the observed predicate and direction. Cite the detail query "
                    "for the GO term selected as the reference to this concept."
                ),
                "unresolved_parent_child_questions": [
                    {"child": item["child"]["name"], "parent": item["parent"]["name"],
                     "child_id": item["child"]["id"], "parent_id": item["parent"]["id"]}
                    for item in questions
                ],
            },
        }
        path = self.directory / f"{digest(concept_id)}.json"
        request_id = digest(request)
        if path.exists():
            entry = read_json(path)
            if entry["request_sha256"] != request_id:
                return self.failed(concept_id, "lookup_already_used_for_another_context")
            # The checkpoint records a started call even if the process stopped mid-run.
            return entry.get("result") or self.failed(concept_id, "interrupted_lookup")

        entry = {"request_sha256": request_id, "request": request}
        atomic_json(path, entry)
        result = self._lookup(request)
        atomic_json(path, {**entry, "result": result})
        return result

    @staticmethod
    def failed(concept_id, reason):
        return {"concept_id": concept_id, "status": "LOOKUP_FAILED", "exit_reason": reason}

    def _lookup(self, request):
        concept_id = request["concept_id"]
        command = [
            sys.executable, "-m", "entity_ontology.hierarchy.go_reference",
            "--model-config", str(self.config), "--mode", "auto",
            "--max-tool-calls", str(MAX_TOOL_CALLS), "--model-timeout", "20",
            "--state-dir", str(self.directory / "agent-runs"),
        ]
        try:
            reply = subprocess.run(
                command, input=json.dumps(request, ensure_ascii=False),
                capture_output=True, text=True, timeout=240,
            )
        except subprocess.TimeoutExpired:
            return self.failed(concept_id, "reference_timeout")
        except (OSError, UnicodeError):
            return self.failed(concept_id, "reference_runtime_unavailable")
        if reply.returncode:
            return self.failed(concept_id, f"reference_exit_{reply.returncode}")
        try:
            result = json.loads(reply.stdout)
            if (result["concept_id"] != concept_id or result["status"] not in STATUSES
                    or type(result["tool_calls"]) is not int
                    or not 0 <= result["tool_calls"] <= MAX_TOOL_CALLS):
                raise ValueError("Inconsistent GO reference result")
            # The standalone Agent has already validated the citation against OLS evidence.
            reference_evidence(result)
            return result
        except (KeyError, TypeError, ValueError):
            return self.failed(concept_id, "invalid_reference_output")
