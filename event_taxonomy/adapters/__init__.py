from ..models import ImportBundle
from .official import import_official_jsonl, import_stage2_document
from .workbench import import_graph, import_workspace

PROFILES = ("workbench_graph", "workbench_workspace", "workbench_export",
            "official_jsonl", "original_stage2", "role_nested")


def load_sources(sources, profile, namespace, origin="real_literature", source_run_id=None):
    if profile not in PROFILES:
        raise ValueError("Unsupported source profile: " + profile)
    if not namespace or not isinstance(namespace, str):
        raise ValueError("An explicit dataset namespace is required")
    bundle = ImportBundle()
    for source in sources:
        if profile != "official_jsonl":
            payload = source.json()
            if not isinstance(payload, dict):
                raise ValueError("Source profile requires a JSON object")
            declared = payload.get("data_origin")
            if declared is not None and declared != origin:
                raise ValueError("Source data_origin conflicts with configured origin")
        # Legacy files have no global run IDs: pin a run ID in configuration.
        if profile in {"official_jsonl", "original_stage2", "role_nested"} and not source_run_id:
            raise ValueError("Legacy import requires source_run_id")
        if profile == "workbench_graph":
            import_graph(source, bundle, namespace, origin)
        elif profile in {"workbench_workspace", "workbench_export"}:
            import_workspace(source, bundle, namespace, origin, profile == "workbench_export")
        elif profile == "official_jsonl":
            import_official_jsonl(source, bundle, namespace, origin, source_run_id)
        else:
            import_stage2_document(source, bundle, namespace, origin, source_run_id, profile)
    return bundle
