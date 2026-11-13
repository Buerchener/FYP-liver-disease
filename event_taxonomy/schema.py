"""JSON Schema for complete, serialized records; cross-record checks are separate."""
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints
from . import models

RECORDS = (models.SourceRef, models.Document, models.Evidence, models.EventInstance,
           models.ConceptCandidate, models.UniqueConcept, models.ParentChildEdge, models.Taxonomy)


def _type_schema(annotation):
    if annotation is Any:
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if is_dataclass(annotation):
        return {"$ref": "#/$defs/" + annotation.__name__}
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Union:
        return {"anyOf": [_type_schema(arg) for arg in args]}
    if origin is list:
        return {"type": "array", "items": _type_schema(args[0])}
    if origin is dict:
        return {"type": "object", "additionalProperties": _type_schema(args[1])}
    return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[annotation]}


def record_schema():
    definitions = {}
    for model in RECORDS:
        hints = get_type_hints(model)
        props = {f.name: _type_schema(hints[f.name]) for f in fields(model)}
        props["schema_version"] = {"const": models.SCHEMA_VERSION}
        definitions[model.__name__] = {"type": "object", "properties": props,
                                      "required": list(props), "additionalProperties": False}
    definitions["ConceptCandidate"]["properties"]["candidate_role"] = {"enum": ["event_type", "topic", "unknown"]}
    definitions["UniqueConcept"]["properties"]["concept_role"] = {"enum": ["event_type", "topic", "unknown"]}
    definitions["Evidence"]["properties"]["alignment_status"] = {"enum": [
        "verified_exact", "verified_whitespace", "ambiguous", "unverified", "passage_only", "missing"]}
    definitions["ParentChildEdge"]["properties"]["relation"] = {"const": "SUBSUMES"}
    definitions["ParentChildEdge"]["properties"]["status"] = {"enum": ["provisional", "accepted"]}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Event Taxonomy Records",
            "$defs": definitions, "oneOf": [{"$ref": "#/$defs/" + model.__name__} for model in RECORDS]}
