"""Bounded, read-only GO lookups using EMBL-EBI OLS. See TRANSPORT.md."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

LOG = logging.getLogger(__name__)
REST_URL = "https://www.ebi.ac.uk/ols4/api"
MCP_URL = REST_URL + "/mcp"
PROTOCOL = "2025-03-26"
OBO = "http://purl.obolibrary.org/obo/"
IAO_DEFINITION = OBO + "IAO_0000115"
OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"
EXACT_SYNONYM = OBO_IN_OWL + "hasExactSynonym"
GO_NAMESPACE = OBO_IN_OWL + "hasOBONamespace"
GO_ASPECTS = {"biological_process", "molecular_function", "cellular_component"}
GO_ID = re.compile(r"GO:[0-9]{7}\Z")
REQUIRED_TOOLS = {
    "searchClasses": {"query", "ontologyId", "includeObsoleteEntities", "pageNum", "pageSize"},
    "getChildren": {"ontologyId", "classIri", "pageNum", "pageSize"},
    "getAncestors": {"ontologyId", "classIri", "pageNum", "pageSize"},
}
SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
PREDICATES = {
    SUBCLASS: "is_a",
    OBO + "BFO_0000050": "part_of",
    OBO + "BFO_0000051": "has_part",
    OBO + "RO_0002211": "regulates",
    OBO + "RO_0002212": "negatively_regulates",
    OBO + "RO_0002213": "positively_regulates",
}
MAX_BYTES = 4 * 1024 * 1024


class OLSError(RuntimeError):
    """A failed or unverifiable lookup. Callers should record the failed action."""


def _iri(go_id):
    return OBO + go_id.replace(":", "_")


def _id_from_iri(value):
    if isinstance(value, str) and value.startswith(OBO):
        go_id = value[len(OBO):].replace("_", ":")
        if GO_ID.fullmatch(go_id) and value == _iri(go_id):
            return go_id
    return None


def _values(raw):
    """Strings from a single named OWL annotation, including axiom annotations."""
    values = raw if isinstance(raw, list) else [raw]
    texts = []
    for item in values:
        if isinstance(item, dict):
            kinds = item.get("type")
            if isinstance(kinds, list) and "reification" in kinds:
                item = item.get("value")
        if isinstance(item, str) and item.strip() and item not in texts:
            texts.append(item)
    return texts


def _array(value, field):
    if not isinstance(value, list):
        raise OLSError(f"Missing or malformed OLS field {field}")
    return value


def _read(request, timeout):
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise OLSError("OLS response exceeded the byte limit")
            return response.headers, raw.decode("utf-8")
    except HTTPError as exc:
        raise OLSError(f"OLS HTTP {exc.code}") from exc
    except (URLError, OSError, UnicodeError) as exc:
        raise OLSError(f"OLS transport failed: {type(exc).__name__}") from exc


def _json(raw):
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        raise OLSError("Invalid OLS JSON") from exc
    if not isinstance(obj, dict):
        raise OLSError("Expected an OLS JSON object")
    return obj


class OLSClient:
    """A synchronous shareable client. Discovery is lazy and serialized."""

    def __init__(self, mode="auto", *, timeout=12.0, page_size=8, mcp_url=MCP_URL, rest_url=REST_URL):
        if mode not in {"auto", "rest", "mcp"}:
            raise ValueError("mode must be auto, mcp, or rest")
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("timeout must be in (0, 120] seconds")
        if type(page_size) is not int or not 1 <= page_size <= 30:
            raise ValueError("page_size must be in [1, 30]")
        for endpoint in (mcp_url, rest_url):
            parts = urlsplit(endpoint)
            if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
                raise ValueError("OLS endpoint URLs must be credential-free base URLs")
        self.mode, self.timeout, self.page_size = mode, timeout, page_size
        self.mcp_url, self.rest_url = mcp_url, rest_url.rstrip("/")
        self._lock = threading.Lock()
        self._discovery = None
        self._session = None

    def discover(self):
        """Return a snapshot of the selected backend and discovery outcome."""
        with self._lock:
            if self._discovery is None:
                if self.mode == "rest":
                    self._discovery = {"transport": "rest", "endpoint": self.rest_url}
                else:
                    try:
                        self._discovery = self._discover_mcp()
                    except OLSError as exc:
                        if self.mode == "mcp":
                            raise
                        LOG.warning("OLS MCP discovery failed; using REST: %s", exc)
                        self._discovery = {"transport": "rest", "endpoint": self.rest_url, "fallback_reason": str(exc)}
            return dict(self._discovery)

    def _rpc(self, method, params, *, notification=False):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            message["id"] = uuid4().hex
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if method != "initialize":
            headers["MCP-Protocol-Version"] = PROTOCOL
            if self._session:
                headers["Mcp-Session-Id"] = self._session
        request = Request(self.mcp_url, json.dumps(message).encode(), headers)
        response_headers, body = _read(request, self.timeout)
        if notification and not body.strip():
            return {}
        if response_headers.get("Content-Type", "").startswith("text/event-stream"):
            frames = body.replace("\r\n", "\n").split("\n\n")
            candidates = []
            for frame in frames:
                data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
                if data:
                    candidates.append(_json(data))
        else:
            candidates = [_json(body)]
        for rpc in candidates:
            if "error" in rpc:
                raise OLSError(f"MCP {method} reported a JSON-RPC error")
            if rpc.get("id") != message.get("id"):
                continue
            if rpc.get("jsonrpc") == "2.0" and isinstance(rpc.get("result"), dict):
                if method == "initialize":
                    self._session = response_headers.get("Mcp-Session-Id")
                return rpc["result"]
        raise OLSError(f"Invalid MCP reply to {method}")

    def _discover_mcp(self):
        reply = self._rpc("initialize", {
            "protocolVersion": PROTOCOL, "capabilities": {},
            "clientInfo": {"name": "liverkg-go-reference", "version": "1"},
        })
        if reply.get("protocolVersion") != PROTOCOL:
            raise OLSError("Unsupported MCP protocol version")
        self._rpc("notifications/initialized", {}, notification=True)
        catalog = self._rpc("tools/list", {})
        items = _array(catalog.get("tools"), "tools/list")
        tools = {item["name"]: item.get("inputSchema", {}) for item in items if isinstance(item, dict) and "name" in item}
        for name, params in REQUIRED_TOOLS.items():
            schema = tools.get(name, {})
            properties = schema.get("properties", {})
            if not params.issubset(properties) or set(schema.get("required", [])) - params:
                raise OLSError(f"Incompatible or missing OLS MCP tool {name}")
        return {"transport": "mcp", "endpoint": self.mcp_url, "protocol": PROTOCOL,
                "server_version": reply.get("serverInfo", {}).get("version"), "tools": sorted(tools)}

    def _term_url(self, go_id, *, graph=False):
        encoded = quote(quote(_iri(go_id), safe=""), safe="")
        if graph:
            return f"{self.rest_url}/ontologies/go/terms/{encoded}/graph"
        return f"{self.rest_url}/v2/ontologies/go/classes/{encoded}"

    def _get(self, url):
        _, body = _read(Request(url, headers={"Accept": "application/json"}), self.timeout)
        return _json(body)

    def _tool(self, name, arguments):
        response = self._rpc("tools/call", {"name": name, "arguments": arguments})
        content = _array(response.get("content"), "content")
        if response.get("isError"):
            raise OLSError(f"MCP {name} reported a tool error")
        if len(content) != 1 or content[0].get("type") != "text":
            raise OLSError(f"Unexpected MCP {name} content")
        return _json(content[0]["text"])

    def call(self, name, args):
        """Run one logical tool and return GO evidence plus its physical requests."""
        if name not in {"search", "detail", "ancestors", "children", "relations"}:
            raise ValueError("Unknown GO lookup tool")
        param = "query" if name == "search" else "go_id"
        if not isinstance(args, dict) or set(args) != {param} or not isinstance(args[param], str):
            raise ValueError(f"{name} requires exactly one string argument: {param}")
        value = args[param].strip()
        if name == "search":
            if not value or len(value) > 500:
                raise ValueError("query must have 1–500 characters")
        elif not GO_ID.fullmatch(value):
            raise ValueError("go_id must be GO: followed by seven digits")
        result = {"terms": [], "edges": [], "transport": "rest", "requests": []}
        if name in {"detail", "relations"}:
            # An explicit MCP mode requires a compatible server even for the two
            # records that only the REST API exposes in a verifiable form.
            if self.mode == "mcp":
                self.discover()
            if name == "detail":
                self._detail(value, result)
            else:
                self._relations(value, result)
        else:
            info = self.discover()
            result["transport"] = info["transport"]
            if "fallback_reason" in info:
                result["fallback_reason"] = info["fallback_reason"]
            if info["transport"] == "mcp":
                self._mcp_page(name, value, result)
            else:
                self._rest_page(name, value, result)
        return result

    def _detail(self, go_id, result):
        obj, url = self._active_class(go_id, result)
        labels = _values(obj.get("label"))
        definitions = _values(obj.get(IAO_DEFINITION))
        if not labels:
            raise OLSError("GO class has no label")
        namespaces = _values(obj.get(GO_NAMESPACE))
        namespace = namespaces[0] if len(namespaces) == 1 and namespaces[0] in GO_ASPECTS else None
        result["terms"] = [{"go_id": go_id, "name": labels[0], "definition": definitions[0] if definitions else None,
                            "namespace": namespace, "exact_synonyms": _values(obj.get(EXACT_SYNONYM)), "source": url}]

    def _active_class(self, go_id, result):
        url = self._term_url(go_id)
        obj = self._get(url)
        if (obj.get("curie") != go_id or obj.get("iri") != _iri(go_id) or obj.get("ontologyId") != "go"
                or obj.get("isObsolete") is not False or obj.get("isDefiningOntology") is not True
                or not isinstance(obj.get("type"), list) or "class" not in obj["type"]):
            raise OLSError("OLS did not confirm an active, GO-defined class with the requested ID")
        result["requests"].append({"transport": "rest", "url": url})
        return obj, url

    def _mcp_page(self, name, value, result):
        args = {"ontologyId": "go", "pageNum": 0, "pageSize": self.page_size}
        if name == "search":
            tool = "searchClasses"
            args.update(query=value, includeObsoleteEntities=False)
        else:
            tool = {"ancestors": "getAncestors", "children": "getChildren"}[name]
            args["classIri"] = _iri(value)
        page = self._tool(tool, args)
        rows = _array(page.get("items"), "items")
        result["requests"].append({"transport": "mcp", "endpoint": self.mcp_url, "tool": tool, "arguments": args})
        self._page_terms(rows, name, value, result)
        result["total_before_go_filter"] = page.get("totalElements")
        result["page_size"] = self.page_size

    def _rest_page(self, name, value, result):
        if name == "search":
            params = {"ontology": "go", "type": "class", "local": "true", "obsoletes": "false",
                      "rows": self.page_size, "q": value}
            url = f"{self.rest_url}/search?{urlencode(params)}"
            data = self._get(url).get("response", {})
            rows = _array(data.get("docs"), "response.docs")
            total = data.get("numFound")
        else:
            params = {"id": value, "page": 0, "size": self.page_size}
            url = f"{self.rest_url}/ontologies/go/{name}?{urlencode(params)}"
            data = self._get(url)
            total = data.get("page", {}).get("totalElements")
            rows = data.get("_embedded", {}).get("terms")
            if rows is None and total == 0:
                rows = []
            rows = _array(rows, "_embedded.terms")
        result["requests"].append({"transport": "rest", "url": url})
        self._page_terms(rows, name, value, result)
        result["total_before_go_filter"] = total
        result["page_size"] = self.page_size

    def _page_terms(self, rows, name, focus, result):
        mcp = result["transport"] == "mcp"
        if name != "search" and not rows:
            # OLS also returns an empty page for an unknown ID; distinguish a true leaf.
            self._active_class(focus, result)
        verified_children = set()
        if rows and name == "children" and not mcp:
            graph_rows, graph_url = self._graph(focus, result)
            verified_children = {_id_from_iri(edge.get("source")) for edge in graph_rows
                                 if edge.get("uri") == SUBCLASS and edge.get("target") == _iri(focus)}
        for obj in rows[:self.page_size]:
            go_id = _id_from_iri(obj.get("iri"))
            supplied_id = obj.get("curie") if mcp else obj.get("obo_id")
            ontology = obj.get("ontologyId") if mcp else obj.get("ontology_name")
            if not go_id or ontology != "go":
                continue  # GO imports BFO, RO and other non-GO classes.
            if supplied_id != go_id:
                raise OLSError("Inconsistent GO ID and IRI in OLS page")
            if mcp:
                if "class" not in obj.get("type", []) or obj.get("isObsolete") is not False:
                    continue
            elif obj.get("is_obsolete") or (name == "search" and obj.get("type") != "class"):
                continue
            labels = _values(obj.get("label"))
            if not labels:
                raise OLSError("Missing GO label in OLS page")
            source = self._term_url(go_id)
            result["terms"].append({"go_id": go_id, "name": labels[0], "definition": None, "source": source})
            if name != "children":
                continue
            if mcp:
                parents = _array(obj.get("directParent"), "directParent")
                if not any(isinstance(p, dict) and _id_from_iri(p.get("iri")) == focus for p in parents):
                    continue
            elif go_id not in verified_children:
                continue
            result["edges"].append({"subject": go_id, "predicate": "is_a", "object": focus, "predicate_iri": SUBCLASS,
                                    "source": source if mcp else graph_url})

    def _graph(self, go_id, result):
        url = self._term_url(go_id, graph=True)
        graph = self._get(url)
        rows = _array(graph.get("edges"), "edges")
        nodes = _array(graph.get("nodes"), "nodes")
        result["requests"].append({"transport": "rest", "url": url})
        if not any(node.get("iri") == _iri(go_id) for node in nodes):
            if nodes or rows:
                raise OLSError("Requested GO class missing from OLS graph")
            self._active_class(go_id, result)
        return rows, url

    def _relations(self, go_id, result):
        rows, url = self._graph(go_id, result)
        seen = set()
        for edge in rows:
            if edge.get("source") != _iri(go_id):
                continue
            target = _id_from_iri(edge.get("target"))
            predicate = PREDICATES.get(edge.get("uri"))
            if not target or not predicate or (predicate, target) in seen:
                continue
            seen.add((predicate, target))
            result["edges"].append({"subject": go_id, "predicate": predicate, "object": target,
                                    "predicate_iri": edge["uri"], "source": url})
        result["total_supported_outgoing"] = len(result["edges"])
        result["edges"] = result["edges"][:self.page_size]
