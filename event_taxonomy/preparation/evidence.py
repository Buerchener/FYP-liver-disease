import re
from typing import Optional
from ..models import Document, Evidence, SourceRef, stable_id


def locate_evidence(document: Document, quote: Optional[str], source: SourceRef,
                    upstream_status: Optional[str] = None) -> Evidence:
    """Keep original quotes; enumerate repeated matches without choosing one."""
    if quote is not None and not isinstance(quote, str):
        raise ValueError("Evidence quote must be a string or null")
    spans = []
    if not quote or not quote.strip():
        status, granularity, quote = "passage_only", "passage", None
    else:
        position = document.text.find(quote)
        while position >= 0:
            spans.append({"start": position, "end": position + len(quote)})
            position = document.text.find(quote, position + 1)
        status = "verified_exact" if spans else "unverified"
        if not spans:
            pattern = r"\s+".join(re.escape(part) for part in quote.split())
            spans = [{"start": m.start(), "end": m.end()}
                     for m in re.finditer(pattern, document.text)]
            if spans:
                status = "verified_whitespace"
        if len(spans) > 1:
            status = "ambiguous"
        # Locating text alone does not establish that it is a complete sentence.
        granularity = "quote"
    return Evidence(stable_id("evidence", document.document_id, quote, source.source_run_id,
                              source.record_locator), document.document_id, quote, spans,
                    granularity, status, source, upstream_status)


def primary_sentence(evidence):
    """Only expose source text that was located unambiguously, never a paraphrase."""
    for item, document in evidence:
        if item.alignment_status in {"verified_exact", "verified_whitespace"}:
            span = item.spans[0]
            return document.text[span["start"]:span["end"]]
    return None
