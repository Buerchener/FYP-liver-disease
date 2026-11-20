from cognitive_agent.evidence_pack import EvidencePackBuilder
from cognitive_agent.evidence_arguments import find_argument_competitor
from cognitive_agent.schema.schema_profile import PredicateRule, SchemaProfile


def _profile():
    return SchemaProfile(
        name="test", contract_version="1", semantic_target="RELATION_TRUTH",
        entity_types=frozenset({"Gene", "Disease"}),
        predicates={"ASSOCIATED_WITH": PredicateRule(
            name="ASSOCIATED_WITH", allowed_signatures=frozenset({("Gene", "Disease")}),
            explicit_patterns=(r"\bassociated with\b",),
        )},
    )


def _run(text, relation=None, aliases=None, quote=None):
    relation = relation or {
        "subject": "TP53", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
        "object": "HCC", "object_type": "Disease",
    }
    aliases = aliases or {
        ("TP53", "Gene"): [], ("KRAS", "Gene"): [], ("HCC", "Disease"): [],
    }
    return find_argument_competitor(
        relation, source_index=EvidencePackBuilder().index(text), profile=_profile(),
        entity_aliases=aliases, quote=quote or text,
    )


def test_finds_source_aligned_subject_competitor_in_independent_clause():
    text = "RESULTS: TP53 was measured; KRAS was associated with HCC."
    result = _run(text)
    assert result["status"] == "COMPETING_ARGUMENT"
    assert result["role"] == "subject"
    assert result["replacement"] == {"mention": "KRAS", "type": "Gene"}
    evidence = result["evidence"]
    assert text[evidence["char_start"]:evidence["char_end"]] == evidence["text"]
    trigger = result["trigger"]
    assert text[trigger["char_start"]:trigger["char_end"]] == "associated with"
    assert result["arguments"]["subject"]["text"] == "KRAS"
    assert result["arguments"]["object"]["text"] == "HCC"
    for endpoint in result["arguments"].values():
        assert text[endpoint["char_start"]:endpoint["char_end"]] == endpoint["text"]


def test_does_not_flag_coordinated_subjects():
    assert _run("RESULTS: TP53 and KRAS were associated with HCC.") == {}


def test_direct_candidate_support_vetoes_other_clause_competitor():
    assert _run("RESULTS: TP53 was associated with HCC; KRAS too.") == {}


def test_requires_verbatim_quote_and_single_parent_sentence():
    text = "RESULTS: TP53 was measured. KRAS was associated with HCC."
    result = find_argument_competitor(
        {"subject": "TP53", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
         "object": "HCC", "object_type": "Disease"},
        source_index=EvidencePackBuilder().index(text), profile=_profile(),
        entity_aliases={("TP53", "Gene"): [], ("KRAS", "Gene"): [], ("HCC", "Disease"): []},
        quote="TP53 was measured. KRAS was associated with HCC.",
    )
    assert result == {}
    assert _run("RESULTS: TP53 was measured; KRAS was associated with HCC.", quote="TP53 measured; KRAS associated with HCC") == {}


def test_ignores_same_canonical_alias():
    aliases = {("TP53", "Gene"): ["P53"], ("P53", "Gene"): ["TP53"],
               ("KRAS", "Gene"): [], ("HCC", "Disease"): []}
    relation = {"subject": "TP53", "subject_type": "Gene", "predicate": "ASSOCIATED_WITH",
                "object": "HCC", "object_type": "Disease"}
    assert _run("RESULTS: TP53 was measured; P53 was associated with HCC.", relation, aliases) == {}


def test_finds_competing_object_and_requires_affirmative_support():
    text = "RESULTS: TP53 was associated with fibrosis; HCC was measured."
    aliases = {("TP53", "Gene"): [], ("fibrosis", "Disease"): [], ("HCC", "Disease"): []}
    result = _run(text, aliases=aliases)
    assert result["role"] == "object"
    assert result["replacement"] == {"mention": "fibrosis", "type": "Disease"}
    negated = "RESULTS: TP53 was measured; KRAS was not associated with HCC."
    assert _run(negated) == {}
