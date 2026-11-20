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


def test_parallel_parenthetical_groups_keep_objects_with_their_subjects():
    text = ("RESULTS: scRNA-seq validated cell-type-specific expression patterns of CCND1 "
            "(epithelial cells, endothelial cells, hepatocytes, macrophages) and "
            "IL7R (T/NK cells).")
    profile = SchemaProfile(
        name="expression", contract_version="1", semantic_target="RELATION_TRUTH",
        entity_types=frozenset({"Gene", "CellType"}),
        predicates={"EXPRESSED_IN": PredicateRule(
            name="EXPRESSED_IN", allowed_signatures=frozenset({("Gene", "CellType")}),
            explicit_patterns=(r"expression patterns of",),
            parenthetical_group_binding=True,
        )},
    )
    aliases = {(name, kind): [] for name, kind in (
        ("CCND1", "Gene"), ("IL7R", "Gene"), ("epithelial cells", "CellType"),
        ("endothelial cells", "CellType"), ("hepatocytes", "CellType"),
        ("macrophages", "CellType"), ("T/NK cells", "CellType"),
    )}
    index = EvidencePackBuilder().index(text)
    def run(obj):
        return find_argument_competitor(
            {"subject": "CCND1", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
             "object": obj, "object_type": "CellType"},
            source_index=index, profile=profile, entity_aliases=aliases, quote=text,
        )
    wrong = run("T/NK cells")
    assert wrong["status"] == "COMPETING_ARGUMENT"
    assert wrong["binding_mode"] == "PARENTHETICAL_GROUP"
    assert wrong["replacement"] == {"mention": "IL7R", "type": "Gene"}
    assert text[wrong["arguments"]["subject"]["char_start"]:wrong["arguments"]["subject"]["char_end"]] == "IL7R"
    assert text[wrong["arguments"]["object"]["char_start"]:wrong["arguments"]["object"]["char_end"]] == "T/NK cells"
    assert run("hepatocytes") == {}
    reverse = find_argument_competitor(
        {"subject": "IL7R", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
         "object": "hepatocytes", "object_type": "CellType"},
        source_index=index, profile=profile, entity_aliases=aliases, quote=text,
    )
    assert reverse["replacement"] == {"mention": "CCND1", "type": "Gene"}
    owner_span = reverse["arguments"]["subject"]
    assert text[owner_span["char_start"]:owner_span["char_end"]] == "CCND1"
    assert owner_span["char_start"] == text.index("CCND1")
    ablated = find_argument_competitor(
        {"subject": "CCND1", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
         "object": "T/NK cells", "object_type": "CellType"},
        source_index=index, profile=profile, entity_aliases=aliases, quote=text,
        compare_groups=False,
    )
    assert ablated == {}


def test_parenthetical_groups_require_separate_grounded_lists():
    text = "RESULTS: expression patterns of CCND1 and IL7R were validated in hepatocytes."
    profile = SchemaProfile(
        name="expression", contract_version="1", semantic_target="RELATION_TRUTH",
        entity_types=frozenset({"Gene", "CellType"}),
        predicates={"EXPRESSED_IN": PredicateRule(
            name="EXPRESSED_IN", allowed_signatures=frozenset({("Gene", "CellType")}),
            explicit_patterns=(r"expression patterns of",),
            parenthetical_group_binding=True,
        )},
    )
    aliases = {(n, t): [] for n, t in (("CCND1", "Gene"), ("IL7R", "Gene"), ("hepatocytes", "CellType"), ("T/NK cells", "CellType"))}
    result = find_argument_competitor(
        {"subject": "CCND1", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
         "object": "T/NK cells", "object_type": "CellType"},
        source_index=EvidencePackBuilder().index(text), profile=profile,
        entity_aliases=aliases, quote=text,
    )
    assert result == {}


def test_parenthetical_binding_fails_closed_on_interposed_clause_and_overlapping_rivals():
    profile = SchemaProfile(
        name="expression", contract_version="1", semantic_target="RELATION_TRUTH",
        entity_types=frozenset({"Gene", "CellType"}),
        predicates={"EXPRESSED_IN": PredicateRule(
            name="EXPRESSED_IN", allowed_signatures=frozenset({("Gene", "CellType")}),
            explicit_patterns=(r"expression patterns of",), parenthetical_group_binding=True,
        )},
    )
    aliases = {(n, t): [] for n, t in (
        ("CCND1", "Gene"), ("IL7R", "Gene"), ("TP53", "Gene"),
        ("hepatocytes", "CellType"), ("T/NK cells", "CellType"),
    )}
    def run(text):
        return find_argument_competitor(
            {"subject": "CCND1", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
             "object": "T/NK cells", "object_type": "CellType"},
            source_index=EvidencePackBuilder().index(text), profile=profile,
            entity_aliases=aliases, quote=text,
        )
    assert run("RESULTS: expression patterns of CCND1 (hepatocytes), after adjustment, and IL7R (T/NK cells).") == {}
    assert run("RESULTS: expression patterns of CCND1 (hepatocytes), IL7R (T/NK cells), and TP53 (T/NK cells).") == {}
    assert run("RESULTS: expression patterns of CCND1 (hepatocytes), CCND1 (T/NK cells).") == {}
    assert run(
        "RESULTS: expression patterns of CCND1 (hepatocytes and T/NK cells) and IL7R (T/NK cells)."
    ) == {}
    assert run(
        "RESULTS: expression patterns of CCND1 (hepatocytes), IL7R (unlabelled cells), and TP53 (T/NK cells)."
    )["replacement"]["mention"] == "TP53"


def test_parenthetical_binding_is_opt_in_per_predicate_rule():
    text = "RESULTS: expression patterns of CCND1 (hepatocytes) and IL7R (T/NK cells)."
    profile = SchemaProfile(
        name="custom", contract_version="1", semantic_target="RELATION_TRUTH",
        entity_types=frozenset({"Gene", "CellType"}),
        predicates={"EXPRESSED_IN": PredicateRule(
            name="EXPRESSED_IN", allowed_signatures=frozenset({("Gene", "CellType")}),
            explicit_patterns=(r"expression patterns of",),
        )},
    )
    aliases = {(n, t): [] for n, t in (("CCND1", "Gene"), ("IL7R", "Gene"),
                                       ("hepatocytes", "CellType"), ("T/NK cells", "CellType"))}
    result = find_argument_competitor(
        {"subject": "CCND1", "subject_type": "Gene", "predicate": "EXPRESSED_IN",
         "object": "T/NK cells", "object_type": "CellType"},
        source_index=EvidencePackBuilder().index(text), profile=profile,
        entity_aliases=aliases, quote=text,
    )
    assert result == {}
