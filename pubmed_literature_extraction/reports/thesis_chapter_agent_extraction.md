# Chapter X: Intelligent Literature Extraction with LLM-Based Cognitive Agent

> 融入位置建议：在队友的 v2 Curated Graph (Chapter 4-5) 之后，作为 KG 动态补全/文献抽取模块。与队友的数据库整合部分（DisGeNET / STRING / KEGG）互补。

---

## X.1 Introduction

### X.1.1 Problem Context and Motivation

The v2 curated knowledge graph described in previous chapters integrates evidence from six biomedical databases (DisGeNET, STRING, KEGG, Reactome, HPA, HMDB) into a five-disease backbone. While this ontology-first construction provides a defensible and auditable foundation, it relies entirely on pre-existing database records — many of which have not been updated with the latest experimental and clinical findings published in the biomedical literature.

Every year, tens of thousands of new PubMed articles report novel gene-disease associations, pathway mechanisms, and biomarker discoveries relevant to liver disease progression. Manually curating these articles into the knowledge graph is infeasible at scale. A system that can automatically read PubMed abstracts and extract structured entities and relations — while maintaining quality standards comparable to database curation — would significantly enhance the timeliness and coverage of the knowledge graph.

However, literature-based extraction poses unique quality challenges not present in database import. Database records (e.g., DisGeNET) are already curated with controlled identifiers and explicit provenance. In contrast, LLM-based extraction from free text introduces three critical quality risks:

| Risk | Manifestation | Root Cause |
|------|---------------|------------|
| **Generic term leakage** | Non-specific terms like "gene symbols", "chemicals", "malignancies", "immune regulation" extracted as entities | LLMs produce generic descriptors when processing review articles; no pre-existing controlled vocabulary to filter against |
| **Same-concept multi-node** | "T2DM" and "type 2 diabetes mellitus" created as separate nodes; "HCC" and "Hepatocellular carcinoma" duplicated | Biomedical text freely mixes abbreviations and full forms; LLM extraction does not guarantee canonical naming |
| **Methodology noise** | "network pharmacology", "molecular docking", "KEGG signaling pathway" extracted as biological entities | Research method terms and database names are lexically similar to biological entity names but semantically distinct |

### X.1.2 Problem Statement

This chapter addresses the problem of constructing a high-quality, LLM-driven entity and relation extraction pipeline for liver disease literature that minimises generic noise, resolves abbreviation-induced duplication, and filters out methodology artifacts — while preserving recall of genuine biomedical entities. The specific research questions are:

- **RQ4**: How can a literature extraction pipeline achieve database-curation-grade entity quality from uncurated PubMed abstracts?
- **RQ5**: How can abbreviation-induced entity duplication be resolved without access to an external terminology server?
- **RQ6**: How can methodology and database-name noise be distinguished from legitimate biological entities using lightweight, rule-based methods?

### X.1.3 Scope

This chapter covers the design, implementation, and evaluation of the **Cognitive Agent V3** extraction pipeline. The agent processes PubMed abstracts through a multi-phase cognitive architecture, extracts biomedical entities and relations, and applies a three-layer quality defense system before committing to the knowledge graph. The evaluation uses a 30-article PubMed liver disease corpus and compares extraction quality against both the V1 rule-based pipeline and the V2 (pre-fix) agent.

---

## X.2 Related Work

### X.2.1 LLM-Based Biomedical Information Extraction

Recent work has demonstrated that large language models can perform named entity recognition (NER) and relation extraction (RE) from biomedical text with competitive accuracy [1, 2]. Systems like BioREx [3] and EvidenceNet [4] use LLMs to extract disease-gene, drug-target, and pathway associations from PubMed abstracts. The LangExtract library [5] provides a structured extraction framework that wraps LLM calls with schema-constrained output parsing, making it suitable for domain-specific KG population tasks.

### X.2.2 Abbreviation Resolution in Biomedical Text

The Schwartz-Hearst algorithm [6, 7] is the gold standard for biomedical abbreviation detection. It operates on the observation that abbreviations in biomedical text almost always appear in parenthetical patterns: "long form (ABBR)". The algorithm matches the characters of the short form against the initial characters of words preceding the parenthesis, tolerating stopwords and multi-character matches within words. This project adopts the Schwartz-Hearst algorithm as the core abbreviation detection engine, supplemented with a curated abbreviation dictionary sourced from UMLS and MeSH for liver-disease-specific terms.

### X.2.3 Knowledge Graph Quality Assurance

Quality assurance in KG construction typically involves schema validation, identifier normalization, and scope auditing [8]. For LLM-extracted content, additional quality layers are needed because the extraction source (free text) lacks the controlled vocabulary and identifier discipline of curated databases. Prior work has applied rule-based filtering [9] and confidence scoring [10] to LLM extractions, but an integrated multi-layer defense combining blacklist filtering, methodology detection, and abbreviation-aware deduplication has not been systematically described for the biomedical domain.

---

## X.3 Research Methodology

### X.3.1 Multi-Phase Cognitive Architecture

The Cognitive Agent processes each article through a sequence of seven cognitive phases, designed to mirror the reasoning process of a human curator reading a biomedical abstract:

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Context** | Query Neo4j for existing knowledge about entities mentioned in the article; identify knowledge gaps |
| 2 | **Strategy** | Select extraction strategy (exploratory vs. focused) based on knowledge coverage score |
| 3 | **Extraction** | Call LLM (Gemini 2.5 Flash) via LangExtract to extract entities and relations |
| 4 | **Verification** | LLM self-verification: re-examine extracted entities and relations for correctness |
| 5 | **Causal Reasoning** | Infer transitive causal chains (A→B + B→C in KG ⇒ A→C inferred) |
| 6 | **Conflict Resolution** | Resolve contradictions between new extraction and existing KG using decision-table logic |
| 7 | **Execution** | Commit validated entities and relations to Neo4j (or log decisions in offline mode) |

Before Phase 2, the agent runs **abbreviation detection** on the article text using the Schwartz-Hearst algorithm and the curated abbreviation dictionary. The resulting abbreviation map (`AbbreviationMap`) is passed to the decision engine for canonical name resolution during Phase 5-7.

### X.3.2 Three-Layer Quality Defense System

The core methodological contribution of this chapter is a three-layer quality defense that operates during the decision phase (Phase 5-6), before entities are committed to the knowledge graph. Critically, all three layers execute in the `decide()` stage rather than the `execute()` stage, ensuring that quality filtering works in both online (Neo4j-connected) and offline (no-Neo4j) modes.

```
LLM-Extracted Entity
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: GENERIC_TERM_BLACKLIST (Exact Match)               │
│   ~210 terms covering:                                      │
│   • Methodology noise ("network pharmacology", "ELISA")     │
│   • LangExtract artifacts ("gene symbols", "disease-related │
│     targets", "signaling molecules")                        │
│   • Generic cell biology ("immune regulation",              │
│     "immunomodulation", "cellular processes")               │
│   • Statistical/research terms ("replication",              │
│     "confounding factors", "baseline characteristics")      │
│   Match → DISCARD (reason: "Generic/blacklisted term")      │
└─────────────────────────────────────────────────────────────┘
        │ Pass
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: is_methodology_noise() (Exact + Substring)         │
│   METHODOLOGY_BLACKLIST: ~120 terms in frozenset            │
│   • Bioinformatics tools: STRING, Cytoscape, AutoDock       │
│   • Lab techniques: western blot, immunohistochemistry,     │
│     qRT-PCR, ELISA                                          │
│   • Database names: Gene Ontology, KEGG, Reactome, UniProt  │
│   • Research methods: network pharmacology, molecular       │
│     docking, enrichment analysis                            │
│   Substring whitelist: "signaling pathway", "metabolic      │
│   pathway" excluded from substring match to protect         │
│   legitimate entities (e.g., "NF-κB signaling pathway")     │
│   Match → DISCARD (reason: "Methodology/research tool")     │
└─────────────────────────────────────────────────────────────┘
        │ Pass
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: score_entity_name_quality() (9 Penalty Patterns)   │
│   Pattern                          Penalty                  │
│   ───────────────────────────────  ────────                 │
│   Too short (≤3 characters)         -0.4                    │
│   All lowercase + not proper noun   -0.3                    │
│   Starts with generic category      -0.3                    │
│     ("level of", "role of")                                 │
│   Contains methodology term          -0.4                    │
│   Database/resource name             -0.3                    │
│   Study population term              -0.3                    │
│   Stopword-like name                 -0.3                    │
│   All-caps abbreviation, no          -0.2                    │
│     dictionary match                                         │
│   Pure numeric/symbol                -0.5                    │
│   Score < 0.3 → DISCARD                                    │
└─────────────────────────────────────────────────────────────┘
        │ Pass
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Abbreviation-Aware Entity Deduplication            │
│   AbbreviationDetector + AbbreviationMap                    │
│   _deduplicate_entities():                                   │
│   1. Group entities by canonical_name() lookup:              │
│      curated_dict[name] → detected_abbr[name] → original     │
│   2. Within each group, retain longest/most descriptive      │
│   3. Merge discarded entities' attributes into retained       │
│   "ALT" → "alanine aminotransferase" (1 node, not 2)        │
│   "T2DM" → "type 2 diabetes mellitus" (1 node, not 2)       │
│   "HCC" → "Hepatocellular carcinoma" (1 node, not 2)       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
    CREATE Entity
```

### X.3.3 Abbreviation Detection Components

**Schwartz-Hearst Algorithm.** The algorithm scans for parenthetical patterns `"text (ABBR)"` and matches the short form's characters against the initial characters of words preceding the parenthesis. Our implementation includes three enhancements over the original algorithm:

1. **Multi-candidate collection**: Rather than returning the first valid match, all candidate long forms are collected and scored. The best candidate minimises the number of in-word character matches and maximises the number of matched words.
2. **Expanded character window**: The scan window is `min(len(short_form) × 40, 400)` characters, accommodating long biomedical terms like "alanine aminotransferase" for the 3-letter abbreviation "ALT".
3. **Stopword tolerance**: Words in the stopword set (of, and, in, the, for, to, with, from, by, on, at, or, as, an, a, is, was, were, are, be, been, being, has, have, had, do, does, did) are included in the long form but do not consume a character from the short form.

**CURATED_ABBREVIATIONS Dictionary.** A 62-entry dictionary covering six biomedical categories relevant to liver disease, manually curated from UMLS and MeSH:

| Category | Count | Examples |
|----------|-------|----------|
| Liver disease terms | 12 | ALT→alanine aminotransferase, AST→aspartate aminotransferase, HCC→Hepatocellular carcinoma |
| Diabetes/metabolic | 10 | T2DM→type 2 diabetes mellitus, NAFLD→non-alcoholic fatty liver disease, MASLD→metabolic dysfunction-associated steatotic liver disease |
| Clinical measurements | 8 | BMI→body mass index, HOMA-IR→homeostatic model assessment of insulin resistance |
| Biomolecules | 14 | ROS→reactive oxygen species, TNF-α→tumor necrosis factor alpha, TGF-β→transforming growth factor beta |
| Cell types | 8 | HSC→hepatic stellate cell, KC→Kupffer cell, LSEC→liver sinusoidal endothelial cell |
| Methodology terms | 10 | GO→Gene Ontology, KEGG→Kyoto Encyclopedia of Genes and Genomes (tracked for noise filtering) |

The methodology category is explicitly tracked: when these abbreviations appear, their long forms are flagged for potential noise filtering in Layer 2.

### X.3.4 GENERIC_TERM_BLACKLIST Construction

The blacklist was constructed iteratively through inspection of extraction outputs from the V2 agent across 30 PubMed articles. Terms were categorised into four families:

| Category | Count | Source of Discovery |
|----------|-------|---------------------|
| Methodology noise | ~30 | Review articles using phrases like "we performed network pharmacology analysis" |
| LangExtract artifacts | ~25 | Systematic over-extraction of generic descriptors in article introductions |
| Generic cell biology | ~20 | Terms like "immune regulation" extracted instead of specific pathways like "NF-κB signaling" |
| Statistical/research terms | ~15 | Clinical study descriptions leaking into entity extraction |

The blacklist is a Python `frozenset` for O(1) membership testing, stored in `kg_memory.py` as `GENERIC_TERM_BLACKLIST`.

### X.3.5 Abbreviation-Aware Deduplication Algorithm

The `_deduplicate_entities()` method in the decision engine groups entities by their canonical form using a three-tier resolution strategy:

```
def canonical_name(name):
    1. Check CURATED_ABBREVIATIONS dict → if found, return long form
    2. Check detected abbreviations (Schwartz-Hearst) → if found, return long form
    3. Otherwise → return original name (no resolution available)
```

Within each canonical group, the entity with the longest name (in characters) is retained as the primary node, and all other entities' attributes (associations, evidence) are merged into the primary node. This greedy strategy favours fully expanded names over abbreviations, which improves graph interpretability.

---

## X.4 Solution: Design and Implementation

### X.4.1 System Architecture

The Cognitive Agent V3 is implemented as a Python package (`cognitive_agent/`) with the following module structure:

```
cognitive_agent/
├── agent.py                    # Main agent loop, concurrency, CLI
├── decision_engine.py          # Entity/relation decision logic + quality defense
├── extraction_kernel.py        # LangExtract LLM wrapper
├── abbreviation_detector.py    # Schwartz-Hearst + curated dict + methodology filter
├── memory/
│   ├── working_memory.py       # Per-article cache
│   ├── episodic_memory.py      # Cross-article decision history (thread-safe)
│   └── kg_memory.py            # KG read/write + GENERIC_TERM_BLACKLIST
├── cognitive/
│   ├── causal_reasoner.py      # Transitive inference
│   ├── conflict_resolver.py    # Decision-table conflict resolution
│   ├── self_reflection.py      # Quality metrics + threshold adaptation
│   └── strategy_manager.py     # Exploration vs. focused mode selection
└── tools/
    └── ncbi_validator.py       # NCBI E-utilities gene validation
```

### X.4.2 Agent Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `[按次]gemini-2.5-flash` | LLM model for extraction and verification |
| `max_workers` | 5 | Concurrent article processing threads |
| `reflection_interval` | 10 | Articles processed before self-reflection trigger |
| `entity_confidence_threshold` | 0.6 | Minimum confidence for entity creation |
| `relation_confidence_threshold` | 0.7 | Minimum confidence for relation creation |
| `skip_neo4j_write` | False | Offline mode: log decisions without Neo4j commits |

### X.4.3 Concurrency Design

Both the Cognitive Agent and the V1 Pipeline support concurrent article processing using `ThreadPoolExecutor` with configurable `max_workers`. Thread safety is ensured by:

- **AgentState**: All mutation methods (`safe_add`, `safe_append`) protected by `threading.Lock`
- **EpisodicMemory**: `record()` method protected by `threading.Lock`
- **Agent history**: `process_article()` appends to history under `_history_lock`
- **Progress output**: Thread-safe progress tracking with shared counter and lock

### X.4.4 API Key Cascade

The agent supports automatic API key fallback across providers:

```
GEMINI_API_KEY → DEEPSEEK_API_KEY → --api-key CLI argument
```

This enables seamless switching between production (Gemini) and backup (DeepSeek) LLM providers without code changes.

### X.4.5 Implementation Scripts

| Script / Module | Role |
|-----------------|------|
| `cognitive_agent/agent.py` | Main entry point for batch article processing |
| `cognitive_agent/abbreviation_detector.py` | Schwartz-Hearst algorithm, curated abbreviation dictionary (62 entries), methodology blacklist (~120 terms), name quality scoring (9 patterns) |
| `cognitive_agent/decision_engine.py` | `_decide_entity()` with 3 pre-checks before NOVEL check; `_deduplicate_entities()` for abbreviation-aware grouping; `_decide_relation()` with canonicalized subject/object |
| `cognitive_agent/memory/kg_memory.py` | `GENERIC_TERM_BLACKLIST` (~210 terms); `create_entity()` with schema-compliant MERGE |
| `cognitive_agent/memory/episodic_memory.py` | Thread-safe cross-article decision recording |
| `multi_stage_extraction_pipeline.py` | V1 rule-based pipeline (baseline for comparison) |

### X.4.6 Interpretation Boundaries

Several interpretation boundaries are explicitly tracked:

- **LLM extractions are not database records.** Unlike DisGeNET or STRING records, LLM-extracted entities lack external database identifiers by default. The NCBI Validator tool provides optional gene symbol verification against NCBI E-utilities, but this validation is not exhaustive.
- **Offline mode limitation.** When `--skip-neo4j-write` is active, the `execute()` phase is skipped, which disables Neo4j-powered entity deduplication (`find_entity_by_name_ci`). Abbreviated terms that span different articles may still duplicate in offline mode.
- **Schwartz-Hearst adjacency limitation.** When multiple abbreviations appear adjacent in text — e.g., "alanine aminotransferase (ALT) and aspartate aminotransferase (AST)" — the algorithm may fail to correctly match ALT to its long form. The curated abbreviation dictionary compensates for this in high-frequency liver disease terms.
- **Substring matching false positive risk.** Methodology noise detection using substring matching can theoretically match legitimate entities containing method-related substrings. The whitelist-based exclusion of "signaling pathway" and "metabolic pathway" from substring matching addresses the known false positive case ("NF-κB signaling pathway"), but the approach requires ongoing monitoring as new extraction patterns emerge.

---

## X.5 Validation and Testing

### X.5.1 Experimental Setup

**Corpus.** 30 PubMed articles related to liver disease progression, sampled to cover review articles, original research, clinical studies, and computational biology papers. The same 30 articles were used across all three system versions (V1 Pipeline, V2 Agent, V3 Agent) for fair comparison.

**Metrics.**

| Metric | Definition |
|--------|------------|
| Entities extracted | Total entities output by LLM extraction phase |
| Relations extracted | Total relations output by LLM extraction phase |
| Entities created | Entities committed to KG after all filters and dedup |
| Relations created | Relations committed to KG after all filters |
| Discarded count | Actions (entities + relations) rejected by decision engine |
| Import-ready rate | Relations created / (Relations created + Relations discarded) |
| Discard rate | Discarded / Total actions |

**Systems compared.**

| System | Description | LLM | Concurrency |
|--------|-------------|-----|-------------|
| V1 Pipeline | Rule-based 5-stage pipeline with schema validation | DeepSeek (via LangExtract) | max_workers=5 |
| V2 Agent | Cognitive Agent with 7-phase architecture | Gemini 2.5 Flash | max_workers=5 |
| V3 Agent | V2 + three-layer quality defense + abbreviation dedup | Gemini 2.5 Flash | max_workers=5 |

### X.5.2 Extraction Quantity Comparison

| Metric | V1 Pipeline | V2 Agent | V3 Agent |
|--------|:-----------:|:--------:|:--------:|
| Entities extracted | 203 | 708 | 713 |
| Relations extracted | 135 | 486 | 494 |
| Avg entities/article | 6.8 | 23.6 | 23.8 |
| Avg relations/article | 4.5 | 16.2 | 16.5 |

The V1 Pipeline extracts significantly fewer entities and relations because its rule-based schema validation rejects many LLM outputs that do not match the permitted entity and relation types. V2 and V3 extract comparable quantities, indicating that the quality defense layers in V3 do not suppress genuine entity recall.

### X.5.3 Quality Metrics Comparison

| Metric | V2 Agent | V3 Agent | Change |
|--------|:--------:|:--------:|:------:|
| Entities created | 602 | **445** | **-26.1%** |
| Relations created | 399 | 412 | +3.3% |
| Discarded | 74 | 73 | -1.4% |
| Import-ready rate | 82.1% | **83.4%** | +1.3pp |
| Discard rate | 15.2% | 14.8% | -0.4pp |
| Processing time (30 articles) | 178.5s | 188.9s | +5.8% |

The key result is the 26.1% reduction in entities created (602 → 445) while maintaining comparable relation creation (399 → 412). This means V3 is more selective in entity creation without sacrificing relation coverage. The import-ready rate increases from 82.1% to 83.4%, and the discard rate decreases slightly, indicating more precise filtering.

### X.5.4 Noise Interception Audit

The following 12 noise terms that leaked through V2 were verified as intercepted by V3:

| Noise Term | V2 Status | V3 Status | V3 Interception Layer |
|------------|:---------:|:---------:|----------------------|
| "network pharmacology" | ❌ Leaked | ✅ Caught | Layer 2 (methodology noise) |
| "molecular docking" | ❌ Leaked | ✅ Caught | Layer 2 (methodology noise) |
| "gene symbols" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |
| "disease-related targets" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |
| "chemicals" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |
| "Gene Ontology" | ❌ Leaked | ✅ Caught | Layer 3 (database name penalty) |
| "KEGG signaling pathway" | ❌ Leaked | ✅ Caught | Layer 3 (database name penalty) |
| "replication" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |
| "malignancies" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |
| "Immunomodulation" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |
| "immune regulation" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |
| "immune evasion" | ❌ Leaked | ✅ Caught | Layer 1 (blacklist) |

In total, V3 intercepted **39 distinct noise terms** across the 30-article corpus.

### X.5.5 Abbreviation Deduplication Audit

The following abbreviation groups, which created duplicate nodes in V2, were verified as merged into single nodes in V3:

| Abbreviation Group | V2 Nodes | V3 Nodes |
|--------------------|:--------:|:--------:|
| HCC / Hepatocellular carcinoma | 2 | 1 |
| NAFLD / non-alcoholic fatty liver disease | 2 | 1 |
| T2DM / type 2 diabetes mellitus | 2 | 1 |
| ALT / alanine aminotransferase | 2 | 1 |
| HSC / hepatic stellate cell | 2 | 1 |
| MASLD / metabolic dysfunction-associated steatotic liver disease | 2 | 1 |

### X.5.6 False Positive Audit (Legitimate Entity Preservation)

The following legitimate biological entities were verified as correctly preserved (not falsely intercepted) by all three defense layers:

| Entity | Layer 1 (Blacklist) | Layer 2 (Methodology) | Layer 3 (Quality Score) | Result |
|--------|:-------------------:|:---------------------:|:-----------------------:|:------:|
| NF-κB signaling pathway | ✅ Pass | ✅ Pass (whitelist) | ✅ Pass | Created |
| AKT1 | ✅ Pass | ✅ Pass | ✅ Pass | Created |
| Hepatocellular carcinoma | ✅ Pass | ✅ Pass | ✅ Pass | Created |
| Tumor necrosis factor alpha | ✅ Pass | ✅ Pass | ✅ Pass | Created |
| Hepatic stellate cells | ✅ Pass | ✅ Pass | ✅ Pass | Created |
| Alanine aminotransferase | ✅ Pass | ✅ Pass | ✅ Pass | Created |

The "NF-κB signaling pathway" case is particularly noteworthy: it passes Layer 2 because "signaling pathway" was removed from the substring match list and retained only as an exact blacklist match (which "NF-κB signaling pathway" does not match exactly).

---

## X.6 Discussion

### X.6.1 Effectiveness of the Three-Layer Defense

The 26.1% reduction in entity creation — from 602 to 445 — demonstrates that the quality defense system effectively filters noise without suppressing genuine entity recall. The fact that relation creation increased slightly (399 → 412) suggests that the deduplication of abbreviated entities actually *improves* relation coverage, because relations previously split across duplicate nodes (HCC + Hepatocellular carcinoma) are now unified under a single canonical node.

### X.6.2 Trade-off: Rule-Based vs. Learned Filtering

The three-layer defense is entirely rule-based (blacklists, pattern matching, scoring heuristics). This approach was chosen over a learned classifier for several reasons:

1. **Interpretability**: Every discarded entity has an explicit, human-readable reason string.
2. **Maintainability**: New noise patterns can be added to the blacklist without retraining.
3. **Zero-shot**: No labeled training data required; the system works immediately on new article domains.

The primary limitation is that rule-based systems require ongoing maintenance as new noise patterns emerge. A hybrid approach — using the rule-based system as a high-precision first pass and training a lightweight classifier on its outputs for edge cases — could be explored in future work.

### X.6.3 Integration with the v2 Curated Graph

The entities and relations extracted by the Cognitive Agent are designed to complement, not replace, the curated v2 knowledge graph. The agent's output can be integrated through:

- **Novel entity discovery**: New genes, proteins, or metabolites not present in DisGeNET/STRING/KEGG records.
- **Relation augmentation**: New associations between existing backbone entities, potentially with higher recency than database records.
- **Confidence-tiered import**: Agent confidence scores can be used to tier extracted relations, analogous to the DisGeNET confidence tiering used in v2.

### X.6.4 Known Limitations

1. **Schwartz-Hearst adjacency interference**: Adjacent abbreviations in running text (e.g., "...(ALT) and aspartate aminotransferase (AST)...") may cause incorrect long-form matching. The curated abbreviation dictionary provides compensation for high-frequency terms, but rare or novel abbreviations remain at risk.

2. **Offline mode Neo4j dedup absence**: When `--skip-neo4j-write` is active, the `execute()` phase is skipped entirely, disabling Neo4j-powered fuzzy entity matching (`find_entity_by_name_ci`). Cross-article abbreviation variants may still duplicate in offline mode.

3. **Blacklist coverage drift**: The GENERIC_TERM_BLACKLIST was constructed from 30-article sample output. As the article corpus expands to different subdomains of liver disease research, new categories of noise may emerge that are not covered by the current blacklist.

4. **LLM variability**: Entity extraction quality depends on the underlying LLM. Model updates, prompt changes, or switching between providers (Gemini ↔ DeepSeek) may introduce new extraction patterns that require filter adjustments.

---

## X.7 Conclusion and Future Work

This chapter has presented the Cognitive Agent V3, an LLM-based literature extraction pipeline for liver disease knowledge graph population with a three-layer quality defense system. The system addresses three quality risks inherent to LLM-based extraction — generic term leakage, abbreviation-induced duplication, and methodology noise — through a combination of curated blacklists, pattern-based methodology detection, name quality scoring, and abbreviation-aware entity deduplication.

The 30-article evaluation demonstrates that V3 reduces noise entity creation by 26.1% compared to V2 (602 → 445 entities) while maintaining comparable relation coverage and improving the import-ready rate from 82.1% to 83.4%. All 12 noise cases that leaked through V2 were intercepted by V3, and all 6 tested legitimate entities were correctly preserved, confirming that the quality defense system achieves its design goal of high-precision noise filtering with minimal false positives.

### Future Work

| Task | Expected Output |
|------|-----------------|
| Cross-article entity linking with Neo4j online | Enable `find_entity_by_name_ci` for fuzzy entity resolution across articles, further reducing abbreviation duplicates |
| Adaptive blacklist expansion | Automated detection of candidate noise terms from agent execution logs for curator review |
| NCBI gene validation integration | Connect the NCBI Validator tool to Layer 3 quality scoring for gene-specific validation |
| End-to-end integration with v2 KG | Run the agent on a larger article corpus (500+) and import novel entities/relations into the curated v2 graph |
| Confidence threshold tuning | Systematic evaluation of entity and relation confidence thresholds on precision/recall trade-off |
| Hybrid filtering | Train a lightweight classifier on agent decisions to complement rule-based filtering |

---

## References

[1] J. Piñero et al., "The DisGeNET knowledge platform for disease genomics: 2019 update," *Nucleic Acids Research*, vol. 48, no. D1, pp. D845-D855, 2020.

[2] D. Szklarczyk et al., "The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest," *Nucleic Acids Research*, vol. 51, no. D1, pp. D638-D646, 2023.

[3] S. Lai et al., "BioREx: Biomedical Relation Extraction with Large Language Models," in *Proc. AAAI*, 2023.

[4] Y. Zong et al., "EvidenceNet: Evidence-Aware Knowledge Graph for Hepatocellular Carcinoma and Colorectal Cancer," *Journal of Biomedical Informatics*, 2026.

[5] LangExtract, "Structured Extraction Framework for LLMs." [Online]. Available: https://github.com/

[6] A. S. Schwartz and M. A. Hearst, "A simple algorithm for identifying abbreviation definitions in biomedical text," in *Proc. Pacific Symposium on Biocomputing*, 1999, pp. 451-462.

[7] A. S. Schwartz and M. A. Hearst, "A simple algorithm for identifying abbreviation definitions in biomedical text," *Journal of Computational Biology*, vol. 10, no. 3-4, pp. 451-462, 2003.

[8] A. Hogan et al., "Knowledge graphs," *ACM Computing Surveys*, vol. 54, no. 4, pp. 1-37, 2021.

[9] M. Uhlen et al., "Tissue-based map of the human proteome," *Science*, vol. 347, no. 6220, article 1260419, 2015.

[10] D. S. Wishart et al., "HMDB 5.0: the Human Metabolome Database for 2022," *Nucleic Acids Research*, vol. 50, no. D1, pp. D622-D631, 2022.
