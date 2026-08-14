# Agent v3 architecture

## End-to-end data flow

```text
PubMed article
  │
  ├─ content-addressed preprocessing
  │    ├─ study profile
  │    ├─ exact evidence units
  │    ├─ abbreviations
  │    └─ fixed golden examples
  │
  ├─ Gemini primary extraction (cached, chunked)
  │    └─ entity inventory + relation hints
  │
  ├─ EvidenceSelector
  │    └─ minimal exact span → grounded endpoints → schema-valid pairs
  │
  ├─ deterministic pair classifier + active soft-rule priors
  │    └─ predicate / NO_RELATION + separate relation/evidence scores
  │
  ├─ local entailment
  │    ├─ clear decision → verifier
  │    └─ uncertain batch → DeepSeek → optional Qwen conflict critic
  │
  ├─ ConformalRiskRouter
  │    ├─ ACCEPT_LOCAL
  │    ├─ CALL_DEEPSEEK
  │    ├─ CALL_QWEN_CRITIC
  │    └─ ABSTAIN / HUMAN_REVIEW
  │
  ├─ deterministic Verifier after every modification
  ├─ optional read-only RAG / Causal / Conflict research tools
  ├─ Decision Engine
  └─ dry-run or Safe Write
```

## Component responsibilities

| Component | Owns | Cannot do |
| --- | --- | --- |
| Gemini extractor | Initial entity inventory and relation hints | Grant write authority |
| RuleMemory | Frozen soft guidance, priors and downgrade routes | Change schema or hard gates |
| DeepSeek | Rule induction and primary ambiguous-case adjudication | Add arbitrary endpoints or evidence |
| Qwen | Rule criticism, compression and conflict adjudication | Promote a rule alone |
| EvidenceSelector | Exact minimal spans and offsets | Paraphrase or cross paragraphs |
| Pair classifier | Schema-masked predicate / `NO_RELATION` scores | Override evidence validity |
| Conformal router | Add calls, review or abstention | Upgrade a verifier failure |
| Verifier | Schema, endpoint, source, negation and section checks | Learn online rules |
| Causal tool | Separate hypotheses and inference chains | Create import-ready facts without evidence |
| Conflict tool | Create/keep/update/dispute/review suggestions | Bypass re-verification |
| Decision Engine | Final dry-run/write decisions | Write when `import_ready=false` |

## Runtime modes

Compatibility defaults remain unchanged:

```text
--execution-mode legacy
--rule-memory-mode off
--evidence-entailment-mode off
--risk-router-mode off
```

Research progression:

```text
1. agent-v2-shadow + rule-memory shadow + risk-router shadow
2. fixed-candidate replay and rule promotion between runs
3. agent-v2 dry-run + rule-memory active + entailment active
4. frozen calibration + risk-router active, still dry-run
5. one preregistered blind run after expert labels are sealed
```

Active Agent v2 and the active pair classifier are code-restricted to dry-run.
The Safe Write gate remains a separate explicit choice and is restricted to a
local Neo4j database.

## Rule bundle and offline learning

Each bundle contains a revision, lifecycle state, previous-bundle hash and
closed-DSL rules. An evaluation process loads it once and freezes its hash.
The runtime never promotes or edits a rule. Offline learning consumes only
induction ErrorCards; validation and calibration outputs are used solely by the
promotion gate.

Artifacts:

```text
active_rules.json          machine-readable frozen bundle
RULE_CONTEXT.md            bounded human-readable view
rule_learning_audit.json   inducer, critic and gate decisions
```

Malformed bundles fail closed and disable rule memory. If Qwen is unavailable,
an old validated bundle may still be used, but no new rule can be promoted.

## Cache and call accounting

Primary extraction and auxiliary actions use content/configuration-aware keys.
The rule-bundle hash is part of the primary prompt and therefore invalidates
the extraction cache whenever active guidance changes. Credentials never enter
keys, values, traces or reports.

The report records model attempts, successes, invalid JSON, token counts,
latency, local/remote entailment sources, rule matches, conformal group fallback
and decisions. A cache hit consumes no remote budget. Repeated identical states
are deduplicated, and two unchanged auxiliary rounds terminate further calls.

## Failure modes

| Failure | Behaviour |
| --- | --- |
| Rule JSON/DSL invalid | Reject rule or whole bundle; keep previous bundle |
| DeepSeek unavailable | Local abstention or old bundle; no invented answer |
| Qwen unavailable | No automatic rule promotion; conflicts go to review |
| Returned quote not in source | Discard remote decision and retain NEI |
| Mondrian group <20 | Use global calibration and report fallback |
| Calibration absent | No certified local acceptance; route/review/abstain |
| Model disagreement persists | Human review |
| Hard verifier flag | Abstain regardless of model/rule score |
| Cache corruption | Evict and recompute deterministic component |
| Blind leakage detected | Abort experiment before inference |

## Research artifacts

- Development split: `gold_annotations/splits/agent_v3_dev_manifest_seed20260814.json`
- Blind preregistration: `gold_annotations/blind50/blind50_preregistered_seed20260814.json`
- Blank expert template: `gold_annotations/blind50/blind50_annotation_template.jsonl`
- Rule learning: `scripts/learn_agent_rules.py`
- Calibration: `scripts/fit_conformal_router.py`
- Leakage check: `scripts/check_agent_v3_leakage.py`
- Ablation/statistics: `scripts/evaluate_agent_v3_experiments.py`

