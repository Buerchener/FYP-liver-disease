# Liver KG v2 Backbone Scope Audit

Database: `liver-kg-core-v02`

Audit date: `2026-06-24`

## Conclusion

Current v2 data passes the backbone scope audit.

No unrelated disease nodes, orphan nodes, or out-of-backbone disease targets were found. Every non-disease layer is traceable back to genes associated with the five backbone diseases.

Backbone diseases:

- `UMLS:C0400966` / `NAFLD`
- `UMLS:C3241937` / `NASH`
- `UMLS:C0239946` / `Fibrosis`
- `UMLS:C0023890` / `Cirrhosis`
- `UMLS:C2239176` / `HCC`

Important interpretation note: STRING, KEGG, Reactome, HPA, and HMDB gene-metabolite data are backbone-derived context layers. They are in scope because they are attached to backbone genes or five backbone diseases, but individual PPI/pathway/expression records should not be described as direct disease evidence unless the relation itself carries disease evidence.

## Scope Rules Used

| Layer | Required scope condition | Result |
|---|---|---|
| Disease | Only the five backbone Disease nodes may exist. | Pass |
| Progression | `PROGRESSES_TO` must connect only backbone Disease nodes. | Pass |
| Gene | Every Gene must have at least one `Gene -[:ASSOCIATED_WITH]-> Disease` edge to a backbone Disease. | Pass |
| Protein | Every Protein must be encoded by a backbone Gene. | Pass |
| PPI | Every PPI endpoint must be a Protein encoded by a backbone Gene. | Pass |
| Pathway | Every Pathway must have at least one incoming `Gene -[:PARTICIPATES_IN]-> Pathway` from a backbone Gene. | Pass |
| HPA expression | Every expression edge must start from a backbone Gene. | Pass |
| HPA prognosis | Every prognostic edge must start from a backbone Gene and target a backbone Disease. | Pass |
| Metabolite | Every Metabolite must be linked to a backbone Gene and to at least one backbone Disease. | Pass |

## Key Counts

| Label | Count |
|---|---:|
| Disease | 5 |
| Gene | 836 |
| Protein | 793 |
| Pathway | 1721 |
| Tissue | 1 |
| Metabolite | 36 |

| Relationship | Count |
|---|---:|
| `ASSOCIATED_WITH` | 1079 |
| `PROGRESSES_TO` | 4 |
| `ENCODES` | 793 |
| `INTERACTS_WITH` | 7154 |
| `PARTICIPATES_IN` | 9947 |
| `EXPRESSED_IN` | 798 |
| `PROGNOSTIC_IN` | 1384 |
| `ASSOCIATED_WITH_METABOLITE` | 97 |

## Disease Checks

Disease nodes found:

- `UMLS:C0400966:NAFLD`
- `UMLS:C3241937:NASH`
- `UMLS:C0239946:Fibrosis`
- `UMLS:C0023890:Cirrhosis`
- `UMLS:C2239176:HCC`

Out-of-backbone Disease nodes: `0`

Progression edges:

- `NAFLD -> NASH`
- `NASH -> Fibrosis`
- `Fibrosis -> Cirrhosis`
- `Cirrhosis -> HCC`

Out-of-backbone progression edges: `0`

## Gene Backbone Checks

Gene nodes: `836`

Genes without Disease association: `0`

Genes associated with non-backbone Disease nodes: `0`

Gene-disease association distribution:

| Disease | Disease ID | Edges | Distinct genes |
|---|---|---:|---:|
| NAFLD | `UMLS:C0400966` | 103 | 103 |
| NASH | `UMLS:C3241937` | 107 | 107 |
| Fibrosis | `UMLS:C0239946` | 2 | 2 |
| Cirrhosis | `UMLS:C0023890` | 169 | 169 |
| HCC | `UMLS:C2239176` | 655 | 655 |

The `ASSOCIATED_WITH` relationship currently has two endpoint patterns:

| Start | End | Source | Count |
|---|---|---|---:|
| Gene | Disease | DisGeNET | 1036 |
| Metabolite | Disease | HMDB | 43 |

This is not out-of-scope, but it is semantically mixed. It may be worth splitting the HMDB metabolite-disease relation into a more specific type later.

## Downstream Layer Checks

Protein:

- Proteins: `793`
- Proteins without encoding Gene: `0`
- Proteins from Genes without Disease association: `0`
- PPI edges with unencoded endpoint: `0`

Pathway:

- Pathways: `1721`
- Pathways without incoming backbone Gene membership: `0`
- Pathways from Genes without Disease association: `0`
- Gene-pathway memberships from Genes without Disease association: `0`

HPA:

- Tissue expression edges: `798`
- Expression edges from Genes without Disease association: `0`
- Prognostic edges: `1384`
- Prognostic target non-backbone Disease nodes: `0`
- Prognostic edges from Genes without Disease association: `0`

HMDB:

- Metabolites: `36`
- Metabolites without Gene link: `0`
- Metabolites without Disease link: `0`
- Metabolites linked to non-backbone Disease nodes: `0`
- Metabolites linked from Genes without Disease association: `0`

Metabolite-disease distribution:

| Disease | Disease ID | Edges | Distinct metabolites |
|---|---|---:|---:|
| NAFLD | `UMLS:C0400966` | 17 | 17 |
| Cirrhosis | `UMLS:C0023890` | 18 | 18 |
| HCC | `UMLS:C2239176` | 8 | 8 |

No HMDB disease-metabolite edges were found for NASH or Fibrosis in the current conservative HMDB disease-linked subset. That is absence of matching HMDB evidence, not an out-of-scope import.

## Residual Caveats

1. Pathways are not disease-exclusive pathways. They are pathways containing at least one backbone Gene.
2. STRING PPI edges are protein interaction context among mapped backbone proteins. They are not direct disease evidence.
3. HPA liver expression is tissue context for backbone Genes. HPA LIHC prognosis targets HCC, but it should still be treated as HPA cancer-prognostic context rather than DisGeNET disease association.
4. HMDB currently uses the conservative disease-linked scope. It is not a full metabolite expansion.
5. `ASSOCIATED_WITH` is overloaded across Gene-Disease and Metabolite-Disease edges. Scope is valid, but schema semantics could be cleaner if metabolite-disease edges get their own relation type.

Raw audit JSON: `reports/v2_backbone_scope_audit_20260624.json`
