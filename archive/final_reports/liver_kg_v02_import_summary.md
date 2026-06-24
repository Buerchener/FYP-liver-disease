# Liver KG v2 Import Summary

Database: `liver-kg-core-v02`

## Source Layers

| Source | Imported layer | Output directory |
|---|---|---|
| DisGeNET | Disease, Gene, Gene-Disease associations | `data/disgenet_liver_backbone/` |
| STRING | Gene-Protein mapping, PPI | `data/string_human_liver_genes_v02/` |
| KEGG | Gene-Pathway memberships | `data/kegg_human_liver_genes_v02/` |
| Reactome | Gene-Pathway memberships | `data/reactome_human_liver_genes_v02/` |
| HPA | Liver expression, LIHC prognostic context | `data/hpa_liver_gene_context_v02/` |
| HMDB | Disease-linked metabolite context | `data/hmdb_human_liver_genes_v02/` |

## Node Counts

| Label | Count |
|---|---:|
| Disease | 5 |
| Gene | 836 |
| Protein | 793 |
| Pathway | 1721 |
| Tissue | 1 |
| Metabolite | 36 |

## Relationship Counts

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

## Source-Specific Counts

| Layer | Count |
|---|---:|
| DisGeNET high-confidence GDA | 621 |
| DisGeNET medium-confidence GDA | 415 |
| STRING mapped genes | 793 |
| STRING PPI edges | 7154 |
| KEGG mapped genes | 595 |
| KEGG pathways | 333 |
| KEGG memberships | 5310 |
| Reactome mapped genes | 677 |
| Reactome pathways | 1388 |
| Reactome memberships | 4637 |
| HPA liver tissue expression | 798 |
| HPA LIHC prognostic relations | 1384 |
| HMDB disease-linked metabolites | 36 |
| HMDB Gene-Metabolite relations | 97 |
| HMDB Metabolite-Disease relations | 43 |

## Disease Progression Backbone

```text
NAFLD -> NASH -> Fibrosis -> Cirrhosis -> HCC
```

`Healthy liver` is not included in the v2 Neo4j data because this v2 build only
imports the five disease backbone nodes.

## Known Constraints

- DisGeNET data comes from the account's curated-access API response.
- Candidate and rejected DisGeNET tier files are currently empty under the v2
  balanced filter.
- STRING leaves some genes unmapped, including `VEGFA`; this is intentional
  under the conservative mapping rule.
- HPA single-cell expression is not imported.
- HMDB is not full HMDB; it is restricted to disease-linked metabolites.
