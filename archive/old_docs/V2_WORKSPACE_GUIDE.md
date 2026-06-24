# Liver KG v2 Workspace Guide

This workspace currently contains both earlier v0/v1 artifacts and the current
Neo4j v2 build. Treat the v2 build as the active database version unless a task
explicitly says otherwise.

## Active Database

```text
Neo4j database: liver-kg-core-v02
Neo4j HTTP base: http://100.104.181.96:7474
```

Do not write passwords into scripts, reports, or committed files. Pass
`NEO4J_PASSWORD` through the shell environment when running import commands.

## v2 Build Order

```text
1. DisGeNET disease backbone and Gene-Disease associations
2. Disease progression backbone
3. STRING Gene-Protein mapping and Protein-Protein interactions
4. KEGG Gene-Pathway memberships
5. Reactome Gene-Pathway memberships
6. HPA liver expression and LIHC prognostic context
7. HMDB disease-linked metabolite context
```

This order matters because downstream sources are expanded only from the
DisGeNET-derived Gene set in `liver-kg-core-v02`.

## Current v2 Data Products

| Source | v2 data directory | Main content |
|---|---|---|
| DisGeNET | `data/disgenet_liver_backbone/` | Raw pages, staging JSONL, curated Disease/Gene/GDA TSVs, confidence tiers |
| STRING | `data/string_human_liver_genes_v02/` | Gene-STRING mapping and high-confidence PPI |
| KEGG | `data/kegg_human_liver_genes_v02/` | KEGG pathway nodes and Gene-Pathway memberships |
| Reactome | `data/reactome_human_liver_genes_v02/` | Static Ensembl2Reactome pathway mapping for current Genes |
| HPA | `data/hpa_liver_gene_context_v02/` | Liver tissue expression and LIHC prognostic relations |
| HMDB | `data/hmdb_human_liver_genes_v02/` | Disease-linked metabolites and Gene-Metabolite links |

Earlier non-`_v02` directories are previous-run artifacts and should not be used
as the source of truth for the v2 database unless you are intentionally
comparing versions.

## Active v2 Scripts

Run commands from this folder root (`/Users/buerchener/Desktop/FYP_ontology_v2`).

```bash
# 1. Crawl DisGeNET backbone into staging files.
python3 scripts/crawl_disgenet_liver_backbone.py

# 2. Import curated DisGeNET tables into v2.
NEO4J_PASSWORD='...' python3 scripts/import_disgenet_backbone_neo4j.py --database liver-kg-core-v02

# 3. Import STRING mapping and PPI.
NEO4J_PASSWORD='...' python3 scripts/import_string_neo4j.py \
  --neo4j-url http://100.104.181.96:7474/db/liver-kg-core-v02/tx/commit \
  --output-dir data/string_human_liver_genes_v02 \
  --apply

# 4. Import KEGG pathway memberships.
NEO4J_PASSWORD='...' python3 scripts/import_kegg_neo4j.py \
  --neo4j-url http://100.104.181.96:7474/db/liver-kg-core-v02/tx/commit \
  --output-dir data/kegg_human_liver_genes_v02 \
  --apply

# 5. Import Reactome using the local static Ensembl2Reactome file.
NEO4J_PASSWORD='...' python3 scripts/import_reactome_static_neo4j.py --apply

# 6. Import HPA liver expression and LIHC prognostic context.
NEO4J_PASSWORD='...' python3 scripts/import_hpa_neo4j.py \
  --neo4j-url http://100.104.181.96:7474/db/liver-kg-core-v02/tx/commit \
  --output-dir data/hpa_liver_gene_context_v02 \
  --apply

# 7. Import conservative HMDB disease-linked metabolite context.
NEO4J_PASSWORD='...' python3 scripts/import_hmdb_neo4j.py \
  --hmdb-xml /Users/buerchener/Downloads/hmdb_metabolites.xml \
  --out-dir data/hmdb_human_liver_genes_v02 \
  --neo4j-url http://100.104.181.96:7474/db/liver-kg-core-v02/tx/commit \
  --scope disease-linked
```

## Current v2 Graph Summary

| Node label | Count |
|---|---:|
| Disease | 5 |
| Gene | 836 |
| Protein | 793 |
| Pathway | 1721 |
| Tissue | 1 |
| Metabolite | 36 |

| Relationship type | Count |
|---|---:|
| ASSOCIATED_WITH | 1079 |
| PROGRESSES_TO | 4 |
| ENCODES | 793 |
| INTERACTS_WITH | 7154 |
| PARTICIPATES_IN | 9947 |
| EXPRESSED_IN | 798 |
| PROGNOSTIC_IN | 1384 |
| ASSOCIATED_WITH_METABOLITE | 97 |

## Important Modeling Notes

- Fibrosis is not split into F1/F2/F3/F4 in v2.
- DisGeNET GDA records are from the configured account's curated-access API
  responses.
- STRING mapping is intentionally conservative; unmapped genes should not be
  forced into the PPI layer without manual review.
- Reactome v2 uses the local `Ensembl2Reactome.txt` file from
  `/Users/buerchener/Desktop/data_organized_backup_20260616.zip`, because
  per-gene Reactome ContentService calls were too slow for the 836-gene set.
- HPA v2 uses bulk TSV downloads for tissue expression and cancer prognosis;
  single-cell expression is not imported.
- HMDB v2 uses `scope=disease-linked`, not the full all-gene metabolite
  expansion.
