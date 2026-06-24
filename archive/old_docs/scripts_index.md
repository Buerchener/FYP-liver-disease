# Tools Index

This directory contains import and utility scripts for the liver disease
knowledge graph. The current active build is `liver-kg-core-v02`.

## v2 Primary Scripts

| Script | Role | Writes to Neo4j |
|---|---|---|
| `crawl_disgenet_liver_backbone.py` | Crawls the five-disease DisGeNET backbone into raw/staging/curated/tiered files. | No |
| `import_disgenet_backbone_neo4j.py` | Imports curated DisGeNET Disease, Gene, and Gene-Disease association tables. | Yes |
| `import_string_neo4j.py` | Maps Gene to STRING Protein and imports high-confidence PPI. | Yes with `--apply` |
| `import_kegg_neo4j.py` | Imports KEGG human Gene-Pathway memberships. | Yes with `--apply` |
| `import_reactome_static_neo4j.py` | Imports Reactome memberships from the local static Ensembl2Reactome file. | Yes with `--apply` |
| `import_hpa_neo4j.py` | Imports HPA liver tissue expression and LIHC prognostic relations. | Yes with `--apply` |
| `import_hmdb_neo4j.py` | Imports conservative HMDB disease-linked metabolite context. | Yes unless `--build-only` |

## Legacy or Supporting Scripts

| Script | Status |
|---|---|
| `extract_disgenet_hcc_api.py` | Older/smoke DisGeNET extractor; uses `top-n` sampling, not suitable for the final v2 full crawl. |
| `import_reactome_neo4j.py` | API-based Reactome importer; can be slow on the v2 836-gene set. Prefer `import_reactome_static_neo4j.py` for v2. |
| `export_disgenet_neo4j.py` | Exports older normalized DisGeNET JSONL into Neo4j CSV files. |
| `export_neo4j_ontology_schema.py` | Exports ontology/schema metadata Cypher. |
| `audit_disgenet_api_fields.py` | Audits DisGeNET API field shapes. |
| `validate_extracted_data.py` | Validates older extracted data files against ontology rules. |

## Safety Rules

- Use `NEO4J_PASSWORD` as an environment variable; do not write credentials into
  scripts or reports.
- Check the target `--neo4j-url` before running any import. v2 uses:

```text
http://100.104.181.96:7474/db/liver-kg-core-v02/tx/commit
```

- Scripts use `MERGE` for core IDs, so reruns are intended to be mostly
  idempotent. They may still update properties such as `updated_at`.
