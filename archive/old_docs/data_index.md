# Data Directory Index

This directory contains generated source-specific data products. The current
active version is v2.

## Active v2 Directories

| Directory | Source | Contents |
|---|---|---|
| `disgenet_liver_backbone/` | DisGeNET | Full five-disease backbone crawl, staging JSONL, curated TSVs, confidence tiers |
| `string_human_liver_genes_v02/` | STRING | Gene-STRING mapping and PPI TSVs |
| `kegg_human_liver_genes_v02/` | KEGG | Pathway nodes and Gene-Pathway memberships |
| `reactome_human_liver_genes_v02/` | Reactome | Static mapping outputs from Ensembl2Reactome |
| `hpa_liver_gene_context_v02/` | Human Protein Atlas | Liver tissue expression and LIHC prognostic context |
| `hmdb_human_liver_genes_v02/` | HMDB | Disease-linked metabolite context |

## Earlier Artifacts

The following directories are previous-run artifacts. Keep them for comparison,
but do not use them as the source of truth for `liver-kg-core-v02`:

```text
disgenet_liver_stages/
string_human_liver_genes/
kegg_human_liver_genes/
reactome_human_liver_genes/
hpa_liver_gene_context/
hmdb_human_liver_genes/
```

## Raw External Files

`hmdb_raw/` is a placeholder for official HMDB raw downloads. The v2 HMDB import
used the local file:

```text
/Users/buerchener/Downloads/hmdb_metabolites.xml
```

The v2 Reactome import used a static mapping file inside:

```text
/Users/buerchener/Desktop/data_organized_backup_20260616.zip
```
