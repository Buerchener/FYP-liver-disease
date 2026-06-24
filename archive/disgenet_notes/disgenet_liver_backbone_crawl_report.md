# DisGeNET Liver Backbone Crawl Report

- Retrieved at: `2026-06-23T13:42:41.932547+00:00`
- Endpoint: `https://api.disgenet.com/api/v1/gda/summary`
- Scope: NAFLD, NASH, Fibrosis, Cirrhosis, HCC
- Access note: academic account responses reported curated-source access.

## Crawl Counts

| Stage | UMLS CUI | API totalElements | Rows crawled |
|---|---:|---:|---:|
| NAFLD | C0400966 | 103 | 103 |
| NASH | C3241937 | 107 | 107 |
| Fibrosis | C0239946 | 2 | 2 |
| Cirrhosis | C0023890 | 169 | 169 |
| HCC | C2239176 | 655 | 655 |

## Tier Counts

| Stage | High | Medium | Candidate | Rejected |
|---|---:|---:|---:|---:|
| NAFLD | 53 | 50 | 0 | 0 |
| NASH | 43 | 64 | 0 | 0 |
| Fibrosis | 0 | 2 | 0 | 0 |
| Cirrhosis | 56 | 113 | 0 | 0 |
| HCC | 469 | 186 | 0 | 0 |

## Output Summary

- Disease rows: `5`
- Deduplicated Gene rows: `836`
- Import-scope Gene-Disease associations: `1036`
- Raw merged GDA rows: `1036`
- High confidence: `621`
- Medium confidence: `415`
- Candidate: `0`
- Rejected: `0`

## Mapping Risks

- Genes missing Ensembl IDs, affecting HPA mapping: `8`
- Genes missing DisGeNET protein crossrefs, affecting STRING/Reactome mapping: `37`
- Missing Ensembl sample: LOC110806263(110806263), LOC126806658(126806658), LOC126806659(126806659), LOC129994371(129994371), LOC129997612(129997612), GSTT1(2952), KIR3DS1(3813), ND5(4540)
- Missing protein crossref sample: MIR875(100126309), MIR665(100126315), MIR764(100313838), MIR203B(100616173), LOC110806263(110806263), LOC126806658(126806658), LOC126806659(126806659), LOC129994371(129994371), LOC129997612(129997612), PPP4R3C(139420), IFNA1(3439), MIR122(406906), MIR133A1(406922), MIR146A(406938), MIR153-2(406945), MIR155(406947), MIR193A(406968), MIR195(406971), MIR196A2(406973), MIR200B(406984)

## Import Recommendation

- Import only `high_confidence` and `medium_confidence` associations into the core graph.
- Keep `candidate` as staging evidence for later review.
- Keep `rejected` only for audit; do not import it.
- Run STRING mapping conservatively after this import, using stable protein crossrefs before gene-symbol fallback.
