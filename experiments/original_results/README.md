# Archived outputs from the supplied prototype

These CSV and JSON files are preserved from the original project archive for comparison.
They are **not reproduced or independently validated by the unit tests** and may depend
on a historical candidate table and model checkpoints not included here. Never cite
these as benchmarked performance of a new installation. Recompute using the full
TREC 2023 corpus and the exact evaluation configuration.

**Metric compatibility warning (2026):** This folder preserves legacy runs.
The `libcat.trec.dcg` implementation was corrected to TREC 2023's *linear*
relevance gains (0, 1, 2) before the first public release. The scores in
this folder have **not** been regenerated after that correction and must not
be described as directly comparable to fresh `nDCG@10` output.

## Latest recorded three-variant comparison

The file [`trec_medcpt_comparison.json`](trec_medcpt_comparison.json) contains
our latest **retained** comparison on 12,249 candidate trials and 5 breast-cancer
TREC 2023 topics (16–20):

| Variant | Recorded nDCG@10 | EligibleHit@5 | EligibleP@10 | EligibleMRR |
| --- | ---: | ---: | ---: | ---: |
| `v4_base` | 0.3359 | 0.60 | 0.20 | 0.5274 |
| `medcpt_only` | 0.3347 | 0.80 | 0.20 | 0.5051 |
| `v4_medcpt_fusion` | 0.3547 | 0.60 | 0.20 | 0.5274 |

The mean values match the corresponding five-row `*_topic_metrics.csv` files.
This is a historical experiment **record**, not a fresh or independently
validated evaluation. For reproducible comparisons, rerun each variant using
one pinned corpus, model set, and metric implementation, and record a run manifest.
