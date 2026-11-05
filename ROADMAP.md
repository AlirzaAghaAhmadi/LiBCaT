# LiBCaT development roadmap

LiBCaT is **an early-stage research prototype**. This roadmap lists possible engineering
and research directions, **not completed work, promises, or evidence of clinical performance**.
Contributors can propose or refine milestones in GitHub Issues. Priorities may change.

## Starting point — v0.1.0 (implemented baseline)

- Field-aware BM25 with heuristic clinical-term compatibility/contradiction signals.
- Optional integrations for dense retrieval, general cross-encoding and MedCPT reranking.
- Python API, small synthetic-data Streamlit app and local smoke/unit tests.
- Historical TREC 2023 download, parsing and evaluation scripts.

**Current limits:** no newly trained model; heuristic rather than calibrated clinical
scores; no independently verified full-corpus benchmark for this packaged release;
no production/clinical validation. The standard CI checks the lightweight/offline path.

## Proposed milestone A — Extensible retrieval (candidate v0.2)

| Work item | A useful definition of done |
| --- | --- |
| Pluggable retrieval and reranking interfaces | A new retriever/reranker can be registered without editing the Streamlit app; an example plugin and a unit test exist |
| Config-driven experiments | YAML or JSON config selects components, model identifiers, weights, seed and output path; saved run metadata identifies the exact configuration |
| Custom dataset adapter | A documented schema and loader allow a developer to use their own trial corpus; synthetic fixtures validate parsing and missing-field handling |
| Reusable CLI | A documented command can index/query/evaluate without launching Streamlit; failures return actionable messages |

Prefer small, reviewable changes and preserve the offline BM25 path.

## Proposed milestone B — Reproducible benchmark and analysis

| Work item | A useful definition of done |
| --- | --- |
| TREC end-to-end recipe | Dataset provenance, download steps, checksums, model versions, configuration, commands and output files are documented |
| Baseline and ablation comparisons | Same corpus, qrels, candidate set and metric definitions for lexical, hybrid, clinical signals and reranking variants |
| Error and coverage reports | Per-topic scores, eligible-qrel coverage, missed-relevant cases and examples (without personal information) are exportable |
| Regression checks | Test cases cover graded nDCG, qrel parsing, stable rankings and backward-compatible output schemas |

**Scientific boundary:** no claim of improvement until results are rerun on a properly
specified dataset and compared fairly. Avoid tuning on held-out evaluation labels.

## Proposed milestone C — Broader research reuse

- Better dataset adapters and explicit preprocessing/index versioning.
- Caching or incremental index updates for larger trial collections.
- More reliable trial-status/criteria parsing, with documented provenance and limitations.
- Extend clinical concepts beyond breast cancer only after domain-specific tests and
  evaluation; do not assume existing breast-cancer heuristics generalize.
- Optional research dashboard to compare runs and inspect false matches.

A potential **future** direction is a clinical-trial retrieval experimentation platform
for multiple diseases. Clinical decision-support or patient-facing matching would
require separate clinical evaluation, clinician oversight, privacy/security work,
and applicable regulatory review. It is **out of scope** for the current release.

## Good first contributions

These can become GitHub Issues after the repository is published:

1. Add unit tests for contradictory HER2/ER/PR/status expressions and ambiguous text.
2. Improve errors and documentation for custom CSV field mappings.
3. Add a lightweight CLI wrapper around the existing Python query API.
4. Write a small config schema with validation and one documented example.
5. Add a benchmark run-manifest (dataset/model/config/seed/commit identifiers).
6. Improve Streamlit accessibility and synthetic example explanations.

Before starting, read [CONTRIBUTING.md](CONTRIBUTING.md). Each PR should explain its
scope, limitations, local test commands, and whether neural/full-TREC evaluation
was actually run. Large data, patient records and model weights must not be committed.
