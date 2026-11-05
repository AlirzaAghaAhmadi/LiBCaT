# Reusing the LiBCaT retriever

LiBCaT exposes `libcat.TrialRetriever` and `libcat.build_patient_query` as a Python API.
The retriever expects a `pandas.DataFrame` of candidate trials and accepts the following:

| Field | Required? | Purpose |
| --- | --- | --- |
| `retrieval_text` | yes | Full text used for lexical/dense retrieval |
| `nct_id` | required for standard outputs and TREC evaluation | Stable record identifier |
| `title`, `conditions`, `brief_summary`, `eligibility` | recommended | Field-aware BM25 and clinical evidence |
| `official_title`, `interventions` | optional | Additional scoring/rerank text |
| `trial_url`, `phase`, `status`, `countries`, `cities` | optional | Display metadata |

```python
import pandas as pd
from libcat import TrialRetriever, build_patient_query

trials = pd.read_csv("my_trials.csv").fillna("")
query = build_patient_query({"diagnosis": "breast cancer", "her2": "positive"})
retriever = TrialRetriever(trials, mode="lexical")
print(retriever.search(query, top_k=10)[["nct_id", "score", "rank"]])
```

`TrialRetriever` is built once per candidate table, then reused for many queries.
Available modes: `lexical` (no model), `dense` (sentence embeddings), `hybrid`
(dense + BM25 + cross-encoder), and `auto` (prefers hybrid, falls back to lexical).
Modes other than `lexical` require `pip install -e ".[dense]"` and may download model
weights on first use. Dense embeddings are cached under `~/.cache/libcat/`, or
`LIBCAT_CACHE_DIR` if set.

**Important limitation:** the clinical concept parser and reranking weights were
engineered for breast-cancer trial retrieval (HER2/ER/PR, stage, ECOG, treatment).
Other diseases and nonclinical corpora can supply `retrieval_text`, but the scoring
heuristics must be reviewed and retested for that domain. Clinical contradiction
features are heuristic text signals, not an eligibility checker. No patient-identifying
information belongs in the repository or demo fixtures.
