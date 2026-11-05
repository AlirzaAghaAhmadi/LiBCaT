# Working with large TREC data

The historical **TREC Clinical Trials 2023** collection consists of six corpus ZIP
archives (approximately 2 GB in total), plus comparatively small topic and judgment files.
These archives are **inputs, not source code**. They must not be tracked by Git.

## Recommended layout (local only)

```text
data/
  sample/sample_trials.csv       # small synthetic sample tracked in Git
  demo_trials.csv                # generated, ignored by Git
  demo_snapshot_meta.json        # generated, ignored by Git
  trec/
    ClinicalTrials.2023-05-08.trials0.zip  # ... through trials5.zip
    topics2023.xml
    qrels2023.txt
    breast_trials.csv            # generated from historical corpus
```

## Two ways to prepare the evaluation corpus

**Reuse your existing local 2-3 GB files:** copy/move the six ZIPs into `data/trec/`
with the original filenames, then obtain topics and qrels via
`python scripts/download_trec_data.py` (it skips files already present).

**Fetch directly from the official sources:** run
`python scripts/download_trec_data.py --corpus` to fetch small metadata and the full
historical corpus; then run `python scripts/parse_trec_corpus.py`.

The parser refuses incomplete 6-part downloads by default; `--allow-partial` exists
for development and must not be used to report final metrics.

## Should you upload a copy to Hugging Face or Zenodo?

Not automatically. Prefer linking to the upstream TREC download and preserving a
clear provenance trail. If you want to republish the full dataset, first verify
redistribution permissions and required attribution with the data owner. Hosting
policies, quotas, access restrictions and upload costs may apply. A dataset card
should include source, collection date, split semantics, and checksums. The MIT
license in this repository licenses **our source code only**, not upstream datasets,
third-party pretrained model weights or any external service.

## Check size before committing

```bash
git status --short
# No files under data/trec or data/demo_trials.csv should appear.
```

To work with alternative external datasets, adapt the columns outlined in
`docs/integrate-your-data.md` and create your own loading script.
