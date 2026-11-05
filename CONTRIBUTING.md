# Contributing to LiBCaT

LiBCaT is a research prototype for reproducible *breast-cancer clinical-trial
retrieval*. Contributions are welcome, especially tests, retrieval methods,
data adapters, evaluation tools, documentation and reproducibility improvements.

Start with the **[step-by-step developer guide in README](README.md#developer-workflow)**
and the **[development roadmap](ROADMAP.md)** for possible next contributions.
It includes a file map, executable commands, and a synthetic TREC evaluation.

## Before you code

1. Check open Issues and existing Pull Requests for duplicates. For substantial
   changes, open an Issue first with the proposed interface and validation plan.
2. Fork the repository if you do not have write access; invited collaborators can
   work with feature branches in the upstream repository.
3. Use Python 3.10+ and install `python -m pip install -e ".[dev,web]"` inside a
   virtual environment. Neural dependencies are optional (`.[dense]`).
4. Make focused changes on a branch named `feat/...`, `fix/...` or `docs/...`.

## Required local checks

```bash
python -m pytest -q
python scripts/smoke_test.py
python examples/custom_corpus.py
python scripts/run_trec_eval.py --mode lexical \
  --trials tests/fixtures/mini_trials.csv \
  --topics tests/fixtures/mini_topics.xml \
  --qrels tests/fixtures/mini_qrels.txt \
  --output-dir results/dev-fixture
```

The tests and fixtures are synthetic and can run without downloading the full
TREC collection or neural models. If your PR changes Streamlit, also check
`streamlit run app.py` in **Lexical (offline)** mode. If you touch optional neural
paths, report any model/hardware/configuration used to validate the change, and
state clearly if the heavy path was not tested. The opt-in check is
`python scripts/verify_optional_models.py --mode biomedical` (requires downloading
large model weights).

## Pull Request checklist

- [ ] The PR describes the problem, approach, scope and any limitations.
- [ ] Public API changes are documented, with a runnable example if relevant.
- [ ] New behavior includes tests based on synthetic, non-identifying data.
- [ ] Offline lexical behavior and existing tests still pass.
- [ ] Metric changes include definitions, dataset provenance and evaluation setup.
- [ ] No large datasets, real patient data, credentials or model weights are committed.
- [ ] CI passes on the PR before merging into `main`.

## Data, evaluation and safety

- Keep the 2023 historical TREC corpus distinct from the current ClinicalTrials.gov
  demo snapshot. Only compare runs generated against the *same* evaluation setup.
- TREC qrels are graded relevance judgments, **not** proof of clinical eligibility.
  Preserve the eligible-only metrics and qrel-coverage interpretation.
- Do not tune parameters using held-out test judgments and then present the same
  judgments as an unbiased test. Document splits, seeds, model versions and weights.
- Never put patient-identifying information in fixtures, Issues, logs, or PRs.
- Respect third-party dataset/model licenses and redistribution restrictions.
- LiBCaT results are research rankings, not medical advice or eligibility decisions.

## Maintainers and collaboration

Invite a collaborator through GitHub **Settings → Collaborators and teams** after
publishing the repository. Keep `main` protected with PR review and CI when
available. Contributors should use PRs (rather than direct pushes to `main`) so
changes are reviewable and credited to their GitHub account.

If an issue contains sensitive patient information, do not repost the details;
follow [SECURITY.md](SECURITY.md) instead.
