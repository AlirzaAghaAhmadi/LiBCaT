"""Explicit opt-in integration check for local neural models (large downloads).

Run after: python -m pip install -e ".[all]"
This is NOT part of default CI because it may download model weights (>1 GB total).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libcat import TrialRetriever, build_patient_query
from libcat.medcpt import MedCPTReranker, rerank_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("hybrid", "biomedical"), default="biomedical",
        help="hybrid checks MiniLM embedding and general cross-encoder; biomedical adds MedCPT",
    )
    args = p.parse_args()

    trials = pd.read_csv(ROOT / "data/sample/sample_trials.csv").fillna("")
    query = build_patient_query({
        "diagnosis": "breast cancer",
        "stage": "Stage IV / metastatic",
        "her2": "positive",
        "performance_status": "ECOG 1",
    })

    print("Loading local hybrid retriever (first run downloads pretrained weights)...", flush=True)
    retriever = TrialRetriever(trials, mode="hybrid")
    if retriever.mode_active != "hybrid" or retriever.dense is None or retriever.reranker is None:
        raise SystemExit(
            "Incomplete Hybrid pipeline: dense and cross-encoder models must both load. "
            "Check optional dependencies, network access and Hugging Face cache."
        )
    results = retriever.search(query, top_k=5)
    assert len(results) == 5 and results["nct_id"].notna().all()
    print("PASS: hybrid embeddings + general cross-encoder")

    if args.mode == "biomedical":
        print("Loading MedCPT cross-encoder (first run downloads pretrained weights)...", flush=True)
        reranker = MedCPTReranker()
        reranked = rerank_results(query, results, reranker, rerank_k=5)
        assert len(reranked) == 5 and reranked["medcpt_score"].notna().all()
        print(f"PASS: MedCPT biomedical reranking on {reranker.device_name}")


if __name__ == "__main__":
    main()
