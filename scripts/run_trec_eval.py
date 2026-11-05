from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libcat.retrieval import TrialRetriever
from libcat.trec import (
    load_topics, load_qrels, ndcg_at_k, eligible_hit_at_k,
    eligible_precision_at_k, eligible_recall_at_k, eligible_mrr, qrel_coverage,
)


def main():
    p = argparse.ArgumentParser(description="Evaluate the same retriever on Breast Cancer TREC 2023 topics.")
    p.add_argument("--trials", default=str(ROOT / "data" / "trec" / "breast_trials.csv"))
    p.add_argument("--topics", default=str(ROOT / "data" / "trec" / "topics2023.xml"))
    p.add_argument("--qrels", default=str(ROOT / "data" / "trec" / "qrels2023.txt"))
    p.add_argument("--mode", choices=["lexical", "dense", "hybrid", "auto"], default="auto")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--output-dir", default=str(ROOT / "results"))
    args = p.parse_args()

    for path in (args.trials, args.topics, args.qrels):
        if not Path(path).is_file():
            raise SystemExit(f"Missing input: {path}. See README.md -> TREC evaluation.")
    trials = pd.read_csv(args.trials).fillna("")
    topics = load_topics(args.topics, template_contains="breast")
    qrels = load_qrels(args.qrels)
    if topics.empty:
        raise SystemExit("No topic whose template contains 'breast' was found. Inspect topics2023.xml template names.")

    retriever = TrialRetriever(trials, mode=args.mode)
    candidate_ids = set(trials["nct_id"].astype(str))
    rows = []
    run_rows = []

    for _, topic in topics.iterrows():
        tid = str(topic["topic_id"])
        topic_qrels = qrels[qrels["topic_id"].astype(str) == tid]
        grade_map = dict(zip(topic_qrels["nct_id"].astype(str), topic_qrels["grade"].astype(int)))
        results = retriever.search(topic["query"], top_k=args.top_k)
        ranked = results["nct_id"].astype(str).tolist()
        metrics = {
            "topic_id": tid,
            "template": topic["template"],
            "nDCG@10": ndcg_at_k(ranked, grade_map, 10),
            "EligibleHit@5": eligible_hit_at_k(ranked, grade_map, 5),
            "EligibleP@10": eligible_precision_at_k(ranked, grade_map, 10),
            "EligibleRecall@10": eligible_recall_at_k(ranked, grade_map, 10),
            "EligibleMRR": eligible_mrr(ranked, grade_map),
            "EligibleQrelCoverage": qrel_coverage(candidate_ids, grade_map, 2),
        }
        rows.append(metrics)
        for _, r in results.iterrows():
            run_rows.append({"topic_id": tid, "nct_id": r["nct_id"], "rank": int(r["rank"]), "score": float(r["score"])})

    metrics_df = pd.DataFrame(rows)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(outdir / "trec_topic_metrics.csv", index=False)
    pd.DataFrame(run_rows).to_csv(outdir / "trec_run.csv", index=False)
    summary = {
        "retrieval_mode_requested": args.mode,
        "retrieval_mode_active": retriever.mode_active,
        "candidate_trials": len(trials),
        "breast_topics": len(topics),
        **{col: float(metrics_df[col].mean()) for col in metrics_df.columns if col not in {"topic_id", "template"}},
    }
    (outdir / "trec_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["EligibleQrelCoverage"] < 0.90:
        print("WARNING: breast candidate restriction covers <90% of eligible qrels. Improve the candidate-table rule before reporting final metrics.")


if __name__ == "__main__":
    main()
