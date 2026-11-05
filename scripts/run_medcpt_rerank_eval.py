from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# When copied into <project>/scripts/, ROOT becomes the project root.
sys.path.insert(0, str(ROOT))

from libcat.retrieval import TrialRetriever
from libcat.trec import (
    load_topics,
    load_qrels,
    ndcg_at_k,
    eligible_hit_at_k,
    eligible_precision_at_k,
    eligible_recall_at_k,
    eligible_mrr,
    qrel_coverage,
)


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


from libcat.medcpt import MedCPTReranker, build_medcpt_document


def topic_metrics(ranked: list[str], grade_map: dict[str, int], candidate_ids: set[str]) -> dict[str, float]:
    return {
        "nDCG@10": ndcg_at_k(ranked, grade_map, 10),
        "EligibleHit@5": eligible_hit_at_k(ranked, grade_map, 5),
        "EligibleP@10": eligible_precision_at_k(ranked, grade_map, 10),
        "EligibleRecall@10": eligible_recall_at_k(ranked, grade_map, 10),
        "EligibleMRR": eligible_mrr(ranked, grade_map),
        "EligibleQrelCoverage": qrel_coverage(candidate_ids, grade_map, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate stable v4 hybrid retrieval plus NCBI MedCPT biomedical reranking."
    )
    p.add_argument("--trials", default=str(ROOT / "data" / "trec" / "breast_trials.csv"))
    p.add_argument("--topics", default=str(ROOT / "data" / "trec" / "topics2023.xml"))
    p.add_argument("--qrels", default=str(ROOT / "data" / "trec" / "qrels2023.txt"))
    p.add_argument("--base-top-k", type=int, default=100)
    p.add_argument("--rerank-k", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model", default="ncbi/MedCPT-Cross-Encoder")
    p.add_argument("--output-dir", default=str(ROOT / "results"))
    args = p.parse_args()

    trials = pd.read_csv(args.trials).fillna("")
    topics = load_topics(args.topics, template_contains="breast")
    qrels = load_qrels(args.qrels)
    if topics.empty:
        raise SystemExit("No Breast Cancer TREC topics found.")

    print("Loading stable project retriever in hybrid mode...")
    base = TrialRetriever(trials, mode="hybrid")
    reranker = MedCPTReranker(args.model)
    candidate_ids = set(trials["nct_id"].astype(str))

    all_variant_rows: dict[str, list[dict]] = {
        "v4_base": [],
        "medcpt_only": [],
        "v4_medcpt_fusion": [],
    }
    run_rows: list[dict] = []

    for _, topic in topics.iterrows():
        tid = str(topic["topic_id"])
        query = str(topic["query"])
        topic_qrels = qrels[qrels["topic_id"].astype(str) == tid]
        grade_map = dict(
            zip(topic_qrels["nct_id"].astype(str), topic_qrels["grade"].astype(int))
        )

        print(f"\nTopic {tid}: stable v4 retrieval...")
        base_results = base.search(query, top_k=max(args.base_top_k, args.rerank_k)).reset_index(drop=True)
        base_ids = base_results["nct_id"].astype(str).tolist()

        # Baseline metrics are recomputed in the same run for an apples-to-apples comparison.
        base_metrics = topic_metrics(base_ids, grade_map, candidate_ids)
        all_variant_rows["v4_base"].append({
            "topic_id": tid,
            "template": topic["template"],
            **base_metrics,
        })

        n = min(args.rerank_k, len(base_results))
        shortlist = base_results.iloc[:n].copy().reset_index(drop=True)
        docs = [build_medcpt_document(row) for _, row in shortlist.iterrows()]

        print(f"Topic {tid}: MedCPT reranking {n} candidates...")
        med_raw = reranker.scores(query, docs, batch_size=args.batch_size)
        med = minmax(med_raw)
        base_score = minmax(shortlist["score"].to_numpy(dtype=float))

        # Variant A: biomedical cross-encoder only within the stable v4 shortlist.
        med_order = np.argsort(-med)
        med_ids = shortlist.iloc[med_order]["nct_id"].astype(str).tolist()
        # Append the remainder of v4 so MRR can still inspect beyond the reranked shortlist.
        med_ids += base_ids[n:]
        med_metrics = topic_metrics(med_ids, grade_map, candidate_ids)
        all_variant_rows["medcpt_only"].append({
            "topic_id": tid,
            "template": topic["template"],
            **med_metrics,
        })

        # Variant B: preserve v4 constraint-aware evidence while adding biomedical relevance.
        # The fixed 55/45 blend is intentionally predeclared; no per-topic tuning.
        fused = 0.55 * base_score + 0.45 * med
        fused_order = np.argsort(-fused)
        fused_ids = shortlist.iloc[fused_order]["nct_id"].astype(str).tolist()
        fused_ids += base_ids[n:]
        fused_metrics = topic_metrics(fused_ids, grade_map, candidate_ids)
        all_variant_rows["v4_medcpt_fusion"].append({
            "topic_id": tid,
            "template": topic["template"],
            **fused_metrics,
        })

        for rank, idx in enumerate(fused_order, start=1):
            row = shortlist.iloc[int(idx)]
            run_rows.append({
                "variant": "v4_medcpt_fusion",
                "topic_id": tid,
                "nct_id": str(row["nct_id"]),
                "rank": rank,
                "v4_score": float(base_score[int(idx)]),
                "medcpt_score": float(med[int(idx)]),
                "score": float(fused[int(idx)]),
            })

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for variant, rows in all_variant_rows.items():
        df = pd.DataFrame(rows)
        df.to_csv(outdir / f"trec_{variant}_topic_metrics.csv", index=False)
        summary = {
            "variant": variant,
            "candidate_trials": len(trials),
            "breast_topics": len(topics),
            **{
                col: float(df[col].mean())
                for col in df.columns
                if col not in {"topic_id", "template"}
            },
        }
        summaries[variant] = summary

    pd.DataFrame(run_rows).to_csv(outdir / "trec_medcpt_fusion_run.csv", index=False)
    (outdir / "trec_medcpt_comparison.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    print("\n=== MedCPT comparison ===")
    print(json.dumps(summaries, indent=2))
    print("\nSaved separate experiment outputs under results/; stable v4 files were not overwritten.")


if __name__ == "__main__":
    main()
