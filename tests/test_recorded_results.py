"""Check the recorded research summary against its retained per-topic results."""

import json
from pathlib import Path

import pandas as pd


def test_recorded_trec_summary_matches_per_topic_csv():
    archive = Path(__file__).resolve().parents[1] / "experiments" / "original_results"
    summary = json.loads((archive / "trec_medcpt_comparison.json").read_text())
    variants = {
        "v4_base": "trec_v4_base_topic_metrics.csv",
        "medcpt_only": "trec_medcpt_only_topic_metrics.csv",
        "v4_medcpt_fusion": "trec_v4_medcpt_fusion_topic_metrics.csv",
    }
    metrics = ("nDCG@10", "EligibleHit@5", "EligibleP@10",
               "EligibleMRR", "EligibleQrelCoverage")
    for name, csv_filename in variants.items():
        rows = pd.read_csv(archive / csv_filename)
        assert len(rows) == summary[name]["breast_topics"] == 5
        assert summary[name]["candidate_trials"] == 12249
        assert set(rows["topic_id"].astype(int)) == {16, 17, 18, 19, 20}
        for metric in metrics:
            assert abs(float(rows[metric].mean()) - float(summary[name][metric])) < 1e-10
