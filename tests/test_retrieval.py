from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libcat.retrieval import TrialRetriever
from libcat.text_utils import build_patient_query, tokenize


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(ROOT / "data" / "sample" / "sample_trials.csv").fillna("")

    def test_stage_iv_does_not_create_stage_i(self):
        toks = set(tokenize("Stage IV / metastatic breast cancer"))
        self.assertIn("stage_iv", toks)
        self.assertNotIn("stage_i", toks)

    def test_her2_metastatic_retrieval(self):
        query = build_patient_query({
            "diagnosis": "breast cancer",
            "stage": "Stage IV / metastatic",
            "her2": "positive",
            "er": "positive",
            "pr": "positive",
            "previous_chemotherapy": "yes",
            "previous_radiotherapy": "no",
            "prior_surgery": "no",
            "treatment_setting": "metastatic",
            "performance_status": "ECOG 1",
        })
        r = TrialRetriever(self.df, mode="lexical").search(query, top_k=5)
        self.assertEqual("SYNTH001", r.iloc[0]["nct_id"])
        rank_map = dict(zip(r["nct_id"], r["rank"]))
        self.assertLess(rank_map["SYNTH001"], rank_map.get("SYNTH003", 999))
        self.assertLess(rank_map["SYNTH008"], rank_map.get("SYNTH003", 999))

    def test_her2_positive_beats_her2_low_and_negative(self):
        query = build_patient_query({
            "diagnosis": "breast cancer",
            "stage": "Stage IV / metastatic",
            "her2": "positive",
            "previous_chemotherapy": "yes",
        })
        r = TrialRetriever(self.df, mode="lexical").search(query, top_k=6)
        rank_map = dict(zip(r["nct_id"], r["rank"]))
        self.assertLess(rank_map["SYNTH001"], rank_map["SYNTH004"])
        self.assertLess(rank_map["SYNTH008"], rank_map.get("SYNTH006", 999))

    def test_triple_negative_retrieval(self):
        query = build_patient_query({
            "diagnosis": "triple negative breast cancer",
            "stage": "metastatic",
            "her2": "negative",
            "er": "negative",
            "pr": "negative",
        })
        r = TrialRetriever(self.df, mode="lexical").search(query, top_k=3)
        self.assertEqual("SYNTH002", r.iloc[0]["nct_id"])

    def test_negative_history_does_not_search_for_therapy(self):
        query = build_patient_query({
            "diagnosis": "breast cancer",
            "previous_radiotherapy": "no",
            "prior_surgery": "no",
        })
        self.assertNotIn("radiotherapy", query.lower())
        self.assertNotIn("surgery", query.lower())


if __name__ == "__main__":
    unittest.main()