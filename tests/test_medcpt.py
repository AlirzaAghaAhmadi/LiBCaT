"""Offline unit tests of MedCPT data preparation and ordering, without model download."""
import unittest

import numpy as np
import pandas as pd

from libcat.medcpt import build_medcpt_document, rerank_results


class StubReranker:
    def scores(self, query, documents, batch_size=8):
        # Intentionally reverse the original top two (no actual model loading).
        return np.array([0.1, 0.9])


class MedCPTPipelineTests(unittest.TestCase):
    def test_document_contains_study_fields(self):
        doc = build_medcpt_document(pd.Series({
            "title": "Study A", "conditions": "breast cancer",
            "eligibility": "ECOG <= 2", "brief_summary": "sample description",
        }))
        self.assertIn("Study A", doc)
        self.assertIn("Conditions: breast cancer", doc)
        self.assertIn("ECOG <= 2", doc)

    def test_rerank_changes_order_and_keeps_base_scores(self):
        base = pd.DataFrame([
            {"nct_id": "SYNTH001", "title": "A", "rank": 1, "score": .8},
            {"nct_id": "SYNTH002", "title": "B", "rank": 2, "score": .5},
        ])
        out = rerank_results("breast cancer", base, StubReranker(), rerank_k=2)
        self.assertEqual(out["nct_id"].tolist(), ["SYNTH002", "SYNTH001"])
        self.assertEqual(out["rank"].tolist(), [1, 2])
        self.assertEqual(out["base_rank"].tolist(), [2, 1])
        self.assertEqual(out["base_score"].tolist(), [.5, .8])
        self.assertTrue(out["medcpt_score"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
