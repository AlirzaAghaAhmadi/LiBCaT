import math
import unittest
from libcat.trec import (
    dcg, ndcg_at_k, eligible_hit_at_k, eligible_precision_at_k,
    eligible_recall_at_k, eligible_mrr, qrel_coverage,
)

class TrecMetricTests(unittest.TestCase):
    def test_graded_qrels(self):
        grade = {"A": 2, "B": 1, "C": 0}
        self.assertAlmostEqual(ndcg_at_k(["A", "B", "C"], grade, 3), 1.0)
        self.assertLess(ndcg_at_k(["B", "A", "C"], grade, 3), 1.0)
        self.assertEqual(eligible_hit_at_k(["B", "C", "A"], grade, 2), 0)
        self.assertEqual(eligible_hit_at_k(["B", "C", "A"], grade, 3), 1)
        self.assertAlmostEqual(eligible_precision_at_k(["A", "B"], grade, 2), .5)
        self.assertAlmostEqual(eligible_recall_at_k(["B", "A"], grade, 2), 1.0)
        self.assertAlmostEqual(eligible_mrr(["C", "B", "A"], grade), 1/3)
        self.assertAlmostEqual(qrel_coverage({"A", "B"}, grade), 1.0)

    def test_official_trec_linear_gain_not_exponential_gain(self):
        # TREC CT 2023 assigns *gain 2*, not 2**2-1=3, to eligible.
        self.assertAlmostEqual(dcg([2, 1]), 2 + 1 / math.log2(3))
        expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(["B", "A"], {"A": 2, "B": 1}, 2), expected)

    def test_no_eligible_judgments(self):
        grade = {"B": 1, "C": 0}
        self.assertEqual(eligible_recall_at_k(["B"], grade), 0)
        self.assertEqual(qrel_coverage({"B"}, grade), 0)
        self.assertEqual(ndcg_at_k([], {}, 10), 0)
