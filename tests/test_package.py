import unittest
import pandas as pd
from libcat import TrialRetriever, build_patient_query
from libcat.medcpt import rerank_results

class PackageTests(unittest.TestCase):
    def test_empty_candidate_table_fails_helpfully(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            TrialRetriever(pd.DataFrame({'retrieval_text': []}), mode='lexical')

    def test_invalid_mode_fails_helpfully(self):
        with self.assertRaisesRegex(ValueError, 'mode'):
            TrialRetriever(pd.DataFrame({'retrieval_text': ['x']}), mode='banana')

    def test_simple_api(self):
        docs = pd.DataFrame([
            {'nct_id':'1','title':'HER2 positive breast cancer', 'retrieval_text':'HER2 positive breast cancer'},
            {'nct_id':'2','title':'HER2 negative breast cancer', 'retrieval_text':'HER2 negative breast cancer'},
        ])
        q = build_patient_query({'diagnosis':'breast cancer','her2':'positive'})
        r = TrialRetriever(docs,mode='lexical').search(q, top_k=2)
        self.assertEqual(list(r['nct_id'])[0], '1')
        self.assertEqual(list(r['rank']), [1,2])

    def test_no_medcpt_results(self):
        self.assertTrue(rerank_results('test',pd.DataFrame(),object()).empty)
