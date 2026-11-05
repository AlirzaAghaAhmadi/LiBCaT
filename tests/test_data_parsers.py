import unittest

from libcat.clinicaltrials import study_to_row
from scripts.parse_trec_corpus import parse_study, looks_like_breast


class InputParsersTests(unittest.TestCase):
    def test_clinicaltrials_api_row(self):
        raw = {"protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Breast cancer research"},
            "statusModule": {"overallStatus": "RECRUITING", "lastUpdatePostDateStruct": {"date": "2026-01-03"}},
            "designModule": {"studyType": "INTERVENTIONAL"},
            "conditionsModule": {"conditions": ["Breast Cancer"]},
            "descriptionModule": {"briefSummary": "Breast carcinoma protocol"},
            "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria: ECOG 0-2"},
            "contactsLocationsModule": {"locations": [{"country": "United States", "city": "Austin"}]},
        }}
        row = study_to_row(raw)
        self.assertEqual(row["nct_id"], "NCT00000001")
        self.assertEqual(row["last_update"], "2026-01-03")
        self.assertIn("Breast Cancer", row["retrieval_text"])
        self.assertEqual(row["countries"], "United States")

    def test_historical_xml_parsing(self):
        raw = b'''<clinical_study>
          <id_info><nct_id>NCT00000002</nct_id></id_info>
          <brief_title>Trial for breast carcinoma</brief_title>
          <condition>Breast Cancer</condition>
          <brief_summary><textblock>HER2 positive breast disease</textblock></brief_summary>
          <eligibility><criteria><textblock>Inclusion Criteria: ECOG 0-1</textblock></criteria></eligibility>
        </clinical_study>'''
        row = parse_study(raw)
        self.assertEqual(row["nct_id"], "NCT00000002")
        self.assertTrue(looks_like_breast(row))
        self.assertIn("ECOG 0-1", row["eligibility"])
        self.assertEqual(parse_study(b"<broken"), None)

    def test_historical_candidate_filter(self):
        self.assertFalse(looks_like_breast({"conditions": "Lung cancer", "title": "Lung study"}))
