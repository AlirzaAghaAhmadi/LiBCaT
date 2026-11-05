"""Use LiBCaT with your own trial-candidate table (no model download)."""
import pandas as pd
from libcat import TrialRetriever, build_patient_query

# For real use, supply one row per study with a unique identifier and retrieval_text.
trials = pd.DataFrame([
    {"nct_id": "DEMO-A", "title": "HER2 positive metastatic breast cancer",
     "conditions": "Metastatic breast cancer", "brief_summary": "HER2 positive treatment",
     "eligibility": "Inclusion Criteria: ECOG 0-2", "retrieval_text": "HER2 positive metastatic breast cancer ECOG 0-2"},
    {"nct_id": "DEMO-B", "title": "Early stage breast cancer",
     "conditions": "Localized breast cancer", "brief_summary": "Adjuvant treatment",
     "eligibility": "Inclusion Criteria: stage I", "retrieval_text": "early stage I localized breast cancer"},
])
query = build_patient_query({"diagnosis": "breast cancer", "stage": "Stage IV", "her2": "positive"})
ranked = TrialRetriever(trials, mode="lexical").search(query, top_k=2)
print(ranked[["rank", "nct_id", "title", "score", "matching_terms"]].to_string(index=False))
