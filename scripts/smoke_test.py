from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libcat.retrieval import TrialRetriever
from libcat.text_utils import build_patient_query

patient = {
    "diagnosis": "breast cancer",
    "stage": "Stage IV metastatic",
    "her2": "positive",
    "previous_chemotherapy": "previous chemotherapy",
    "performance_status": "ECOG 1",
}
query = build_patient_query(patient)
df = pd.read_csv(ROOT / "data" / "sample" / "sample_trials.csv").fillna("")
r = TrialRetriever(df, mode="lexical").search(query, top_k=5)
print("QUERY:", query)
print(r[["rank", "nct_id", "title", "score", "matching_terms"]].to_string(index=False))
