"""LiBCaT public Python API. Research retrieval, not clinical eligibility screening."""
from .retrieval import TrialRetriever, BM25Retriever
from .text_utils import build_patient_query

__version__ = "0.1.0"
__all__ = ["TrialRetriever", "BM25Retriever", "build_patient_query"]
