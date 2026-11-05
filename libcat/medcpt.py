from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_medcpt_document(row: pd.Series) -> str:
    """Create a concise biomedical trial representation for MedCPT reranking."""
    title = _clean(row.get("title", ""))
    official = _clean(row.get("official_title", ""))
    conditions = _clean(row.get("conditions", ""))
    interventions = _clean(row.get("interventions", ""))
    summary = _clean(row.get("brief_summary", ""))
    eligibility = _clean(row.get("eligibility", ""))

    parts = [
        title,
        official if official and official.lower() != title.lower() else "",
        f"Conditions: {conditions}" if conditions else "",
        f"Interventions: {interventions}" if interventions else "",
        f"Summary: {summary}" if summary else "",
        f"Eligibility: {eligibility}" if eligibility else "",
    ]
    return " ".join(part for part in parts if part)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class MedCPTReranker:
    """Local NCBI MedCPT biomedical cross-encoder. No API or LLM call."""

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "MedCPT requires torch and transformers. Install requirements-dense.txt first."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def device_name(self) -> str:
        return str(self.device)

    def scores(self, query: str, documents: Sequence[str], batch_size: int = 8) -> np.ndarray:
        docs = list(documents)
        if not docs:
            return np.array([], dtype=float)

        values: list[np.ndarray] = []
        torch = self.torch
        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start : start + batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                elif logits.ndim == 2:
                    logits = logits[:, -1]
                values.append(logits.detach().cpu().numpy().reshape(-1))

        return np.concatenate(values) if values else np.array([], dtype=float)


def rerank_results(
    query: str,
    results: pd.DataFrame,
    reranker: MedCPTReranker,
    rerank_k: int = 10,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Rerank only the first ``rerank_k`` hybrid candidates with MedCPT.

    The stable v4 retrieval output is preserved in ``base_rank`` and ``base_score``.
    The displayed ``score`` becomes a normalized MedCPT score for the reranked block.
    """
    if results.empty:
        return results.copy()

    n = min(max(int(rerank_k), 1), len(results))
    shortlist = results.iloc[:n].copy().reset_index(drop=True)
    remainder = results.iloc[n:].copy().reset_index(drop=True)

    shortlist["base_rank"] = shortlist.get("rank", pd.Series(range(1, n + 1))).to_numpy()
    shortlist["base_score"] = pd.to_numeric(shortlist.get("score"), errors="coerce")

    documents = [build_medcpt_document(row) for _, row in shortlist.iterrows()]
    raw = reranker.scores(query, documents, batch_size=batch_size)
    normalized = _minmax(raw)

    shortlist["medcpt_raw_score"] = raw
    shortlist["medcpt_score"] = normalized
    order = np.argsort(-normalized)
    shortlist = shortlist.iloc[order].reset_index(drop=True)
    shortlist["rank"] = np.arange(1, len(shortlist) + 1)
    shortlist["score"] = shortlist["medcpt_score"]
    shortlist["ranking_stage"] = "MedCPT biomedical rerank"

    if remainder.empty:
        return shortlist

    # The app currently displays at most 10 results, but keep the remainder valid
    # for callers that request a larger frame.
    remainder["base_rank"] = remainder.get("rank", pd.Series(dtype=float))
    remainder["base_score"] = pd.to_numeric(remainder.get("score"), errors="coerce")
    remainder["medcpt_raw_score"] = np.nan
    remainder["medcpt_score"] = np.nan
    remainder["ranking_stage"] = "Hybrid only"
    remainder["rank"] = np.arange(len(shortlist) + 1, len(shortlist) + len(remainder) + 1)

    return pd.concat([shortlist, remainder], ignore_index=True, sort=False)
