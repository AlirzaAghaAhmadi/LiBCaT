from __future__ import annotations

import hashlib
import os
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .text_utils import tokenize, matching_terms, join_nonempty


GENERIC_TOKENS = {
    "breast", "cancer", "treatment", "therapy", "disease", "study", "trial",
    "positive", "negative", "yes", "no", "unknown", "stage",
}


def _token_weight(term: str) -> float:
    if term.startswith(("her2_", "er_", "pr_", "hr_")):
        return 5.0 if term.startswith("her2_") else 3.0
    if term == "triple_negative":
        return 6.0
    if term.startswith("stage_") or term.startswith("disease_"):
        return 4.5
    if term.startswith("setting_"):
        return 3.0
    if term.startswith("therapy_"):
        return 1.6
    if term.startswith("ecog_"):
        return 1.0
    if term in GENERIC_TOKENS:
        return 0.10
    return 1.0


class BM25Retriever:
    """Small dependency-free BM25 implementation with clinical concept weights."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.documents = [d or "" for d in documents]
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(d) for d in self.documents]
        self.doc_len = np.array([len(t) for t in self.tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.term_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for tokens in self.tokens:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        n = max(len(self.documents), 1)
        self.idf = {
            t: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.array([], dtype=float)

        q_tokens = list(dict.fromkeys(tokenize(query)))
        scores = np.zeros(len(self.documents), dtype=float)
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_len[i]
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                s += (
                    _token_weight(term)
                    * self.idf.get(term, 0.0)
                    * (freq * (self.k1 + 1))
                    / (freq + norm)
                )
            scores[i] = s
        return scores


class DenseRetriever:
    def __init__(self, documents: Sequence[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval is not installed. Run: pip install -r requirements-dense.txt"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents = [d or "" for d in documents]

        # Cache document embeddings so repeated TREC runs do not spend ~5 minutes
        # encoding the same 12k trials again. The cache key changes automatically
        # if either the model or candidate-table text changes.
        digest = hashlib.sha256(model_name.encode("utf-8"))
        for doc in self.documents:
            digest.update(doc.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        cache_dir = Path(os.environ.get("LIBCAT_CACHE_DIR", Path.home() / ".cache" / "libcat"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"dense_{digest.hexdigest()[:20]}.npy"

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if cached.ndim == 2 and cached.shape[0] == len(self.documents):
                self.embeddings = cached
                print(f"Loaded cached dense embeddings: {cache_path}")
            else:
                cache_path.unlink()  # regenerate incomplete or incompatible cache
                self.embeddings = self._encode_and_cache(cache_path)
        else:
            self.embeddings = self._encode_and_cache(cache_path)

    def _encode_and_cache(self, cache_path: Path) -> np.ndarray:
        embeddings = np.asarray(self.model.encode(
            self.documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        ), dtype=np.float32)
        np.save(cache_path, embeddings)
        print(f"Saved dense embedding cache: {cache_path}")
        return embeddings

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(self.embeddings @ q, dtype=float)


class LocalCrossEncoderReranker:
    """Small local reranker. This is NOT an LLM/API call."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Neural reranking requires sentence-transformers. Run: pip install -r requirements-dense.txt"
            ) from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    def scores(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.array([], dtype=float)
        pairs = [(query, d) for d in documents]
        vals = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vals, dtype=float).reshape(-1)


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _rank_score(values: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal-rank score, robust when lexical/dense score scales differ."""
    if values.size == 0:
        return values
    order = np.argsort(-values)
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return 1.0 / (k + ranks.astype(float))


def _concept_values(tokens: set[str], prefix: str) -> set[str]:
    return {t[len(prefix):] for t in tokens if t.startswith(prefix)}


def _normalise_negated_clinical_phrases(text: str) -> str:
    """Normalize a few high-impact negated phrases before concept extraction.

    This is intentionally narrow. It prevents phrases such as "no evidence of
    metastasis" from being interpreted as positive metastatic evidence. It is
    retrieval-time feature extraction only; it does not remove a trial.
    """
    s = text or ""
    replacements = [
        (r"(?i)\bno\s+(?:clinical\s+|radiologic(?:al)?\s+|radiographic\s+)?evidence\s+of\s+(?:distant\s+)?metasta(?:sis|ses|tic\s+disease)\b", " nonmetastatic "),
        (r"(?i)\bwithout\s+(?:evidence\s+of\s+)?(?:distant\s+)?metasta(?:sis|ses|tic\s+disease)\b", " nonmetastatic "),
        (r"(?i)\babsence\s+of\s+(?:distant\s+)?metasta(?:sis|ses|tic\s+disease)\b", " nonmetastatic "),
        (r"(?i)\bnot\s+metastatic\b", " nonmetastatic "),
        (r"(?i)\bM0\b", " nonmetastatic "),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)
    return s


def _extract_patient_ecog(text: str) -> int | None:
    """Extract an explicit patient ECOG value from the query, if present."""
    m = re.search(r"(?i)\bECOG(?:\s+performance\s+status)?\s*[:=]?\s*([0-4])\b", text or "")
    return int(m.group(1)) if m else None


def _extract_allowed_ecog(text: str) -> set[int]:
    """Best-effort extraction of ECOG values allowed by inclusion text."""
    raw = text or ""
    allowed: set[int] = set()

    # ECOG 0-1 / 0 ~ 2 / 0 to 1
    for m in re.finditer(
        r"(?i)\bECOG(?:\s+performance\s+status)?[^.;\n]{0,35}?([0-4])\s*(?:-|–|—|~|to|through)\s*([0-4])\b",
        raw,
    ):
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = sorted((a, b))
        allowed.update(range(lo, hi + 1))

    # ECOG <= 2 / ≤2 / less than or equal to 2
    for m in re.finditer(
        r"(?i)\bECOG(?:\s+performance\s+status)?[^.;\n]{0,30}?(?:<=|≤|less\s+than\s+or\s+equal\s+to)\s*([0-4])\b",
        raw,
    ):
        hi = int(m.group(1))
        allowed.update(range(0, hi + 1))

    # ECOG < 3
    for m in re.finditer(
        r"(?i)\bECOG(?:\s+performance\s+status)?[^.;\n]{0,30}?(?:<|less\s+than)\s*([1-4])\b",
        raw,
    ):
        hi = int(m.group(1)) - 1
        allowed.update(range(0, hi + 1))

    # ECOG 0 or 1 / 0 and 1
    for m in re.finditer(
        r"(?i)\bECOG(?:\s+performance\s+status)?[^.;\n]{0,25}?([0-4])\s*(?:or|and|/)\s*([0-4])\b",
        raw,
    ):
        allowed.update({int(m.group(1)), int(m.group(2))})

    # If there is one simple explicit value and no range was found.
    if not allowed:
        m = re.search(
            r"(?i)\bECOG(?:\s+performance\s+status)?\s*[:=]?\s*([0-4])\b",
            raw,
        )
        if m:
            allowed.add(int(m.group(1)))

    return allowed


def _ecog_compatibility(query: str, inclusion: str) -> tuple[float, float]:
    """Return (support, conflict) for explicit ECOG requirements.

    No requirement => no evidence either way. This avoids guessing when a trial
    does not state performance status in the parsed inclusion block.
    """
    patient = _extract_patient_ecog(query)
    if patient is None:
        return 0.0, 0.0
    allowed = _extract_allowed_ecog(inclusion)
    if not allowed:
        return 0.0, 0.0
    if patient in allowed:
        return 1.0, 0.0
    return 0.0, 1.0


def _profile(text: str) -> dict[str, set[str]]:
    text = _normalise_negated_clinical_phrases(text)
    toks = set(tokenize(text))
    stages = _concept_values(toks, "stage_")
    disease = _concept_values(toks, "disease_")
    settings = _concept_values(toks, "setting_")

    if "iv" in stages:
        disease.add("metastatic")
    if stages & {"i", "ii", "iii"}:
        disease.add("early")
    if settings & {"adjuvant", "neoadjuvant"}:
        disease.add("early")

    return {
        "her2": _concept_values(toks, "her2_"),
        "er": _concept_values(toks, "er_"),
        "pr": _concept_values(toks, "pr_"),
        "hr": _concept_values(toks, "hr_"),
        "stage": stages,
        "disease": disease,
        "setting": settings,
        "therapy": _concept_values(toks, "therapy_"),
        "ecog": _concept_values(toks, "ecog_"),
        "special": {"triple_negative"} if "triple_negative" in toks else set(),
    }


def _status_evidence(
    query_vals: set[str], doc_vals: set[str], weight: float, marker: str
) -> tuple[float, float]:
    if not query_vals or not doc_vals:
        return 0.0, 0.0
    if query_vals & doc_vals:
        return weight, 0.0

    if marker == "her2":
        if "positive" in query_vals and "low" in doc_vals:
            return 0.0, weight * 0.60
        if "low" in query_vals and "positive" in doc_vals:
            return 0.0, weight * 0.45

    if ({"positive", "negative"} <= (query_vals | doc_vals)) or marker in {"er", "pr", "her2"}:
        return 0.0, weight
    return 0.0, 0.0


def _stage_evidence(
    q: dict[str, set[str]], d: dict[str, set[str]], weight: float = 5.0
) -> tuple[float, float]:
    q_met = "metastatic" in q["disease"] or "iv" in q["stage"]
    d_met = "metastatic" in d["disease"] or "iv" in d["stage"]
    q_early = bool(q["disease"] & {"early", "nonmetastatic"}) or bool(q["stage"] & {"i", "ii", "iii"})
    d_early = bool(d["disease"] & {"early", "nonmetastatic"}) or bool(d["stage"] & {"i", "ii", "iii"})

    if q_met:
        if d_met:
            return weight, 0.0
        if d_early:
            return 0.0, weight
    if q_early:
        if d_early:
            return weight, 0.0
        if d_met:
            return 0.0, weight
    return 0.0, 0.0


def _medical_evidence(
    query_profile: dict[str, set[str]], doc_profile: dict[str, set[str]]
) -> tuple[float, float]:
    positive = 0.0
    conflict = 0.0
    possible = 0.0

    if query_profile["stage"] or query_profile["disease"]:
        possible += 5.0
        p, c = _stage_evidence(query_profile, doc_profile, 5.0)
        positive += p
        conflict += c

    for marker, weight in (("her2", 5.0), ("er", 2.5), ("pr", 2.0)):
        qv = query_profile[marker]
        if not qv:
            continue
        possible += weight
        p, c = _status_evidence(qv, doc_profile[marker], weight, marker)
        positive += p
        conflict += c

    if query_profile["special"]:
        possible += 5.5
        if query_profile["special"] & doc_profile["special"]:
            positive += 5.5
        elif any(doc_profile[m] & {"positive"} for m in ("her2", "er", "pr")):
            conflict += 4.0

    if query_profile["setting"]:
        possible += 1.5
        if query_profile["setting"] & doc_profile["setting"]:
            positive += 1.5

    if query_profile["therapy"]:
        possible += 1.0
        if query_profile["therapy"] & doc_profile["therapy"]:
            positive += 1.0

    if query_profile["ecog"]:
        possible += 0.5
        if query_profile["ecog"] & doc_profile["ecog"]:
            positive += 0.5

    if possible <= 0:
        return 0.0, 0.0
    return min(1.0, positive / possible), min(1.0, conflict / possible)


def _split_eligibility(text: str) -> tuple[str, str]:
    """Best-effort split of ClinicalTrials.gov eligibility into inclusion/exclusion."""
    raw = text or ""
    if not raw.strip():
        return "", ""

    inc = re.search(r"(?i)\binclusion\s+criteria\s*:?", raw)
    exc = re.search(r"(?i)\bexclusion\s+criteria\s*:?", raw)

    if inc and exc:
        if inc.start() < exc.start():
            return raw[inc.end():exc.start()].strip(), raw[exc.end():].strip()
        return raw[inc.end():].strip(), raw[exc.end():inc.start()].strip()
    if inc:
        return raw[inc.end():].strip(), ""
    if exc:
        return raw[:exc.start()].strip(), raw[exc.end():].strip()
    return raw, ""


class TrialRetriever:
    """Clinical-trial retriever with lexical, dense and hybrid neural reranking.

    lexical: field-aware weighted BM25 + transparent clinical concept reranking.
    dense: local sentence embedding retrieval + concept reranking.
    hybrid: lexical+dense rank fusion, then a small local CrossEncoder reranks
            the strongest candidates. No LLM and no external API is used.
    auto: hybrid when sentence-transformers is available, otherwise lexical.
    """

    def __init__(
        self,
        trials: pd.DataFrame,
        text_column: str = "retrieval_text",
        mode: str = "auto",
        dense_weight: float = 0.35,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates: int = 120,
    ):
        if mode not in {"lexical", "dense", "hybrid", "auto"}:
            raise ValueError("mode must be one of: lexical, dense, hybrid, auto")
        if trials.empty:
            raise ValueError("No trials supplied: the candidate table is empty")
        if text_column not in trials.columns:
            raise ValueError(f"Missing required text column: {text_column}")

        self.trials = trials.reset_index(drop=True).copy()
        self.text_column = text_column
        self.documents = self.trials[text_column].fillna("").astype(str).tolist()

        def col(name: str) -> list[str]:
            if name not in self.trials.columns:
                return [""] * len(self.trials)
            return self.trials[name].fillna("").astype(str).tolist()

        titles = col("title")
        official_titles = col("official_title")
        conditions = col("conditions")
        interventions = col("interventions")
        summaries = col("brief_summary")
        eligibility = col("eligibility")

        inclusions: list[str] = []
        exclusions: list[str] = []
        for text in eligibility:
            inc, exc = _split_eligibility(text)
            inclusions.append(inc)
            exclusions.append(exc)

        # Retrieval focus: title/condition/summary say what the study is about.
        self.focus_documents = [
            join_nonempty([titles[i], official_titles[i], conditions[i], interventions[i], summaries[i]])
            for i in range(len(self.trials))
        ]
        self.title_condition_documents = [
            join_nonempty([titles[i], official_titles[i], conditions[i], interventions[i]])
            for i in range(len(self.trials))
        ]
        self.summary_documents = summaries
        self.inclusion_documents = inclusions
        self.exclusion_documents = exclusions

        # Neural reranker sees high-confidence study description plus inclusion
        # criteria, not the exclusion block mixed into a single positive passage.
        self.rerank_documents = [
            join_nonempty([
                self.focus_documents[i],
                f"Inclusion criteria: {inclusions[i]}" if inclusions[i] else "",
            ])
            for i in range(len(self.trials))
        ]

        self.bm25_title_condition = BM25Retriever(self.title_condition_documents)
        self.bm25_summary = BM25Retriever(self.summary_documents)
        self.bm25_inclusion = BM25Retriever(self.inclusion_documents)
        self.bm25_full = BM25Retriever(self.documents)

        # Concept evidence can safely use focus + inclusion. Exclusion statements
        # are deliberately excluded from positive evidence.
        self.document_profiles = [
            _profile(join_nonempty([self.focus_documents[i], self.inclusion_documents[i]]))
            for i in range(len(self.trials))
        ]

        self.dense = None
        self.reranker = None
        self.mode_requested = mode
        self.mode_active = mode
        self.dense_weight = float(dense_weight)
        self.rerank_candidates = max(20, int(rerank_candidates))

        if mode in {"dense", "hybrid", "auto"}:
            try:
                # Dense model also sees inclusion criteria because TREC relevance
                # depends heavily on whether the patient fits inclusion rules.
                dense_docs = self.rerank_documents
                self.dense = DenseRetriever(dense_docs, model_name=model_name)
                if mode == "auto":
                    self.mode_active = "hybrid"
            except Exception:
                if mode == "auto":
                    self.mode_active = "lexical"
                else:
                    raise

        if self.mode_active == "hybrid":
            try:
                self.reranker = LocalCrossEncoderReranker(reranker_model_name)
            except Exception as exc:
                # Keep hybrid usable even if the reranker model cannot be loaded.
                print(f"WARNING: local cross-encoder reranker unavailable; using hybrid fusion only: {exc}")
                self.reranker = None

        if mode == "lexical":
            self.mode_active = "lexical"

    def _lexical_scores(self, query: str) -> np.ndarray:
        title_condition = _minmax(self.bm25_title_condition.scores(query))
        summary = _minmax(self.bm25_summary.scores(query))
        inclusion = _minmax(self.bm25_inclusion.scores(query))
        full = _minmax(self.bm25_full.scores(query))

        # Inclusion is useful, but title/condition still dominate to prevent an
        # incidental eligibility mention from defining trial relevance.
        return (
            0.43 * title_condition
            + 0.24 * summary
            + 0.20 * inclusion
            + 0.13 * full
        )

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        if not query.strip():
            raise ValueError("Query is empty.")

        lexical = self._lexical_scores(query)
        dense = None

        q_profile = _profile(query)
        coverage = np.zeros(len(self.documents), dtype=float)
        conflicts = np.zeros(len(self.documents), dtype=float)
        ecog_support = np.zeros(len(self.documents), dtype=float)
        ecog_conflict = np.zeros(len(self.documents), dtype=float)
        for i, d_profile in enumerate(self.document_profiles):
            coverage[i], conflicts[i] = _medical_evidence(q_profile, d_profile)
            ecog_support[i], ecog_conflict[i] = _ecog_compatibility(
                query, self.inclusion_documents[i]
            )

        # Performance status is a high-value eligibility signal in TREC. Keep it
        # separate from generic concept coverage so an explicit ECOG mismatch
        # cannot be washed out by many disease/biomarker keyword matches.
        compatibility = np.clip(0.82 * coverage + 0.18 * ecog_support, 0.0, 1.0)
        contradiction = np.clip(0.72 * conflicts + 0.55 * ecog_conflict, 0.0, 1.0)

        if self.mode_active == "dense":
            assert self.dense is not None
            dense = _minmax(self.dense.scores(query))
            base = dense
            first_stage_raw = 0.58 * base + 0.42 * compatibility - 0.82 * contradiction

        elif self.mode_active == "hybrid":
            assert self.dense is not None
            dense = _minmax(self.dense.scores(query))

            # RRF adds complementary rank evidence without assuming BM25 and
            # cosine scores have comparable distributions.
            rrf = _minmax(_rank_score(lexical) + _rank_score(dense))
            score_fusion = (1.0 - self.dense_weight) * lexical + self.dense_weight * dense
            first_stage_raw = (
                0.50 * score_fusion
                + 0.18 * rrf
                + 0.32 * compatibility
                - 0.82 * contradiction
            )

        else:
            base = lexical
            first_stage_raw = 0.54 * base + 0.46 * compatibility - 0.88 * contradiction

        first_stage = _minmax(first_stage_raw)
        final_raw = first_stage.copy()
        cross_scores = np.full(len(self.documents), np.nan, dtype=float)

        # Two-stage local reranking only for hybrid mode. This keeps the expensive
        # model focused on the most promising candidates and improves top-10 quality.
        if self.mode_active == "hybrid" and self.reranker is not None:
            n = min(self.rerank_candidates, len(first_stage))
            cand_idx = np.argsort(-first_stage)[:n]
            docs = [self.rerank_documents[i] for i in cand_idx]
            ce = _minmax(self.reranker.scores(query, docs))
            cross_scores[cand_idx] = ce

            reranked = (
                0.52 * first_stage[cand_idx]
                + 0.34 * ce
                + 0.22 * compatibility[cand_idx]
                - 0.72 * contradiction[cand_idx]
            )
            # Ensure reranked candidates stay above non-shortlisted documents.
            reranked = _minmax(reranked)
            final_raw = np.zeros(len(first_stage), dtype=float)
            final_raw[:] = 0.15 * first_stage
            final_raw[cand_idx] = 0.20 + 0.80 * reranked

        final = _minmax(final_raw)
        k = max(1, min(int(top_k), len(final)))
        order = np.argsort(-final)[:k]

        out = self.trials.iloc[order].copy()
        out["score"] = final[order]
        out["lexical_score"] = lexical[order]
        out["dense_score"] = dense[order] if dense is not None else np.nan
        out["cross_encoder_score"] = cross_scores[order]
        out["concept_coverage"] = coverage[order]
        out["concept_conflict"] = conflicts[order]
        out["ecog_support"] = ecog_support[order]
        out["ecog_conflict"] = ecog_conflict[order]
        out["clinical_compatibility"] = compatibility[order]
        out["clinical_contradiction"] = contradiction[order]
        out["matching_terms"] = [
            ", ".join(matching_terms(query, self.rerank_documents[i])) for i in order
        ]
        out["rank"] = np.arange(1, len(out) + 1)
        return out
