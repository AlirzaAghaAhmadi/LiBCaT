from __future__ import annotations

import re
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "the", "to", "with", "without",
    "patient", "patients", "study", "trial", "diagnosis", "previous", "prior",
    "yes", "no", "unknown", "positive", "negative",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_for_phrases(text: str) -> str:
    s = clean_text(text).lower()
    replacements = {
        "her-2": "her2",
        "stage 4": "stage iv",
        "stage 3": "stage iii",
        "stage 2": "stage ii",
        "stage 1": "stage i",
        "non-metastatic": "nonmetastatic",
        "non metastatic": "nonmetastatic",
        "triple-negative": "triple negative",
        "her2-positive": "her2 positive",
        "her2-negative": "her2 negative",
        "her2-low": "her2 low",
        "er-positive": "er positive",
        "er-negative": "er negative",
        "pr-positive": "pr positive",
        "pr-negative": "pr negative",
        "hormone-receptor": "hormone receptor",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def tokenize(text: str) -> list[str]:
    """Tokenize and add deterministic breast-cancer retrieval concepts.

    This is retrieval-time feature extraction only. It does not determine trial
    eligibility and never removes a trial.
    """
    normalised = _normalise_for_phrases(text)
    base = [m.group(0).lower() for m in TOKEN_RE.finditer(normalised)]
    out = list(base)

    # Receptor / biomarker concepts.
    marker_values = {"positive", "negative", "low", "high", "unknown"}
    for i in range(len(base) - 1):
        a, b = base[i], base[i + 1]
        if a in {"her2", "er", "pr"} and b in marker_values:
            out.append(f"{a}_{b}")
        if a == "triple" and b == "negative":
            out.append("triple_negative")

    # HR-positive / HR-negative phrasing.
    for i in range(len(base)):
        if base[i] == "hr" and i + 1 < len(base) and base[i + 1] in {"positive", "negative"}:
            out.append(f"hr_{base[i + 1]}")
        if base[i] == "hormone" and i + 1 < len(base) and base[i + 1] == "receptor":
            window = base[i : min(i + 6, len(base))]
            if "positive" in window:
                out.append("hr_positive")
            if "negative" in window:
                out.append("hr_negative")

    # IMPORTANT: exact stage matching. Word boundaries prevent "stage i" from
    # being falsely detected inside "stage iv".
    stage_match = re.search(r"\bstage\s+(iv|iii|ii|i)\b", normalised)
    if stage_match:
        out.append(f"stage_{stage_match.group(1)}")

    # Disease extent / setting concepts.
    if "nonmetastatic" in base:
        out.append("disease_nonmetastatic")
    elif any(t in base for t in ("metastatic", "metastasis", "metastases")):
        out.append("disease_metastatic")

    if re.search(r"\bearly\s+stage\b", normalised) or "localized" in base or "localised" in base:
        out.append("disease_early")
    if re.search(r"\blocally\s+advanced\b", normalised):
        out.append("disease_locally_advanced")
    if "advanced" in base:
        out.append("disease_advanced")
    if "recurrent" in base or "recurrence" in base:
        out.append("disease_recurrent")

    if "adjuvant" in base:
        out.append("setting_adjuvant")
    if "neoadjuvant" in base:
        out.append("setting_neoadjuvant")

    # Previous-treatment concepts. Presence is useful for retrieval; absence is
    # intentionally not converted into a negative query term.
    if "chemotherapy" in base or "chemo" in base:
        out.append("therapy_chemotherapy")
    if "radiotherapy" in base or "radiation" in base:
        out.append("therapy_radiotherapy")
    if "mastectomy" in base or "lumpectomy" in base or "surgery" in base:
        out.append("therapy_surgery")
    if "endocrine" in base:
        out.append("therapy_endocrine")
    if "immunotherapy" in base:
        out.append("therapy_immunotherapy")

    # Performance status.
    for i, token in enumerate(base[:-1]):
        if token == "ecog" and base[i + 1] in {"0", "1", "2", "3", "4"}:
            out.append(f"ecog_{base[i + 1]}")

    return out


def meaningful_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS]


def join_nonempty(parts: Iterable[Any], sep: str = ". ") -> str:
    vals = [clean_text(p) for p in parts]
    return sep.join(v for v in vals if v)


def _is_unknown(value: str) -> bool:
    return value.lower().strip() in {"", "unknown", "not provided", "n/a", "none"}


def _yes_no_field(value: Any, positive_phrase: str) -> str:
    v = clean_text(value)
    if _is_unknown(v):
        return ""
    low = v.lower()
    if low in {"yes", "y", "true", "1"}:
        return positive_phrase
    if low in {"no", "n", "false", "0"}:
        return ""
    return v


def build_patient_query(patient: dict[str, Any]) -> str:
    """Build a deterministic retrieval query from the structured patient form."""
    parts: list[str] = []

    diagnosis = clean_text(patient.get("diagnosis"))
    if not _is_unknown(diagnosis):
        parts.append(diagnosis)

    stage = clean_text(patient.get("stage"))
    if not _is_unknown(stage):
        parts.append(stage)

    her2 = clean_text(patient.get("her2"))
    if not _is_unknown(her2):
        parts.append(f"HER2 {her2}")

    er = clean_text(patient.get("er"))
    if not _is_unknown(er):
        parts.append(f"ER {er}")

    pr = clean_text(patient.get("pr"))
    if not _is_unknown(pr):
        parts.append(f"PR {pr}")

    chemo = _yes_no_field(patient.get("previous_chemotherapy"), "prior chemotherapy")
    if chemo:
        parts.append(chemo)

    radio = _yes_no_field(patient.get("previous_radiotherapy"), "prior radiotherapy")
    if radio:
        parts.append(radio)

    surgery = _yes_no_field(patient.get("prior_surgery"), "prior breast surgery")
    if surgery:
        parts.append(surgery)

    setting = clean_text(patient.get("treatment_setting"))
    if not _is_unknown(setting):
        parts.append(setting)

    performance = clean_text(patient.get("performance_status"))
    if not _is_unknown(performance):
        parts.append(performance)

    other = clean_text(patient.get("other_details"))
    if not _is_unknown(other):
        parts.append(other)

    return ". ".join(parts)


def matching_terms(query: str, document: str, limit: int = 10) -> list[str]:
    """Interpretable overlaps, prioritising clinical concept tokens."""
    q = meaningful_tokens(query)
    d = set(meaningful_tokens(document))
    seen: set[str] = set()
    hits: list[str] = []

    ordered = sorted(q, key=lambda t: (0 if "_" in t else 1, q.index(t)))
    for token in ordered:
        if token in d and token not in seen:
            hits.append(token)
            seen.add(token)
        if len(hits) >= limit:
            break
    return hits