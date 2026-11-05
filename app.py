from __future__ import annotations

from html import escape
from pathlib import Path
from time import perf_counter
import ast
import re

import pandas as pd
import streamlit as st

from libcat.retrieval import TrialRetriever
from libcat.text_utils import build_patient_query
from libcat.medcpt import MedCPTReranker, rerank_results


ROOT = Path(__file__).resolve().parent
REAL_DATA = ROOT / "data" / "demo_trials.csv"
SAMPLE_DATA = ROOT / "data" / "sample" / "sample_trials.csv"

st.set_page_config(
    page_title="LiBCaT | Clinical Trial Retrieval",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Visual system
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    :root {
        --bg: #070B14;
        --bg-2: #0B1020;
        --panel: rgba(15, 23, 42, 0.88);
        --panel-2: rgba(20, 29, 51, 0.92);
        --panel-hover: rgba(25, 36, 62, 0.96);
        --text: #F8FAFC;
        --muted: #94A3B8;
        --muted-2: #64748B;
        --line: rgba(148, 163, 184, 0.16);
        --purple: #8B5CF6;
        --purple-2: #A78BFA;
        --cyan: #22D3EE;
        --blue: #60A5FA;
        --rose: #FB7185;
        --green: #34D399;
        --amber: #FBBF24;
    }

    html, body, [class*="css"] {
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(139,92,246,.14), transparent 26%),
            radial-gradient(circle at 86% 14%, rgba(34,211,238,.10), transparent 23%),
            linear-gradient(180deg, #070B14 0%, #090D18 58%, #070A12 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 11, 20, 0.76);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(148,163,184,.08);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0F1C 0%, #0C1221 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: #DCE5F2;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--line);
    }

    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(23,31,56,.96) 0%, rgba(18,25,47,.96) 55%, rgba(17,28,48,.96) 100%);
        border: 1px solid rgba(139,92,246,.28);
        border-radius: 24px;
        padding: 1.65rem 1.8rem 1.45rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 55px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.04);
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -90px;
        top: -150px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139,92,246,.25), rgba(139,92,246,0) 70%);
        pointer-events: none;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        left: 38%;
        bottom: -175px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(34,211,238,.15), rgba(34,211,238,0) 70%);
        pointer-events: none;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        color: #DDD6FE;
        background: rgba(139,92,246,.15);
        border: 1px solid rgba(167,139,250,.25);
        border-radius: 999px;
        padding: .3rem .7rem;
        font-size: .73rem;
        font-weight: 800;
        letter-spacing: .07em;
        text-transform: uppercase;
        margin-bottom: .8rem;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 2.15rem;
        line-height: 1.08;
        letter-spacing: -.035em;
        margin: 0 0 .65rem 0;
        text-shadow: 0 2px 18px rgba(0,0,0,.24);
    }

    .hero p {
        color: #B8C5D8;
        max-width: 900px;
        margin: 0;
        font-size: .98rem;
        line-height: 1.68;
    }

    .scope-row {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem;
        margin-top: 1rem;
    }

    .pill {
        display: inline-block;
        border: 1px solid rgba(148,163,184,.18);
        background: rgba(255,255,255,.045);
        color: #CBD5E1;
        padding: .3rem .68rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 650;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
    }

    .section-label {
        color: #F8FAFC;
        font-size: 1.1rem;
        font-weight: 780;
        margin: .75rem 0 .15rem 0;
    }

    .section-help {
        color: var(--muted);
        font-size: .88rem;
        margin: 0 0 .75rem 0;
    }

    div[data-testid="stForm"] {
        background: linear-gradient(180deg, rgba(15,23,42,.94), rgba(12,19,35,.94));
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 20px;
        padding: 1.05rem 1.05rem .45rem 1.05rem;
        box-shadow: 0 14px 38px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.025);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(15,23,42,.94), rgba(12,19,35,.94));
        border: 1px solid rgba(148,163,184,.16) !important;
        border-radius: 19px !important;
        box-shadow: 0 14px 38px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.025);
        transition: border-color .18s ease, transform .18s ease, background .18s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(139,92,246,.34) !important;
        background: var(--panel-hover);
        transform: translateY(-1px);
    }

    label, [data-testid="stWidgetLabel"] p {
        color: #DCE5F2 !important;
        font-weight: 600 !important;
    }

    input, textarea, [data-baseweb="select"] > div {
        background-color: #0D1423 !important;
        color: #F8FAFC !important;
        border-color: rgba(148,163,184,.20) !important;
    }

    input:focus, textarea:focus {
        border-color: rgba(139,92,246,.70) !important;
        box-shadow: 0 0 0 1px rgba(139,92,246,.35) !important;
    }

    [data-baseweb="popover"] {
        color: #F8FAFC !important;
    }

    [data-baseweb="menu"] {
        background: #0D1423 !important;
        border: 1px solid rgba(148,163,184,.18) !important;
    }

    [role="option"] {
        color: #E2E8F0 !important;
    }

    [role="option"]:hover {
        background: rgba(139,92,246,.16) !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(8,14,27,.72);
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 13px;
        padding: .58rem .72rem;
    }

    div[data-testid="stMetric"] label {
        color: #8FA0B8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    .trial-rank {
        width: 2.05rem;
        height: 2.05rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        background: linear-gradient(135deg, rgba(139,92,246,.25), rgba(34,211,238,.12));
        border: 1px solid rgba(167,139,250,.28);
        color: #DDD6FE;
        font-size: .88rem;
        font-weight: 850;
        margin-right: .6rem;
        vertical-align: .15rem;
    }

    .trial-title {
        display: inline;
        color: #F8FAFC;
        font-size: 1.08rem;
        line-height: 1.35;
        font-weight: 780;
    }

    .meta-pill {
        display: inline-block;
        margin: .2rem .28rem .2rem 0;
        padding: .24rem .56rem;
        border-radius: 999px;
        background: rgba(96,165,250,.09);
        border: 1px solid rgba(96,165,250,.18);
        color: #BFDBFE;
        font-size: .73rem;
        font-weight: 680;
    }

    .signal-pill {
        display: inline-block;
        margin: .18rem .25rem .18rem 0;
        padding: .24rem .55rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(139,92,246,.14), rgba(34,211,238,.08));
        border: 1px solid rgba(167,139,250,.21);
        color: #DDD6FE;
        font-size: .73rem;
        font-weight: 680;
    }

    .notice {
        border-radius: 13px;
        padding: .72rem .84rem;
        font-size: .82rem;
        line-height: 1.5;
        margin: .5rem 0 .6rem 0;
    }

    .notice-warn {
        background: rgba(251,191,36,.08);
        border: 1px solid rgba(251,191,36,.22);
        color: #FDE68A;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(52,211,153,.20);
        background: rgba(16,185,129,.09);
        color: #D1FAE5;
    }

    [data-testid="stExpander"] {
        background: rgba(10,16,29,.70);
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 12px;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stLinkButton > a {
        border-radius: 11px !important;
        font-weight: 750 !important;
        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease !important;
    }

    .stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7C3AED 0%, #8B5CF6 47%, #0EA5E9 100%) !important;
        border: 1px solid rgba(167,139,250,.45) !important;
        color: white !important;
        box-shadow: 0 10px 30px rgba(124,58,237,.22);
    }

    .stFormSubmitButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 13px 34px rgba(124,58,237,.32);
    }

    .stLinkButton > a {
        background: rgba(139,92,246,.10) !important;
        border: 1px solid rgba(167,139,250,.22) !important;
        color: #DDD6FE !important;
    }

    .stLinkButton > a:hover {
        background: rgba(139,92,246,.17) !important;
        border-color: rgba(167,139,250,.42) !important;
    }

    .footer-note {
        color: #7F8EA7;
        font-size: .78rem;
        line-height: 1.58;
        border-top: 1px solid rgba(148,163,184,.12);
        margin-top: 1.6rem;
        padding-top: .9rem;
    }

    code {
        color: #C4B5FD !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,.12);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Keep Streamlit chrome quiet during presentation. */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data / retrieval helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_csv(path).fillna("")


@st.cache_resource(show_spinner=False)
def build_retriever(path: str, mtime_ns: int, mode: str) -> TrialRetriever:
    del mtime_ns
    frame = pd.read_csv(path).fillna("")
    return TrialRetriever(frame, mode=mode)


@st.cache_resource(show_spinner=False)
def build_medcpt_reranker() -> MedCPTReranker:
    # Loaded once per Streamlit process; subsequent searches reuse the local model.
    return MedCPTReranker()


def active_data_path() -> Path:
    return REAL_DATA if REAL_DATA.exists() else SAMPLE_DATA


def clean_label(value: object) -> str:
    text = str(value or "").replace("_", " ").strip()
    return text if text else "—"


def trial_url(row: pd.Series) -> str:
    url = str(row.get("trial_url", "") or "").strip()
    if url and url != "https://clinicaltrials.gov/":
        return url
    nct = str(row.get("nct_id", "") or "").strip()
    if nct.startswith("NCT"):
        return f"https://clinicaltrials.gov/study/{nct}"
    return url


def number(row: pd.Series, name: str) -> float | None:
    try:
        value = float(row.get(name))
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def split_values(value: object, limit: int = 8) -> list[str]:
    """Robustly normalize retrieval signal containers/strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return parts[:limit]

    text = str(value).strip()
    if not text:
        return []

    # Handle values serialized as Python lists, e.g. "['her2_positive', 'breast']".
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                parts = [str(x).strip() for x in parsed if str(x).strip()]
                return parts[:limit]
        except (ValueError, SyntaxError):
            pass

    parts = [p.strip().strip("'\"") for p in re.split(r"[;,]", text) if p.strip()]
    return parts[:limit]


SIGNAL_LABELS = {
    "her2_positive": "HER2 positive",
    "her2_negative": "HER2 negative",
    "her2_low": "HER2-low",
    "er_positive": "ER positive",
    "er_negative": "ER negative",
    "pr_positive": "PR positive",
    "pr_negative": "PR negative",
    "stage_iv": "Stage IV",
    "stage_iii": "Stage III",
    "stage_ii": "Stage II",
    "stage_i": "Stage I",
    "disease_metastatic": "Metastatic disease",
    "therapy_chemotherapy": "Prior chemotherapy",
    "therapy_radiotherapy": "Prior radiotherapy",
    "therapy_neoadjuvant": "Neoadjuvant therapy",
    "therapy_adjuvant": "Adjuvant therapy",
}

GENERIC_SIGNALS = {
    "breast", "cancer", "her2", "er", "pr", "stage", "iv", "iii", "ii", "i",
    "metastatic", "chemotherapy", "radiotherapy", "therapy", "treatment", "diagnosis",
    "positive", "negative", "ecog",
}


def display_signals(value: object, limit: int = 8) -> list[str]:
    # UI deliberately shows only interpretable clinical concepts.
    raw = split_values(value, limit=50)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        key = re.sub(r"[^a-z0-9]+", "_", item.strip().lower()).strip("_")
        label = SIGNAL_LABELS.get(key)
        if not label:
            # Do not surface raw retrieval tokens such as breast/cancer/her2.
            continue
        low = label.lower()
        if low not in seen:
            out.append(label)
            seen.add(low)
        if len(out) >= limit:
            break
    return out


def compact_scope(value: object, limit: int = 3) -> tuple[str, str]:
    full = str(value or "").strip()
    if not full:
        return "—", ""
    parts = [p.strip() for p in full.split(";") if p.strip()]
    if not parts:
        preview = full[:145] + ("…" if len(full) > 145 else "")
        return preview, full

    def score(item: str) -> int:
        t = item.lower()
        value = 0
        if "breast" in t: value += 20
        if "her2" in t: value += 8
        if "metast" in t or "stage iv" in t: value += 7
        if "estrogen" in t or "er-positive" in t or "progesterone" in t or "pr-positive" in t: value += 3
        return value

    ranked = sorted(enumerate(parts), key=lambda x: (-score(x[1]), x[0]))
    picked: list[str] = []
    seen: set[str] = set()
    for _, item in ranked:
        key = item.casefold()
        if key in seen:
            continue
        picked.append(item)
        seen.add(key)
        if len(picked) >= limit:
            break
    preview = " · ".join(picked)
    if len(parts) > len(picked):
        preview += f"  +{len(parts)-len(picked)} more"
    return preview, full


def compact_location(cities: object, countries: object, limit: int = 3) -> str:
    city_list = [x.strip() for x in re.split(r"[;,]", str(cities or "")) if x.strip()]
    country_list = [x.strip() for x in re.split(r"[;,]", str(countries or "")) if x.strip()]
    unique_cities = list(dict.fromkeys(city_list))
    unique_countries = list(dict.fromkeys(country_list))

    if unique_cities:
        examples = ", ".join(unique_cities[:limit])
        suffix = f" · +{len(unique_cities)-limit} more" if len(unique_cities) > limit else ""
        return f"{len(unique_cities)} recorded site{'s' if len(unique_cities) != 1 else ''} · {examples}{suffix}"
    if unique_countries:
        return ", ".join(unique_countries[:limit])
    return "—"


def pills(items: list[str], class_name: str = "meta-pill") -> str:
    return "".join(
        f'<span class="{class_name}">{escape(item.replace("_", " "))}</span>'
        for item in items
    )


def patient_summary(patient: dict[str, str]) -> list[str]:
    labels = []
    mapping = [
        ("stage", "Stage"),
        ("her2", "HER2"),
        ("er", "ER"),
        ("pr", "PR"),
        ("performance_status", "Performance"),
        ("treatment_setting", "Setting"),
    ]
    for key, label in mapping:
        value = str(patient.get(key, "") or "").strip()
        if value and value.lower() != "unknown":
            labels.append(f"{label}: {value}")
    return labels


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
path = active_data_path()
df = load_data(str(path), path.stat().st_mtime_ns)

st.markdown(
    """
<div class="hero">
    <div class="eyebrow">LiBCaT · Clinical Trial Retrieval Research</div>
    <h1>Breast Cancer Trial Finder</h1>
    <p>
        Retrieve and rank potentially relevant breast-cancer clinical trials from a fixed
        ClinicalTrials.gov snapshot using lexical, semantic and local reranking signals.
        This prototype supports search and research evaluation; it does not determine eligibility.
    </p>
    <div class="scope-row">
        <span class="pill">Breast cancer</span>
        <span class="pill">United States demo snapshot</span>
        <span class="pill">BM25 · Hybrid · optional MedCPT</span>
        <span class="pill">No LLM API</span>
        <span class="pill">Biomedical reranking</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if path == SAMPLE_DATA:
    st.warning(
        "Synthetic sample data are active. Build `data/demo_trials.csv` to use the fixed "
        "ClinicalTrials.gov demonstration snapshot."
    )
else:
    st.success(f"ClinicalTrials.gov demo snapshot ready · {len(df):,} candidate trials")


with st.sidebar:
    st.markdown("### Ranking profile")
    ranking_profile = st.radio(
        "Choose ranking quality",
        ["Lexical (offline)", "Hybrid (local models)", "Biomedical rerank"],
        index=0,
        help=(
            "Lexical needs no downloaded models. Hybrid adds dense retrieval and a general "
            "cross-encoder; biomedical rerank adds local NCBI MedCPT on the hybrid Top 10."
        ),
    )
    top_k = st.slider("Results to display", 5, 10, 10, 1)
    mode = "lexical" if ranking_profile == "Lexical (offline)" else "hybrid"
    if ranking_profile == "Biomedical rerank":
        st.caption("Local biomedical model · slower on CPU · requires optional dense dependencies")
    elif mode == "hybrid":
        st.caption("Local embedding and cross-encoder models · initial download required")
    else:
        st.caption("BM25 + interpretable clinical signals · no model download")

    st.divider()
    st.markdown("### Demo scope")
    st.caption("Cancer")
    st.write("Breast cancer")
    st.caption("Geography")
    st.write("United States")
    st.caption("Trial source")
    st.write("Fixed ClinicalTrials.gov snapshot")
    st.caption("External API / LLM")
    st.write("None during retrieval")

    st.divider()
    st.markdown("### Interpretation")
    st.caption(
        "Ranking scores are relative retrieval scores. They are not eligibility probabilities, "
        "clinical recommendations or treatment scores."
    )


st.markdown('<div class="section-label">Patient profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-help">Enter known facts only. Leave unavailable clinical information as Unknown.</div>',
    unsafe_allow_html=True,
)

with st.form("patient_form"):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("**Disease**")
        diagnosis = st.text_input("Diagnosis", value="breast cancer")
        stage = st.selectbox(
            "Disease stage",
            ["Unknown", "Stage I", "Stage II", "Stage III", "Stage IV / metastatic"],
            index=4,
        )
        treatment_setting = st.selectbox(
            "Treatment setting",
            ["Unknown", "metastatic", "neoadjuvant", "adjuvant"],
            index=1,
        )

    with c2:
        st.markdown("**Biomarkers**")
        her2 = st.selectbox("HER2", ["unknown", "positive", "negative", "low"], index=1)
        er = st.selectbox("ER", ["unknown", "positive", "negative"], index=1)
        pr = st.selectbox("PR", ["unknown", "positive", "negative"], index=1)

    with c3:
        st.markdown("**Previous care & status**")
        chemo = st.selectbox("Previous chemotherapy", ["unknown", "yes", "no"], index=1)
        radio = st.selectbox("Previous radiotherapy", ["unknown", "yes", "no"], index=0)
        surgery = st.selectbox("Prior breast surgery", ["unknown", "yes", "no"], index=0)

    performance = st.selectbox(
        "Performance status",
        ["Unknown", "ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3", "ECOG 4"],
        index=2,
    )
    other = st.text_area(
        "Other relevant details",
        placeholder="Optional: specific previous drugs, sites of metastasis, additional biomarkers…",
        height=85,
    )

    submitted = st.form_submit_button(
        "Search clinical trials",
        type="primary",
        use_container_width=True,
    )


if submitted:
    patient = {
        "diagnosis": diagnosis,
        "stage": stage,
        "her2": her2,
        "er": er,
        "pr": pr,
        "previous_chemotherapy": chemo,
        "previous_radiotherapy": radio,
        "prior_surgery": surgery,
        "treatment_setting": treatment_setting,
        "performance_status": performance,
        "other_details": other,
    }
    query = build_patient_query(patient)

    summary_items = patient_summary(patient)
    if summary_items:
        st.markdown(pills(summary_items), unsafe_allow_html=True)

    with st.expander("Generated retrieval query", expanded=False):
        st.code(query, language=None)

    try:
        started = perf_counter()
        rerank_device = None
        if ranking_profile == "Biomedical rerank":
            with st.spinner("Hybrid retrieval + biomedical MedCPT reranking…"):
                retriever = build_retriever(str(path), path.stat().st_mtime_ns, "hybrid")
                if retriever.reranker is None:
                    raise RuntimeError(
                        "The general cross-encoder is unavailable. Biomedical mode requires "
                        "both the hybrid reranker and MedCPT; check the model download and "
                        "optional dependencies before retrying."
                    )
                # The app reranks the hybrid Top 10 using MedCPT.
                base_results = retriever.search(query, top_k=10)
                medcpt = build_medcpt_reranker()
                rerank_device = medcpt.device_name
                results = rerank_results(query, base_results, medcpt, rerank_k=10, batch_size=8).head(top_k)
            ranking_label = "Hybrid → MedCPT Top-10"
        else:
            with st.spinner("Ranking candidate trials…"):
                retriever = build_retriever(str(path), path.stat().st_mtime_ns, mode)
                results = retriever.search(query, top_k=top_k)
            if mode == "hybrid" and retriever.reranker is None:
                st.warning(
                    "The general cross-encoder could not be loaded. Results use hybrid "
                    "retrieval without that reranking stage. Check model downloads and "
                    "optional dependencies to enable the complete pipeline."
                )
                ranking_label = "Hybrid (cross-encoder unavailable)"
            else:
                ranking_label = "Lexical BM25" if mode == "lexical" else "Hybrid + cross-encoder"

        elapsed = perf_counter() - started

        st.markdown('<div class="section-label">Ranked results</div>', unsafe_allow_html=True)
        device_note = f' · MedCPT: <b>{escape(rerank_device)}</b>' if rerank_device else ''
        st.markdown(
            f'<div class="section-help">Ranking: <b>{escape(ranking_label)}</b>{device_note} · '
            f'Searched: <b>{len(df):,}</b> candidate trials · '
            f'Showing <b>{len(results)}</b> results · '
            f'Latency: <b>{elapsed:.1f}s</b></div>',
            unsafe_allow_html=True,
        )

        if results.empty:
            st.info("No trials were returned for this query.")

        for _, row in results.iterrows():
            rank = int(row.get("rank", 0))
            title = str(row.get("title", "Untitled trial") or "Untitled trial")
            nct = str(row.get("nct_id", "") or "")
            score = number(row, "score")
            contradiction = number(row, "clinical_contradiction")
            compatibility = number(row, "clinical_compatibility")

            with st.container(border=True):
                st.markdown(
                    f'<span class="trial-rank">{rank}</span>'
                    f'<span class="trial-title">{escape(title)}</span>',
                    unsafe_allow_html=True,
                )

                meta = []
                if nct:
                    meta.append(nct)
                phase = clean_label(row.get("phase"))
                if phase != "—":
                    meta.append(phase)
                status = clean_label(row.get("status"))
                if status != "—":
                    meta.append(status)
                if meta:
                    st.markdown(pills(meta), unsafe_allow_html=True)

                m1, m2, m3 = st.columns([1, 1, 2.4])
                m1.metric("Rerank score" if ranking_profile == "Biomedical rerank" else "Retrieval score", f"{score:.3f}" if score is not None else "—")
                m2.metric("Rank", f"#{rank}")
                scope_preview, scope_full = compact_scope(row.get("conditions", ""), limit=3)
                m3.write("**Study scope**")
                m3.caption(scope_preview)

                matched = display_signals(row.get("matching_terms"), limit=8)
                if matched:
                    st.caption("Matched clinical signals")
                    st.markdown(pills(matched, "signal-pill"), unsafe_allow_html=True)

                if contradiction is not None and contradiction >= 0.25:
                    st.markdown(
                        '<div class="notice notice-warn"><b>Review flag:</b> '
                        'retrieval features detected a possible clinical-text conflict. '
                        'The trial is kept in the ranking because this system retrieves candidates rather than determining eligibility.</div>',
                        unsafe_allow_html=True,
                    )

                summary = str(row.get("brief_summary", "") or "").strip()
                interventions = str(row.get("interventions", "") or "").strip()
                cities = str(row.get("cities", "") or "").strip()
                countries = str(row.get("countries", "") or "").strip()

                left, right = st.columns([1.7, 1], gap="large")
                with left:
                    if summary:
                        st.write("**Study summary**")
                        st.write(summary[:760] + ("…" if len(summary) > 760 else ""))
                    if interventions:
                        st.write("**Interventions**")
                        st.caption(interventions[:430] + ("…" if len(interventions) > 430 else ""))

                with right:
                    st.write("**Snapshot details**")
                    st.caption(f"Sites: {compact_location(cities, countries)}")
                    st.caption(f"Phase: {phase}")
                    st.caption(f"Status: {status}")

                    url = trial_url(row)
                    if url:
                        st.link_button("Open ClinicalTrials.gov", url, use_container_width=True)

                with st.expander("Technical retrieval details"):
                    details: list[tuple[str, float]] = []
                    for label, key in [
                        ("MedCPT score", "medcpt_score"),
                        ("Base hybrid score", "base_score"),
                        ("Lexical score", "lexical_score"),
                        ("Dense score", "dense_score"),
                        ("Cross-encoder score", "cross_encoder_score"),
                        ("Clinical compatibility", "clinical_compatibility"),
                        ("Clinical contradiction", "clinical_contradiction"),
                    ]:
                        value = number(row, key)
                        if value is not None:
                            details.append((label, value))
                    if details:
                        detail_df = pd.DataFrame(details, columns=["Signal", "Value"])
                        st.dataframe(detail_df, hide_index=True, use_container_width=True)
                    else:
                        st.caption("No additional diagnostics are available for this backend.")

                    if scope_full:
                        st.write("**Full recorded study scope**")
                        st.caption(scope_full)
                    if cities or countries:
                        st.write("**Full recorded locations**")
                        full_location = ", ".join(x for x in [cities, countries] if x)
                        st.caption(full_location)

                    st.caption(
                        "These diagnostics explain the retrieval ranking only. They do not constitute "
                        "criterion-level eligibility assessment."
                    )

    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        st.info(
            "If you selected Hybrid/MedCPT, install `requirements-dense.txt` and allow the local "
            "models to download on first use. Lexical mode works without neural models."
        )


st.markdown(
    """
<div class="footer-note">
    <b>Prototype limitation.</b> Retrieval relevance does not establish clinical eligibility.
    Full inclusion/exclusion criteria, current site availability and clinical suitability require professional review.
</div>
""",
    unsafe_allow_html=True,
)