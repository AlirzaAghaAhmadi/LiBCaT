from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

from .text_utils import join_nonempty, clean_text

API_URL = "https://clinicaltrials.gov/api/v2/studies"
KEEP_STATUSES = {"RECRUITING", "NOT_YET_RECRUITING"}


def _get(d: dict[str, Any], *path: str, default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def study_to_row(study: dict[str, Any]) -> dict[str, Any]:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    conds = p.get("conditionsModule", {})
    design = p.get("designModule", {})
    desc = p.get("descriptionModule", {})
    elig = p.get("eligibilityModule", {})
    locmod = p.get("contactsLocationsModule", {})
    arms = p.get("armsInterventionsModule", {})

    locations = locmod.get("locations") or []
    countries = sorted({clean_text(x.get("country")) for x in locations if x.get("country")})
    cities = sorted({clean_text(x.get("city")) for x in locations if x.get("city")})
    interventions = arms.get("interventions") or []
    intervention_names = [clean_text(x.get("name")) for x in interventions if x.get("name")]
    conditions = conds.get("conditions") or []
    phases = design.get("phases") or []

    row = {
        "nct_id": clean_text(ident.get("nctId")),
        "title": clean_text(ident.get("briefTitle")),
        "official_title": clean_text(ident.get("officialTitle")),
        "status": clean_text(status.get("overallStatus")),
        "study_type": clean_text(design.get("studyType")),
        "phase": ", ".join(phases),
        "conditions": "; ".join(map(str, conditions)),
        "interventions": "; ".join(intervention_names),
        "brief_summary": clean_text(desc.get("briefSummary")),
        "eligibility": clean_text(elig.get("eligibilityCriteria")),
        "minimum_age": clean_text(elig.get("minimumAge")),
        "maximum_age": clean_text(elig.get("maximumAge")),
        "sex": clean_text(elig.get("sex")),
        "countries": "; ".join(countries),
        "cities": "; ".join(cities[:30]),
        "last_update": clean_text(status.get("lastUpdatePostDateStruct", {}).get("date") if isinstance(status.get("lastUpdatePostDateStruct"), dict) else ""),
    }
    row["retrieval_text"] = join_nonempty([
        row["title"], row["official_title"], row["conditions"], row["interventions"],
        row["brief_summary"], row["eligibility"], row["phase"],
    ])
    row["trial_url"] = f"https://clinicaltrials.gov/study/{row['nct_id']}" if row["nct_id"] else ""
    return row


def fetch_studies(
    condition: str,
    location: str,
    max_studies: int | None = None,
    page_size: int = 1000,
    timeout: int = 60,
) -> Iterator[dict[str, Any]]:
    """Fetch a broad condition+location snapshot.

    Recruitment status and study type are intentionally restricted during snapshot
    creation, not at app runtime. This follows the project's fixed-table scope.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "clinical-trial-retrieval-feasibility/0.1"})
    token = None
    yielded = 0
    while True:
        params = {
            "query.cond": condition,
            "query.locn": location,
            "pageSize": min(page_size, 1000),
            "format": "json",
            "countTotal": "true",
        }
        if token:
            params["pageToken"] = token
        response = session.get(API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        studies = payload.get("studies", [])
        for study in studies:
            row = study_to_row(study)
            # Pre-snapshot restriction; there are no equivalent runtime hard filters.
            if row["status"] not in KEEP_STATUSES:
                continue
            if row["study_type"] != "INTERVENTIONAL":
                continue
            # query.locn is broad; ensure the selected country is actually listed.
            if location.lower() not in row["countries"].lower():
                continue
            yield row
            yielded += 1
            if max_studies and yielded >= max_studies:
                return
        token = payload.get("nextPageToken")
        if not token:
            return
        time.sleep(0.08)


def build_snapshot(
    output_csv: str | Path,
    condition: str = "Breast Cancer",
    location: str = "United States",
    max_studies: int | None = None,
) -> pd.DataFrame:
    rows = list(fetch_studies(condition, location, max_studies=max_studies))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ClinicalTrials.gov returned no rows after snapshot restrictions.")
    df = df.drop_duplicates(subset=["nct_id"]).reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ClinicalTrials.gov API v2",
        "condition": condition,
        "location": location,
        "statuses": sorted(KEEP_STATUSES),
        "study_type": "INTERVENTIONAL",
        "rows": len(df),
        "api_url": API_URL,
    }
    output_csv.with_name("demo_snapshot_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return df
