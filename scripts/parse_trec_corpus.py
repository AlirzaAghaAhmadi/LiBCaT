from __future__ import annotations

import argparse
import io
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libcat.text_utils import clean_text, join_nonempty


def text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    if node is None:
        return ""
    return clean_text(" ".join(node.itertext()))


def parse_study(raw: bytes) -> dict | None:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None
    nct = text(root, ".//id_info/nct_id")
    title = text(root, ".//brief_title")
    official = text(root, ".//official_title")
    conditions = [clean_text(n.text) for n in root.findall(".//condition") if clean_text(n.text)]
    summary = text(root, ".//brief_summary/textblock")
    detailed = text(root, ".//detailed_description/textblock")
    eligibility = text(root, ".//eligibility/criteria/textblock")
    condition_text = "; ".join(conditions)
    retrieval_text = join_nonempty([title, official, condition_text, summary, detailed, eligibility])
    return {
        "nct_id": nct,
        "title": title,
        "conditions": condition_text,
        "brief_summary": summary,
        "eligibility": eligibility,
        "retrieval_text": retrieval_text,
        "trial_url": f"https://clinicaltrials.gov/study/{nct}" if nct else "",
    }


def looks_like_breast(row: dict) -> bool:
    # Candidate-set restriction, equivalent to fixing the project to one cancer.
    haystack = f"{row.get('conditions','')} {row.get('title','')}".lower()
    return "breast" in haystack and any(x in haystack for x in ["cancer", "carcinoma", "neoplasm", "tumor", "tumour"])


def main():
    p = argparse.ArgumentParser(description="Parse TREC 2023 XML ZIPs into a Breast Cancer candidate table.")
    p.add_argument("--trec-dir", default=str(ROOT / "data" / "trec"))
    p.add_argument("--output", default=str(ROOT / "data" / "trec" / "breast_trials.csv"))
    p.add_argument("--allow-partial", action="store_true", help="Development only: parse fewer than all six ZIPs")
    args = p.parse_args()
    trec_dir = Path(args.trec_dir)
    zips = sorted(trec_dir.glob("ClinicalTrials.2023-05-08.trials*.zip"))
    if not zips:
        raise SystemExit("No TREC corpus ZIPs found. Run: python scripts/download_trec_data.py --corpus")
    expected = {f"ClinicalTrials.2023-05-08.trials{i}.zip" for i in range(6)}
    present = {p.name for p in zips}
    missing = sorted(expected - present)
    if missing and not args.allow_partial:
        raise SystemExit("Incomplete TREC corpus (missing " + ", ".join(missing) + "). "
                         "Download all six ZIPs or use --allow-partial for a non-reportable dev run.")

    rows = []
    total_xml = 0
    for zp in zips:
        print(f"Parsing {zp.name} ...")
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".xml"):
                    continue
                total_xml += 1
                row = parse_study(zf.read(name))
                if row and looks_like_breast(row):
                    rows.append(row)
    if not rows:
        raise SystemExit("No Breast Cancer candidates were found in the supplied XML archives.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["nct_id"]).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Parsed {total_xml:,} XML records; kept {len(df):,} Breast Cancer candidates -> {out}")


if __name__ == "__main__":
    main()
