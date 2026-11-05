from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libcat.clinicaltrials import build_snapshot


def main():
    p = argparse.ArgumentParser(description="Build fixed Breast Cancer + geography demo snapshot.")
    p.add_argument("--condition", default="Breast Cancer")
    p.add_argument("--location", default="United States")
    p.add_argument("--output", default=str(ROOT / "data" / "demo_trials.csv"))
    p.add_argument("--max-studies", type=int, default=None, help="Use 100-300 for a quick API smoke test; omit for full snapshot.")
    args = p.parse_args()

    print(f"Fetching {args.condition!r} trials in {args.location!r} ...")
    df = build_snapshot(args.output, args.condition, args.location, args.max_studies)
    print(f"Saved {len(df):,} fixed candidate trials to {args.output}")
    print("Runtime app will search this table directly; it does not apply patient hard filters.")


if __name__ == "__main__":
    main()
