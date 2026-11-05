"""Report locally available LiBCaT components without downloading anything."""
from __future__ import annotations

from importlib import util
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def _status(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else '--'}] {label}" + (f" — {detail}" if detail else ""))


def main() -> None:
    sys.path.insert(0, str(ROOT))
    print("LiBCaT environment check (no download, no model initialization)")
    _status("Python >= 3.10", sys.version_info >= (3, 10), platform.python_version())
    for name in ("numpy", "pandas", "requests", "streamlit", "sentence_transformers", "torch", "transformers"):
        _status(name, util.find_spec(name) is not None,
                "required for core" if name in {"numpy", "pandas", "requests"} else "optional or app dependency")
    _status("Synthetic sample", (ROOT / "data/sample/sample_trials.csv").is_file(), "included in repository")
    _status("Current demo snapshot", (ROOT / "data/demo_trials.csv").is_file(), "optional, local only")
    trec = ROOT / "data/trec"
    archives = [trec / f"ClinicalTrials.2023-05-08.trials{i}.zip" for i in range(6)]
    _status("TREC 2023 full corpus", all(p.is_file() for p in archives),
            f"{sum(p.is_file() for p in archives)}/6 local ZIPs")
    _status("TREC topics and qrels", all((trec / name).is_file() for name in ("topics2023.xml", "qrels2023.txt")))
    print("\nTo run the lightweight demo: python -m pip install -r requirements.txt; streamlit run app.py")
    print("To run tests: python -m unittest discover -s tests -v")


if __name__ == "__main__":
    main()
