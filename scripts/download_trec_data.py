from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "trec"
DEST.mkdir(parents=True, exist_ok=True)

# Use the track website as the first source for topics because some NIST endpoints
# return HTTP 403 to Python's default downloader. Keep the official NIST URL as
# a fallback. Qrels are hosted by NIST.
FILES = {
    "topics2023.xml": [
        "https://trec-cds.org/topics2023.xml",
        "https://trec.nist.gov/data/trials/topics2023.xml",
    ],
    "qrels2023.txt": [
        "https://trec.nist.gov/data/trials/qrels2023.txt",
    ],
}

CORPUS = {
    f"ClinicalTrials.2023-05-08.trials{i}.zip": [
        f"https://trec-cds.org/2023_data/ClinicalTrials.2023-05-08.trials{i}.zip"
    ]
    for i in range(6)
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://trec.nist.gov/data/trials2023.html",
}


def _human_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def _download_requests(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with requests.get(
        url,
        headers=HEADERS,
        stream=True,
        timeout=(20, 120),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        last_print = 0.0

        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 1.0:
                    if total:
                        pct = downloaded * 100 / total
                        print(
                            f"  {_human_mb(downloaded)} / {_human_mb(total)} ({pct:.1f}%)",
                            end="\r",
                            flush=True,
                        )
                    else:
                        print(f"  {_human_mb(downloaded)}", end="\r", flush=True)
                    last_print = now

    print(" " * 90, end="\r")
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("Downloaded file is empty")
    tmp.replace(path)


def _download_curl(url: str, path: Path) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is not installed")

    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        curl,
        "-fL",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "-A",
        HEADERS["User-Agent"],
        "-e",
        HEADERS["Referer"],
        "-o",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("Downloaded file is empty")
    tmp.replace(path)


def download(urls: list[str], path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path.name} ({_human_mb(path.stat().st_size)})")
        return

    errors: list[str] = []
    for url in urls:
        print(f"downloading: {url}")

        try:
            _download_requests(url, path)
            print(f"saved: {path} ({_human_mb(path.stat().st_size)})")
            return
        except Exception as exc:
            errors.append(f"requests -> {url}: {exc}")
            print(f"  requests failed: {exc}")

        # On some networks NIST blocks scripted HTTP clients but allows curl-like
        # browser requests. Try curl automatically before moving to the next URL.
        try:
            print("  retrying with curl...")
            _download_curl(url, path)
            print(f"saved: {path} ({_human_mb(path.stat().st_size)})")
            return
        except Exception as exc:
            errors.append(f"curl -> {url}: {exc}")
            print(f"  curl failed: {exc}")

    raise RuntimeError(
        f"Could not download {path.name}.\n" + "\n".join(f"- {e}" for e in errors)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="Also download the six large 2023 corpus ZIPs (~2.1 GB total).",
    )
    args = parser.parse_args()

    for name, urls in FILES.items():
        download(urls, DEST / name)

    if args.corpus:
        for name, urls in CORPUS.items():
            download(urls, DEST / name)
    else:
        print("Metadata ready. Add --corpus when you are ready for the large TREC snapshot.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        sys.exit(130)
