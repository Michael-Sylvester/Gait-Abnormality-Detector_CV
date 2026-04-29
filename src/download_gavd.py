"""Download GAVD source videos from YouTube via yt-dlp.

GAVD ships only annotations + YouTube IDs. This script reads the unique
video IDs from the annotation CSVs and downloads each one.

Usage:
  python -m src.download_gavd --meta data/gavd-meta/data --out data/videos
  python -m src.download_gavd --meta data/gavd-meta/data --out data/videos --limit 5

Notes:
  - Some videos may be unavailable (removed, region-locked). They are logged
    to <out>/_download_failures.txt and skipped.
  - We download a single MP4 stream <=720p to keep size reasonable; bboxes
    in the annotations are scaled at extraction time to whatever resolution
    we actually got.
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def load_unique_ids(meta_dir: Path):
    files = sorted(glob.glob(str(meta_dir / "GAVD_Clinical_Annotations_*.csv")))
    if not files:
        raise SystemExit(f"no annotation CSVs found in {meta_dir}")
    dfs = [pd.read_csv(f, low_memory=False, usecols=["id", "url"]) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["id"])
    return df.reset_index(drop=True)


def download_one(yt_id: str, url: str, out_dir: Path) -> tuple[bool, str]:
    """Returns (success, message). Skips if file already exists."""
    target = out_dir / f"{yt_id}.mp4"
    if target.exists() and target.stat().st_size > 0:
        return True, "exists"

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--retries", "5",
        "--fragment-retries", "5",
        "--sleep-requests", "1",
        "--sleep-interval", "2",
        "--max-sleep-interval", "5",
        "-o", str(out_dir / f"{yt_id}.%(ext)s"),
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return False, (proc.stderr.strip().splitlines() or ["unknown"])[-1][:200]
        if not target.exists():
            candidates = list(out_dir.glob(f"{yt_id}.*"))
            if candidates:
                return True, f"got {candidates[0].suffix}"
            return False, "no output file"
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="dir with GAVD_Clinical_Annotations_*.csv")
    ap.add_argument("--out", required=True, help="dir to save videos to")
    ap.add_argument("--limit", type=int, default=0, help="download only first N (0=all)")
    args = ap.parse_args()

    meta_dir = Path(args.meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_unique_ids(meta_dir)
    if args.limit:
        df = df.head(args.limit)
    print(f"unique videos to fetch: {len(df)}")

    failures = []
    ok = skipped = failed = 0
    for i, row in df.iterrows():
        success, msg = download_one(row["id"], row["url"], out_dir)
        tag = "ok " if success and msg != "exists" else ("skip" if msg == "exists" else "FAIL")
        print(f"[{i+1}/{len(df)}] {row['id']:<15} {tag}  {msg}")
        if success and msg == "exists":
            skipped += 1
        elif success:
            ok += 1
        else:
            failed += 1
            failures.append({"id": row["id"], "url": row["url"], "error": msg})

    if failures:
        path = out_dir / "_download_failures.txt"
        path.write_text(json.dumps(failures, indent=2))
        print(f"failures logged to {path}")
    print(f"done. ok={ok} already_present={skipped} failed={failed}")


if __name__ == "__main__":
    main()
