"""One-shot fix for existing per-seq JSONs whose pixel keypoints were stored
in *annotated* (GAVD vid_info) resolution instead of *downloaded* video
resolution. Rescales coords in place and adds video_width/video_height fields.

Safe to re-run — it skips JSONs that already carry video_width/video_height.

Usage:
  python -m src.rescale_jsons --meta data/gavd-meta/data \
                              --videos data/videos \
                              --keypoints data/keypoints
"""
import argparse
import ast
import glob
import json
from pathlib import Path

import cv2
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--keypoints", required=True)
    args = ap.parse_args()

    meta_dir = Path(args.meta)
    videos_dir = Path(args.videos)
    kp_dir = Path(args.keypoints)

    dfs = [pd.read_csv(f, low_memory=False, usecols=["id", "vid_info"])
           for f in sorted(glob.glob(str(meta_dir / "GAVD_Clinical_Annotations_*.csv")))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["id"])
    ann_size = {row["id"]: ast.literal_eval(row["vid_info"]) for _, row in df.iterrows()}

    # Cache actual video dimensions per video_id (lots of seqs share a video)
    actual_size_cache: dict[str, tuple[int, int]] = {}

    def get_actual(video_id: str) -> tuple[int, int] | None:
        if video_id in actual_size_cache:
            return actual_size_cache[video_id]
        p = videos_dir / f"{video_id}.mp4"
        if not p.exists():
            return None
        cap = cv2.VideoCapture(str(p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        actual_size_cache[video_id] = (w, h)
        return (w, h)

    json_files = sorted(glob.glob(str(kp_dir / "*.json")))
    print(f"scanning {len(json_files)} JSONs")
    fixed = skipped = no_video = 0

    for jp in json_files:
        doc = json.loads(Path(jp).read_text())
        if "video_width" in doc and doc["video_width"] is not None:
            skipped += 1
            continue
        vid_id = doc["video_id"]
        actual = get_actual(vid_id)
        if actual is None:
            no_video += 1
            continue
        ann = ann_size.get(vid_id)
        if ann is None:
            no_video += 1
            continue
        sx = actual[0] / ann["width"]
        sy = actual[1] / ann["height"]

        if abs(sx - 1.0) < 1e-3 and abs(sy - 1.0) < 1e-3:
            # already aligned, just stamp the size
            doc["video_width"] = actual[0]
            doc["video_height"] = actual[1]
            Path(jp).write_text(json.dumps(doc))
            skipped += 1
            continue

        for f in doc["frames"]:
            for k in f["keypoints"]:
                if k is None:
                    continue
                k[0] = k[0] * sx
                k[1] = k[1] * sy
        doc["video_width"] = actual[0]
        doc["video_height"] = actual[1]
        Path(jp).write_text(json.dumps(doc))
        fixed += 1

    print(f"done. fixed={fixed} already_correct_or_stamped={skipped} no_video={no_video}")


if __name__ == "__main__":
    main()
