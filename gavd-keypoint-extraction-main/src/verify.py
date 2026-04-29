"""Render a video clip with extracted keypoints overlaid for visual sanity check.

Works with both:
  - GAVD-style JSON (frames have 'frame_num' — we seek to the seq's first frame
    in the source video and overlay only that range)
  - Generic JSON (frames have 'idx' — we read sequentially from frame 0)

Usage:
  python -m src.verify --video data/videos/<id>.mp4 \
                       --json  data/keypoints/<seq_id>.json \
                       --output samples/<seq_id>_overlay.mp4
"""
import argparse
import json
from pathlib import Path

import cv2

from .skeleton import EDGES


def draw_skeleton(frame, keypoints):
    if keypoints is None:
        return
    pts = [None if k is None else (int(k[0]), int(k[1])) for k in keypoints]
    for a, b in EDGES:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    doc = json.loads(Path(args.json).read_text())
    fkey = "frame_num" if "frame_num" in doc["frames"][0] else "idx"
    frames_kp = {f[fkey]: f["keypoints"] for f in doc["frames"]}
    target_indices = sorted(frames_kp.keys())
    start_idx = target_indices[0]
    end_idx = target_indices[-1]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    written = 0
    cur = start_idx
    while cur <= end_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in frames_kp:
            draw_skeleton(frame, frames_kp[cur])
            label = f"{doc.get('seq_id', '')}  pat={doc.get('gait_pat', '')}  frame {cur}"
            cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
            written += 1
            if args.max_frames and written >= args.max_frames:
                break
        cur += 1

    cap.release()
    writer.release()
    print(f"wrote {written} frames -> {args.output}")


if __name__ == "__main__":
    main()
