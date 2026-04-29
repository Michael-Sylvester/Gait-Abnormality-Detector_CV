"""Extract per-sequence keypoints for GAVD using its provided per-frame bboxes.

For each `seq` in the annotations:
  - Open the source video (downloaded by src.download_gavd)
  - Seek to each annotated frame_num
  - Crop to the GAVD bbox (rescaled to whatever resolution we downloaded)
  - Run YOLOv8-Pose on the crop (single-person, top-down)
  - Map keypoints back into full-frame pixel space
  - Normalize (hip-centered, torso-scaled) and save

Outputs per seq:
  data/keypoints/<seq_id>.npy   shape (3, T, 17, 1) float32 — ST-GCN tensor
  data/keypoints/<seq_id>.json  metadata + raw pixel keypoints

Plus a single label index:
  data/keypoints/_labels.csv    seq_id, video_id, dataset, gait_pat, num_frames

Usage:
  python -m src.extract_gavd --meta data/gavd-meta/data \
                             --videos data/videos \
                             --output data/keypoints
"""
import argparse
import ast
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from .normalize import normalize_sequence
from .skeleton import EDGES, JOINT_NAMES, NUM_JOINTS


def device_str():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_dict_field(s):
    if isinstance(s, dict):
        return s
    return ast.literal_eval(s)


def load_annotations(meta_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(meta_dir / "GAVD_Clinical_Annotations_*.csv")))
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def crop_with_padding(frame, bbox, pad=0.10):
    """Crop with a small margin so the pose model has context. Returns (crop, (x0, y0))."""
    h, w = frame.shape[:2]
    cx = bbox["left"] + bbox["width"] / 2
    cy = bbox["top"] + bbox["height"] / 2
    bw = bbox["width"] * (1 + 2 * pad)
    bh = bbox["height"] * (1 + 2 * pad)
    x0 = max(0, int(round(cx - bw / 2)))
    y0 = max(0, int(round(cy - bh / 2)))
    x1 = min(w, int(round(cx + bw / 2)))
    y1 = min(h, int(round(cy + bh / 2)))
    if x1 <= x0 or y1 <= y0:
        return None, (0, 0)
    return frame[y0:y1, x0:x1], (x0, y0)


def pick_pose_in_crop(result, target_size):
    """Within the crop, the gait subject is the largest detection. Returns (17,3) or None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.keypoints is None:
        return None
    xyxy = result.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(np.argmax(areas))
    kp_xy = result.keypoints.xy.cpu().numpy()[i]      # (17, 2)
    kp_conf = result.keypoints.conf
    kp_conf = kp_conf.cpu().numpy()[i] if kp_conf is not None else np.zeros(NUM_JOINTS)
    return np.concatenate([kp_xy, kp_conf[:, None]], axis=1)  # (17, 3)


def extract_seq(model, video_path: Path, seq_df: pd.DataFrame, device: str,
                annotated_size: tuple[int, int]) -> np.ndarray | None:
    """
    Returns raw pixel keypoints (T, 17, 3) in the *annotated* coordinate space,
    or None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ann_w, ann_h = annotated_size
    sx = actual_w / ann_w if ann_w else 1.0
    sy = actual_h / ann_h if ann_h else 1.0

    rows = seq_df.sort_values("frame_num").reset_index(drop=True)
    out = np.full((len(rows), NUM_JOINTS, 3), np.nan, dtype=np.float32)

    # Sequential reads are 10-50x faster than POS_FRAMES seeks for H.264 video,
    # because each seek decodes from the nearest keyframe. We seek only when the
    # frame_num jumps backward or forward by more than SEEK_THRESHOLD frames.
    SEEK_THRESHOLD = 30
    next_pos = -1  # the frame index that the next cap.read() will return

    for i, row in rows.iterrows():
        frame_num = int(row["frame_num"])
        gap = frame_num - next_pos

        if next_pos < 0 or gap < 0 or gap > SEEK_THRESHOLD:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            next_pos = frame_num
        elif gap > 0:
            # Skip ahead with grab() — no decoding, much faster than read().
            for _ in range(gap):
                cap.grab()
                next_pos += 1

        ok, frame = cap.read()
        next_pos += 1
        if not ok or frame is None:
            continue

        bbox_ann = parse_dict_field(row["bbox"])
        bbox = {
            "left": bbox_ann["left"] * sx,
            "top": bbox_ann["top"] * sy,
            "width": bbox_ann["width"] * sx,
            "height": bbox_ann["height"] * sy,
        }
        crop, (x0, y0) = crop_with_padding(frame, bbox)
        if crop is None:
            continue

        results = model.predict(crop, conf=0.25, imgsz=320, device=device, verbose=False)
        kp_crop = pick_pose_in_crop(results[0], crop.shape[:2])
        if kp_crop is None:
            continue

        # Map back to full-frame pixels in the *downloaded* video's resolution.
        # The JSON viewer (or any user) opens the video at its actual size, so
        # keypoints must match that. Normalization in the .npy tensor is
        # scale-invariant, so this choice doesn't affect ST-GCN training.
        kp_full = kp_crop.copy()
        kp_full[:, 0] = kp_full[:, 0] + x0
        kp_full[:, 1] = kp_full[:, 1] + y0
        out[i] = kp_full

    cap.release()
    return out, (actual_w, actual_h)


def write_seq_outputs(out_dir: Path, seq_id: str, video_id: str, dataset: str,
                      gait_pat: str, frame_nums: list[int], raw_seq: np.ndarray,
                      norm_seq: np.ndarray, video_size: tuple[int, int] | None = None):
    frames_json = []
    for i, fn in enumerate(frame_nums):
        kp = raw_seq[i]
        frames_json.append({
            "frame_num": int(fn),
            "keypoints": [
                None if np.isnan(kp[j, 0]) else [float(kp[j, 0]), float(kp[j, 1]), float(kp[j, 2])]
                for j in range(NUM_JOINTS)
            ],
        })
    json_doc = {
        "seq_id": seq_id,
        "video_id": video_id,
        "dataset": dataset,
        "gait_pat": gait_pat,
        "num_frames": len(frame_nums),
        "video_width": video_size[0] if video_size else None,
        "video_height": video_size[1] if video_size else None,
        "skeleton": "coco17",
        "joints": JOINT_NAMES,
        "edges": EDGES,
        "frames": frames_json,
    }
    (out_dir / f"{seq_id}.json").write_text(json.dumps(json_doc))
    tensor = np.transpose(norm_seq, (2, 0, 1))[..., None].astype(np.float32)
    np.save(out_dir / f"{seq_id}.npy", tensor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="yolov8m-pose.pt")
    ap.add_argument("--limit-seqs", type=int, default=0, help="process only first N seqs (0=all)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    meta_dir = Path(args.meta)
    videos_dir = Path(args.videos)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(meta_dir)
    seq_groups = list(df.groupby("seq", sort=False))
    if args.limit_seqs:
        seq_groups = seq_groups[: args.limit_seqs]

    device = device_str()
    print(f"device={device}  model={args.model}  seqs={len(seq_groups)}")
    model = YOLO(args.model)

    label_rows = []
    ok = skipped = no_video = empty = 0
    for i, (seq_id, sdf) in enumerate(seq_groups, 1):
        target_npy = out_dir / f"{seq_id}.npy"
        if args.skip_existing and target_npy.exists():
            skipped += 1
            continue

        first = sdf.iloc[0]
        video_id = first["id"]
        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"[{i}/{len(seq_groups)}] {seq_id}  no video for id={video_id}")
            no_video += 1
            continue

        ann_size = parse_dict_field(first["vid_info"])
        annotated_size = (int(ann_size["width"]), int(ann_size["height"]))

        t0 = time.time()
        raw, video_size = extract_seq(model, video_path, sdf, device, annotated_size)
        if raw is None or np.isnan(raw[:, 0, 0]).all():
            print(f"[{i}/{len(seq_groups)}] {seq_id}  no detections")
            empty += 1
            continue

        norm = normalize_sequence(raw)
        frame_nums = sdf.sort_values("frame_num")["frame_num"].astype(int).tolist()
        write_seq_outputs(out_dir, seq_id, video_id, str(first["dataset"]),
                          str(first["gait_pat"]), frame_nums, raw, norm, video_size)
        detected = int(np.sum(~np.isnan(raw[:, 0, 0])))
        dt = time.time() - t0
        print(f"[{i}/{len(seq_groups)}] {seq_id}  vid={video_id} pat={first['gait_pat']:<14} "
              f"frames={len(frame_nums)} detected={detected} ({dt:.1f}s)")
        label_rows.append({
            "seq_id": seq_id, "video_id": video_id,
            "dataset": first["dataset"], "gait_pat": first["gait_pat"],
            "num_frames": len(frame_nums), "frames_with_pose": detected,
        })
        ok += 1

    if label_rows:
        labels_path = out_dir / "_labels.csv"
        existing = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
        new = pd.DataFrame(label_rows)
        out_df = pd.concat([existing, new]).drop_duplicates(subset=["seq_id"], keep="last")
        out_df.to_csv(labels_path, index=False)
        print(f"labels written to {labels_path} ({len(out_df)} total seqs)")

    print(f"done. ok={ok} skipped={skipped} no_video={no_video} empty={empty}")


if __name__ == "__main__":
    main()
