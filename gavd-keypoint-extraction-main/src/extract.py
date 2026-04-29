"""Extract COCO-17 skeletal keypoints from videos using YOLOv8-Pose.

Per video produces:
  - <name>.json  : human-readable, pixel-space keypoints + metadata
  - <name>.npy   : ST-GCN-ready tensor, shape (3, T, 17, 1) = (x, y, conf, T, V, M)
                   coordinates are normalized (hip-centered, torso-scaled)

Subject selection: we use Ultralytics tracking to assign stable IDs across
frames, then pick the dominant subject (most frames, largest avg bbox).

Usage:
  python -m src.extract --input data/videos --output data/keypoints
  python -m src.extract --input data/videos/clip.mp4 --output data/keypoints
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .normalize import normalize_sequence
from .skeleton import EDGES, JOINT_NAMES, NUM_JOINTS
from .track import pick_subject_track


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def device_str():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_videos(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def extract_video(model, video_path: Path, device: str, conf=0.25, imgsz=640):
    """Run pose tracking over a video and return per-frame detections + meta."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Ultralytics handles frame iteration + tracker (botsort by default).
    results = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        tracker="botsort.yaml",
    )

    per_frame = []  # list of list of {track_id, bbox, kp(17,3)}
    for r in results:
        frame_dets = []
        if r.boxes is not None and r.keypoints is not None and len(r.boxes) > 0:
            ids = r.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(r.boxes)
            xyxy = r.boxes.xyxy.cpu().numpy()
            kp_xy = r.keypoints.xy.cpu().numpy()      # (n, 17, 2)
            kp_conf = r.keypoints.conf
            kp_conf = kp_conf.cpu().numpy() if kp_conf is not None else np.zeros(kp_xy.shape[:2])

            for i, tid in enumerate(ids):
                kp = np.concatenate([kp_xy[i], kp_conf[i, :, None]], axis=1)  # (17, 3)
                frame_dets.append({
                    "track_id": int(tid) if tid is not None else None,
                    "bbox": xyxy[i].tolist(),
                    "kp": kp,
                })
        per_frame.append(frame_dets)

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "num_frames": len(per_frame),
        "per_frame": per_frame,
    }


def assemble_subject_sequence(per_frame, subject_id):
    """Build a (T, 17, 3) array of the subject's keypoints; NaN for missing frames."""
    T = len(per_frame)
    seq = np.full((T, NUM_JOINTS, 3), np.nan, dtype=np.float32)
    bboxes = [None] * T
    for t, dets in enumerate(per_frame):
        for d in dets:
            if d["track_id"] == subject_id:
                seq[t] = d["kp"]
                bboxes[t] = d["bbox"]
                break
    return seq, bboxes


def write_outputs(out_dir: Path, video_path: Path, meta, subject_id, raw_seq, norm_seq, bboxes):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # JSON: pixel-space, human-readable, per-frame
    frames_json = []
    for t in range(meta["num_frames"]):
        kp = raw_seq[t]
        frames_json.append({
            "idx": t,
            "bbox": bboxes[t],
            "keypoints": [
                None if np.isnan(kp[j, 0]) else [float(kp[j, 0]), float(kp[j, 1]), float(kp[j, 2])]
                for j in range(NUM_JOINTS)
            ],
        })

    json_doc = {
        "video": video_path.name,
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "num_frames": meta["num_frames"],
        "skeleton": "coco17",
        "joints": JOINT_NAMES,
        "edges": EDGES,
        "subject_track_id": subject_id,
        "frames": frames_json,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(json_doc, indent=2))

    # NPY: ST-GCN tensor (C, T, V, M) with normalized coords
    # transpose (T, V, C) -> (C, T, V), then add M=1 axis at the end
    tensor = np.transpose(norm_seq, (2, 0, 1))[..., None].astype(np.float32)
    np.save(out_dir / f"{stem}.npy", tensor)


def process_one(model, video_path: Path, out_dir: Path, device: str):
    t0 = time.time()
    meta = extract_video(model, video_path, device)
    if meta["num_frames"] == 0:
        print(f"  [skip] no frames decoded")
        return False

    subject_id = pick_subject_track(
        [[{"track_id": d["track_id"], "bbox": d["bbox"]} for d in dets] for dets in meta["per_frame"]]
    )
    if subject_id is None:
        print(f"  [skip] no person detected in any frame")
        return False

    raw_seq, bboxes = assemble_subject_sequence(meta["per_frame"], subject_id)
    norm_seq = normalize_sequence(raw_seq)
    write_outputs(out_dir, video_path, meta, subject_id, raw_seq, norm_seq, bboxes)

    detected = int(np.sum(~np.isnan(raw_seq[:, 0, 0])))
    dt = time.time() - t0
    print(f"  subject={subject_id}  frames={meta['num_frames']}  detected={detected}  ({dt:.1f}s)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file or directory of videos")
    ap.add_argument("--output", required=True, help="output directory for .json + .npy")
    ap.add_argument("--model", default="yolov8m-pose.pt",
                    help="Ultralytics pose model (n/s/m/l/x). m is a good speed/accuracy default.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip videos whose .json already exists in output dir")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    if not in_path.exists():
        print(f"input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    videos = list_videos(in_path)
    if not videos:
        print(f"no videos found under {in_path}", file=sys.stderr)
        sys.exit(1)

    device = device_str()
    print(f"device={device}  model={args.model}  videos={len(videos)}")
    model = YOLO(args.model)

    ok = 0
    skipped = 0
    failed = 0
    for i, vp in enumerate(videos, 1):
        if args.skip_existing and (out_dir / f"{vp.stem}.json").exists():
            skipped += 1
            continue
        print(f"[{i}/{len(videos)}] {vp.name}")
        try:
            if process_one(model, vp, out_dir, device):
                ok += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [error] {e}")
            failed += 1

    print(f"done. ok={ok} failed={failed} skipped={skipped}")


if __name__ == "__main__":
    main()
