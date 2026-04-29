"""Translate-and-scale normalization for ST-GCN input.

Per frame:
  1. Translate so the hip midpoint is at the origin.
  2. Scale so the torso length (hip midpoint -> shoulder midpoint) is 1.

This removes camera distance and image-position bias so a video filmed
close-up looks the same as one filmed from across the room.

If hips or shoulders have low confidence in a frame, we fall back to the
last known torso length / hip center, then propagate forward. Frames with
no fallback yet remain NaN and are left for the trainer to mask/interpolate.
"""
import numpy as np

from .skeleton import LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER


def _midpoint(kp, a, b, conf_thresh):
    if kp[a, 2] < conf_thresh or kp[b, 2] < conf_thresh:
        return None
    return (kp[a, :2] + kp[b, :2]) / 2.0


def normalize_sequence(keypoints, conf_thresh=0.3, min_torso_px=1.0):
    """
    Args:
        keypoints: np.ndarray, shape (T, 17, 3) — (x, y, conf) in pixel coords.
                   NaN entries indicate frames with no detection.
        conf_thresh: minimum confidence for a joint to be used in normalization.
        min_torso_px: floor on torso length to avoid division by ~0.

    Returns:
        np.ndarray, shape (T, 17, 3) — normalized (x, y, conf).
    """
    out = np.full_like(keypoints, np.nan, dtype=np.float32)
    last_center = None
    last_torso = None

    for t in range(keypoints.shape[0]):
        kp = keypoints[t]
        if np.isnan(kp).all():
            continue

        hip_c = _midpoint(kp, LEFT_HIP, RIGHT_HIP, conf_thresh)
        sh_c = _midpoint(kp, LEFT_SHOULDER, RIGHT_SHOULDER, conf_thresh)

        if hip_c is not None and sh_c is not None:
            torso = float(np.linalg.norm(sh_c - hip_c))
            if torso >= min_torso_px:
                last_center = hip_c
                last_torso = torso

        if last_center is None or last_torso is None:
            continue  # no anchor yet — leave NaN

        out[t, :, 0] = (kp[:, 0] - last_center[0]) / last_torso
        out[t, :, 1] = (kp[:, 1] - last_center[1]) / last_torso
        out[t, :, 2] = kp[:, 2]

    return out
