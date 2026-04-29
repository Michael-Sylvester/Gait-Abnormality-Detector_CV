"""Pick the gait subject from multi-person detections.

GAVD videos are mostly single-subject, but urban clips have passersby.
We pick the track ID that appears in the most frames; ties broken by
average bounding-box area (the subject is usually closest to the camera).
"""
from collections import defaultdict


def pick_subject_track(per_frame_tracks):
    """
    Args:
        per_frame_tracks: list (one entry per frame) of lists of dicts:
            [{"track_id": int, "bbox": [x1,y1,x2,y2]}, ...]

    Returns:
        track_id of the chosen subject, or None if no detections at all.
    """
    frame_counts = defaultdict(int)
    area_sums = defaultdict(float)

    for detections in per_frame_tracks:
        for det in detections:
            tid = det["track_id"]
            if tid is None:
                continue
            x1, y1, x2, y2 = det["bbox"]
            frame_counts[tid] += 1
            area_sums[tid] += max(0.0, (x2 - x1) * (y2 - y1))

    if not frame_counts:
        return None

    def score(tid):
        n = frame_counts[tid]
        avg_area = area_sums[tid] / n
        return (n, avg_area)

    return max(frame_counts.keys(), key=score)
