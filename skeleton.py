"""
src/skeleton.py
---------------
COCO-17 skeleton topology used throughout the GaitVision pipeline.
Imported by the Streamlit frontend and the Modal backend.
"""

# ── Node labels ──────────────────────────────────────────────
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

NUM_NODES = len(KEYPOINT_NAMES)   # 17

# ── Edges (undirected) ───────────────────────────────────────
# Format: (parent_idx, child_idx)
SKELETON_EDGES = [
    # Head
    (0,  1),  # nose → left eye
    (0,  2),  # nose → right eye
    (1,  3),  # left eye → left ear
    (2,  4),  # right eye → right ear
    # Arms
    (5,  7),  # left shoulder → left elbow
    (7,  9),  # left elbow → left wrist
    (6,  8),  # right shoulder → right elbow
    (8, 10),  # right elbow → right wrist
    # Torso
    (5,  6),  # shoulder bar
    (5, 11),  # left shoulder → left hip
    (6, 12),  # right shoulder → right hip
    (11,12),  # hip bar
    # Legs
    (11,13),  # left hip → left knee
    (13,15),  # left knee → left ankle
    (12,14),  # right hip → right knee
    (14,16),  # right knee → right ankle
]

# ── Body-part groups (for colour-coding) ────────────────────
HEAD_NODES  = [0, 1, 2, 3, 4]
LEFT_NODES  = [1, 3, 5, 7, 9, 11, 13, 15]
RIGHT_NODES = [2, 4, 6, 8, 10, 12, 14, 16]
TORSO_NODES = [5, 6, 11, 12]

# ── Adjacency matrix (for ST-GCN graph convolution) ─────────
import numpy as np

def build_adjacency(strategy: str = "spatial") -> np.ndarray:
    """
    Return a [3, V, V] adjacency tensor for ST-GCN.
    strategy: 'spatial' uses the three-partition spatial attention split.
    """
    V = NUM_NODES
    A = np.zeros((3, V, V), dtype=np.float32)

    # Partition 0: self-loops
    for i in range(V):
        A[0, i, i] = 1

    # Partition 1: centripetal (towards body centre)
    # Partition 2: centrifugal (away from body centre)
    centre = {11, 12, 5, 6}   # hips + shoulders as "centre"

    for (i, j) in SKELETON_EDGES:
        i_central = i in centre
        j_central = j in centre
        if i_central and not j_central:
            A[1, i, j] = 1   # centripetal
            A[2, j, i] = 1   # centrifugal
        elif j_central and not i_central:
            A[1, j, i] = 1
            A[2, i, j] = 1
        else:
            A[1, i, j] = 1
            A[1, j, i] = 1

    # Row-normalise each partition
    for k in range(3):
        row_sum = A[k].sum(axis=1, keepdims=True).clip(min=1)
        A[k] = A[k] / row_sum

    return A
