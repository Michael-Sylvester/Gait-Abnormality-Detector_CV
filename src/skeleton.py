"""COCO-17 skeleton definition shared by extraction, normalization, and ST-GCN graph."""

JOINT_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

NUM_JOINTS = 17

EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                    # head
    (5, 6), (5, 11), (6, 12), (11, 12),                # torso
    (5, 7), (7, 9), (6, 8), (8, 10),                   # arms
    (11, 13), (13, 15), (12, 14), (14, 16),            # legs
]

# Indices used by normalization
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
