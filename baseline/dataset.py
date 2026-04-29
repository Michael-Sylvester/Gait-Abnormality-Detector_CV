"""
GaitDataset for binary gait classification (Normal vs Abnormal).

npy layout: (3, T, 17, 1)  —  C=3 (x, y, conf), T=frames, V=17 joints, M=1 person
Coordinates are already normalized (hip-centred, torso-scaled).

Returns per sample:
  x          : float32 tensor (34, fixed_len)  — x,y of 17 joints flattened, NaN→0
  conf_mask  : float32 tensor (fixed_len,)     — mean joint confidence per frame, NaN→0
  label      : int  0=Normal, 1=Abnormal
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit


LABEL_MAP = {"Normal Gait": 0, "Abnormal Gait": 1}
FIXED_LEN = 150


def load_and_split(labels_csv: str, random_state: int = 42):
    df = pd.read_csv(labels_csv)
    df = df[df["detection_rate"] >= 0.5].reset_index(drop=True)

    # 80% train / 10% val / 10% test, grouped by video_id so the same
    # person never appears in more than one split
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=df["video_id"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["video_id"]))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    return df_train, df_val, df_test


class GaitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, keypoint_dir: str, fixed_len: int = FIXED_LEN):
        self.df = df
        self.keypoint_dir = keypoint_dir
        self.fixed_len = fixed_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = LABEL_MAP[row["dataset"]]

        arr = np.load(os.path.join(self.keypoint_dir, f"{row['seq_id']}.npy"))
        # arr: (3, T, 17, 1)
        arr = arr[:, :, :, 0]           # (3, T, 17)
        xy   = arr[:2].transpose(1, 0, 2)   # (T, 2, 17)
        conf = arr[2]                        # (T, 17)

        # replace NaN with 0
        xy   = np.nan_to_num(xy,   nan=0.0)   # (T, 2, 17)
        conf = np.nan_to_num(conf, nan=0.0)   # (T, 17)

        T = xy.shape[0]
        xy_flat = xy.reshape(T, -1)            # (T, 34)
        conf_frame = conf.mean(axis=1)         # (T,)  — mean conf per frame

        xy_flat, conf_frame = self._fix_length(xy_flat, conf_frame)

        x = torch.tensor(xy_flat, dtype=torch.float32).T   # (34, fixed_len)
        m = torch.tensor(conf_frame, dtype=torch.float32)  # (fixed_len,)
        return x, m, label

    def _fix_length(self, xy: np.ndarray, conf: np.ndarray):
        T = xy.shape[0]
        if T >= self.fixed_len:
            start = (T - self.fixed_len) // 2
            return xy[start: start + self.fixed_len], conf[start: start + self.fixed_len]
        pad_len = self.fixed_len - T
        xy_pad   = np.zeros((pad_len, xy.shape[1]),  dtype=xy.dtype)
        conf_pad = np.zeros((pad_len,),              dtype=conf.dtype)
        return np.concatenate([xy, xy_pad]), np.concatenate([conf, conf_pad])


def build_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts  = df["dataset"].value_counts()
    n_total = len(df)
    n_cls   = len(LABEL_MAP)
    weights = torch.zeros(n_cls)
    for name, idx in LABEL_MAP.items():
        weights[idx] = n_total / (n_cls * counts[name])
    return weights


def build_sample_weights(df: pd.DataFrame) -> list:
    cw = build_class_weights(df)
    return [cw[LABEL_MAP[row["dataset"]]].item() for _, row in df.iterrows()]
