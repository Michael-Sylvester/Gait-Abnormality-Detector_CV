"""
GaitDataset for ST-GCN binary gait classification.

Returns per sample:
  x         : float32 tensor (in_channels, fixed_len, 17)
                in_channels=3 → x, y, conf  (default)
                in_channels=2 → x, y only   (channel ablation)
  conf_mask : float32 tensor (fixed_len,)  — mean joint confidence per frame
              always derived from the conf channel regardless of in_channels,
              so weighted pooling works the same in both ablation conditions.
  label     : int  0=Normal, 1=Abnormal

Reuses load_and_split, build_class_weights, build_sample_weights from baseline.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from baseline.dataset import (
    LABEL_MAP,
    load_and_split,
    build_class_weights,
    build_sample_weights,
)

FIXED_LEN = 300   # matches original ST-GCN paper recommendation


class GaitDatasetSTGCN(Dataset):
    def __init__(self, df: pd.DataFrame, keypoint_dir: str,
                 fixed_len: int = FIXED_LEN, in_channels: int = 3):
        assert in_channels in (2, 3), "in_channels must be 2 (x,y) or 3 (x,y,conf)"
        self.df = df
        self.keypoint_dir = keypoint_dir
        self.fixed_len = fixed_len
        self.in_channels = in_channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = LABEL_MAP[row["dataset"]]

        arr = np.load(os.path.join(self.keypoint_dir, f"{row['seq_id']}.npy"))
        arr = arr[:, :, :, 0]                        # (3, T, 17)
        arr = np.nan_to_num(arr, nan=0.0)

        # conf_frame is always from channel 2 so the weighted pooling mask
        # is consistent across both ablation conditions
        conf_frame = arr[2].mean(axis=1)             # (T,)

        arr, conf_frame = self._fix_length(arr, conf_frame)

        x = torch.tensor(arr[:self.in_channels], dtype=torch.float32)  # (C, fixed_len, 17)
        m = torch.tensor(conf_frame, dtype=torch.float32)              # (fixed_len,)
        return x, m, label

    def _fix_length(self, arr: np.ndarray, conf: np.ndarray):
        T = arr.shape[1]
        if T >= self.fixed_len:
            start = (T - self.fixed_len) // 2
            return arr[:, start: start + self.fixed_len, :], conf[start: start + self.fixed_len]
        pad_len = self.fixed_len - T
        arr_pad  = np.zeros((3, pad_len, 17), dtype=arr.dtype)
        conf_pad = np.zeros(pad_len,          dtype=conf.dtype)
        return np.concatenate([arr, arr_pad], axis=1), np.concatenate([conf, conf_pad])
