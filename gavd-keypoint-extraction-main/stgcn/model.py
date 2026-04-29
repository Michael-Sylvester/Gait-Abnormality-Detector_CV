"""
ST-GCN for binary gait classification, following Yan et al. (2018).

Input :  x    (B, 3, T, 17)  — x, y, conf channels over T frames and 17 joints
         mask (B, T)          — per-frame mean confidence (0 = missing frame)
Output:  logits (B, 2)

Architecture (9 ST-GCN blocks, matching the original paper):
  BN → [64,64,64] → [128(s2),128,128] → [256(s2),256,256] → weighted pool → FC
  s2 = temporal stride 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stgcn.graph import build_adjacency


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, A: np.ndarray,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        K = A.shape[0]  # number of partitions (3)
        self.K = K

        A_tensor = torch.tensor(A, dtype=torch.float32)   # (K, V, V)
        self.register_buffer("A", A_tensor)
        # learnable per-edge importance weight, initialised to 1
        self.M = nn.Parameter(torch.ones_like(A_tensor))

        # spatial: one linear transform per partition, applied simultaneously
        self.gcn_w = nn.Conv2d(in_ch, out_ch * K, kernel_size=1)
        self.gcn_bn = nn.BatchNorm2d(out_ch)

        # temporal: 9-frame conv over T with optional stride, keeping V unchanged
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch,
                      kernel_size=(9, 1),
                      padding=(4, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        # residual applied after tcn so strides match

        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, V)
        B, C, T, V = x.shape

        # --- spatial graph convolution ---
        y = self.gcn_w(x)                          # (B, C_out*K, T, V)
        y = y.view(B, self.K, -1, T, V)            # (B, K, C_out, T, V)

        A_eff = self.A * self.M                     # (K, V, V)  element-wise importance
        # aggregate: output[b,c,t,i] = sum_k sum_j A[k,i,j] * y[b,k,c,t,j]
        y = torch.einsum("kij, bkctj -> bcti", A_eff, y)  # (B, C_out, T, V)
        y = F.relu(self.gcn_bn(y), inplace=True)

        # --- temporal convolution (stride applied here) ---
        y = self.tcn(y)                             # (B, C_out, T//stride, V)

        # --- residual (same stride so shapes match) ---
        return F.relu(y + self.residual(x), inplace=True)


class STGCN(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        A = build_adjacency()   # (3, 17, 17)

        self.bn_input = nn.BatchNorm1d(in_channels * 17)

        # 9 blocks — channels and strides matching the original paper
        cfg = [
            # (out_ch, stride)
            (64,  1),
            (64,  1),
            (64,  1),
            (128, 2),
            (128, 1),
            (128, 1),
            (256, 2),
            (256, 1),
            (256, 1),
        ]
        layers = []
        in_ch = in_channels
        for out_ch, stride in cfg:
            layers.append(STGCNBlock(in_ch, out_ch, A, stride=stride, dropout=dropout))
            in_ch = out_ch
        self.blocks = nn.ModuleList(layers)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, 3, T, 17)
        mask : (B, T)
        """
        B, C, T, V = x.shape

        # input batch norm over (C*V) per frame
        y = x.permute(0, 2, 1, 3).contiguous().view(B * T, C * V)
        y = self.bn_input(y)
        y = y.view(B, T, C, V).permute(0, 2, 1, 3)   # (B, C, T, V)

        for block in self.blocks:
            y = block(y)
        # y: (B, 256, T', V)  where T' = T // 4 due to two stride-2 blocks

        # --- weighted temporal pooling ---
        # downsample the mask to match T'
        T_out = y.shape[2]
        m = F.adaptive_avg_pool1d(mask.unsqueeze(1), T_out).squeeze(1)  # (B, T')
        w = m / (m.sum(dim=1, keepdim=True) + 1e-8)                     # (B, T')

        # pool over both T' and V
        pooled = (y * w.unsqueeze(1).unsqueeze(-1)).sum(dim=2)   # (B, 256, V)
        pooled = pooled.mean(dim=-1)                              # (B, 256)

        return self.classifier(pooled)
