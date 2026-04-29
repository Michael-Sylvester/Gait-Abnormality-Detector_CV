"""
Temporal CNN baseline for binary gait classification.

Input :  x    (B, 34, T)  — flattened x,y keypoints over time
         mask (B, T)      — per-frame mean confidence (0 for missing frames)
Output:  logits (B, 2)

Weighted temporal pooling: instead of mean-pooling all T frames equally,
frames are weighted by their confidence mask so missing frames contribute ~0.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GaitCNN(nn.Module):
    def __init__(self, in_channels=34, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64,  kernel=3, dropout=dropout),
            ConvBlock(64,          128, kernel=3, dropout=dropout),
            ConvBlock(128,         256, kernel=3, dropout=dropout),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, mask):
        """
        x    : (B, 34, T)
        mask : (B, T)   values in [0, 1], 0 = missing frame
        """
        feat = self.encoder(x)          # (B, 256, T)

        # weighted temporal pooling
        w = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)   # (B, T), sums to 1
        pooled = (feat * w.unsqueeze(1)).sum(dim=2)          # (B, 256)

        return self.classifier(pooled)  # (B, 2)
