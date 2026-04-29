"""
Precision-recall curve and custom threshold evaluation for the ST-GCN model.

Loads the best checkpoint, runs inference on the chosen split, plots the
precision-recall curve for the Normal class across all thresholds, and
prints a summary table so you can pick an operating point.

Usage — find best threshold on val set (default):
    python -m stgcn.threshold

Usage — evaluate a fixed threshold on the test set:
    python -m stgcn.threshold --split test --threshold 0.85 \
        --keypoint_dir gavd_handoff/data/keypoints \
        --checkpoint stgcn/runs/best_model.pt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader

from baseline.dataset import load_and_split
from stgcn.dataset import GaitDatasetSTGCN
from stgcn.model import STGCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoint_dir", default="gavd_handoff/data/keypoints")
    p.add_argument("--checkpoint",   default="stgcn/runs/best_model.pt")
    p.add_argument("--fixed_len",    type=int,   default=300)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--out_dir",      default="stgcn/runs")
    p.add_argument("--split",        default="val", choices=["val", "test"],
                   help="Which split to evaluate on (default: val)")
    p.add_argument("--threshold",    type=float, default=None,
                   help="Fixed threshold for P(Normal). If omitted, searches "
                        "over thresholds and plots the PR curve.")
    p.add_argument("--in_channels",  type=int, default=3, choices=[2, 3],
                   help="Must match the checkpoint: 3=x,y,conf (default)  2=x,y only (ablation)")
    return p.parse_args()


def collect_probs(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, mask, labels in loader:
            x, mask = x.to(device), mask.to(device)
            probs = torch.softmax(model(x, mask), dim=1).cpu().numpy()
            all_probs.extend(probs[:, 0].tolist())   # P(Normal)
            all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_probs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_csv = os.path.join(args.keypoint_dir, "_labels.csv")
    df_train, df_val, df_test = load_and_split(labels_csv, random_state=args.seed)
    df = df_test if args.split == "test" else df_val

    ds = GaitDatasetSTGCN(df, args.keypoint_dir, args.fixed_len, args.in_channels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = STGCN(in_channels=args.in_channels, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint : {args.checkpoint}")
    print(f"Split             : {args.split}  "
          f"({len(df)} samples — {df['dataset'].value_counts().to_dict()})\n")

    # Ground-truth: 0=Normal, 1=Abnormal
    # We threshold on P(Normal): predict Normal if P(Normal) >= threshold.
    true_labels, prob_normal = collect_probs(model, loader, device)

    # ── fixed-threshold mode: just report and exit ────────────────────────
    if args.threshold is not None:
        t = args.threshold
        preds = 1 - (prob_normal >= t).astype(int)
        acc      = (preds == true_labels).mean()
        macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
        auc      = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                       true_labels, prob_normal)
        print(f"=== {args.split.capitalize()} Results at threshold = {t} ===")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  F1 macro : {macro_f1:.3f}")
        print(f"  AUC      : {auc:.3f}")
        print(classification_report(true_labels, preds,
                                    target_names=["Normal", "Abnormal"], digits=3))
        return

    # ── search mode: PR curve + threshold table (val only) ───────────────
    # sklearn's precision_recall_curve wants the positive class labelled 1.
    # We treat Normal as positive here, so invert the ground-truth labels.
    true_normal = (true_labels == 0).astype(int)
    precision, recall, thresholds = precision_recall_curve(true_normal, prob_normal)
    ap = average_precision_score(true_normal, prob_normal)

    print(f"{'Threshold':>10} {'Normal P':>10} {'Normal R':>10} {'Normal F1':>10} {'Macro F1':>10}")
    print("─" * 55)

    best_thresh, best_macro_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.96, 0.05):
        pred_normal = (prob_normal >= t).astype(int)
        pred_labels = 1 - pred_normal

        macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

        idx = np.searchsorted(thresholds, t, side="right") - 1
        idx = np.clip(idx, 0, len(thresholds) - 1)
        n_prec = float(precision[idx])
        n_rec  = float(recall[idx])
        n_f1   = 2 * n_prec * n_rec / (n_prec + n_rec + 1e-8)

        marker = " ←" if macro_f1 > best_macro_f1 else ""
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresh   = float(t)

        print(f"{t:>10.2f} {n_prec:>10.3f} {n_rec:>10.3f} {n_f1:>10.3f} {macro_f1:>10.3f}{marker}")

    print(f"\nBest threshold by macro F1 : {best_thresh:.2f}  "
          f"(val macro F1 = {best_macro_f1:.3f})")

    print(f"\n--- Val results at threshold = {best_thresh:.2f} ---")
    best_preds = 1 - (prob_normal >= best_thresh).astype(int)
    print(classification_report(true_labels, best_preds,
                                target_names=["Normal", "Abnormal"], digits=3))

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left panel — PR curve
    ax = axes[0]
    ax.plot(recall, precision, color="steelblue", lw=2)
    ax.fill_between(recall, precision, alpha=0.1, color="steelblue")
    ax.set_xlabel("Recall (Normal)")
    ax.set_ylabel("Precision (Normal)")
    ax.set_title(f"Precision-Recall Curve — Normal class\nAP = {ap:.3f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(alpha=0.3)

    for t_annot in sorted({0.30, 0.50, round(best_thresh, 2)}):
        idx = np.searchsorted(thresholds, t_annot, side="right") - 1
        idx = np.clip(idx, 0, len(thresholds) - 1)
        color = "darkred" if abs(t_annot - best_thresh) < 0.01 else "dimgray"
        ax.annotate(
            f"t={t_annot:.2f}",
            xy=(recall[idx], precision[idx]),
            xytext=(recall[idx] + 0.05, precision[idx] - 0.07),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )
        ax.scatter(recall[idx], precision[idx], color=color, s=45, zorder=5)

    # right panel — precision / recall / F1 vs threshold
    ax2 = axes[1]
    ax2.plot(thresholds, precision[:-1], label="Precision (Normal)", color="steelblue",  lw=2)
    ax2.plot(thresholds, recall[:-1],    label="Recall (Normal)",    color="darkorange", lw=2)
    f1_curve = (2 * precision[:-1] * recall[:-1]
                / (precision[:-1] + recall[:-1] + 1e-8))
    ax2.plot(thresholds, f1_curve, label="F1 (Normal)", color="green", lw=2, linestyle="--")
    ax2.axvline(best_thresh, color="darkred", linestyle=":", lw=1.5,
                label=f"Best threshold ({best_thresh:.2f})")
    ax2.axvline(0.5, color="gray", linestyle=":", lw=1,
                label="Default threshold (0.50)")
    ax2.set_xlabel("Threshold on P(Normal)")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision / Recall / F1 vs Threshold\n(Normal class)")
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.out_dir) / "pr_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
