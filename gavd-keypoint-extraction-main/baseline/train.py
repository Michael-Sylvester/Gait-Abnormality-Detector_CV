"""
Train the GaitCNN baseline for binary gait classification.

Usage:
    python -m baseline.train --keypoint_dir gavd_handoff/data/keypoints \
                             --epochs 50 --batch_size 32 --lr 1e-3
"""

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from baseline.dataset import GaitDataset, load_and_split, build_class_weights, build_sample_weights
from baseline.model import GaitCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoint_dir", default="gavd_handoff/data/keypoints")
    p.add_argument("--out_dir",      default="baseline/runs")
    p.add_argument("--fixed_len",    type=int,   default=150)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, mask, labels in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x, mask)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu()
            preds  = logits.argmax(dim=1).cpu()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    f1  = f1_score(all_labels, all_preds, average="macro")
    auc = roc_auc_score(all_labels, all_probs)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return {"acc": acc, "f1_macro": f1, "auc": auc,
            "labels": all_labels, "preds": all_preds}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels_csv = os.path.join(args.keypoint_dir, "_labels.csv")
    df_train, df_val, df_test = load_and_split(labels_csv, random_state=args.seed)
    print(f"Split sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    print(f"Train class dist:\n{df_train['dataset'].value_counts().to_string()}")

    train_ds = GaitDataset(df_train, args.keypoint_dir, args.fixed_len)
    val_ds   = GaitDataset(df_val,   args.keypoint_dir, args.fixed_len)
    test_ds  = GaitDataset(df_test,  args.keypoint_dir, args.fixed_len)

    sample_weights = build_sample_weights(df_train)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    class_weights = build_class_weights(df_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = GaitCNN(in_channels=34, num_classes=2, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1    = 0.0
    history    = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, mask, labels in train_loader:
            x, mask, labels = x.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": avg_loss, **{k: v for k, v in val_metrics.items()
                                                              if k not in ("labels", "preds")}})

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"val_acc={val_metrics['acc']:.3f} | val_f1={val_metrics['f1_macro']:.3f} | "
              f"val_auc={val_metrics['auc']:.3f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    # --- final test evaluation ---
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== Test Results ===")
    print(f"  Accuracy : {test_metrics['acc']:.3f}")
    print(f"  F1 macro : {test_metrics['f1_macro']:.3f}")
    print(f"  AUC      : {test_metrics['auc']:.3f}")
    print(classification_report(test_metrics["labels"], test_metrics["preds"],
                                target_names=["Normal", "Abnormal"]))

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_results.json", "w") as f:
        json.dump({k: v for k, v in test_metrics.items() if k not in ("labels", "preds")}, f, indent=2)

    print(f"\nCheckpoint and logs saved to {out_dir}/")


if __name__ == "__main__":
    main()
