from __future__ import annotations

from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, min_precision: float = 0.3) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.append(thresholds, 1.0)

    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for p, r, t in zip(precision, recall, thresholds):
        if p < min_precision:
            continue
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)}

    if best["f1"] < 0:
        preds = (y_score >= 0.5).astype(int)
        best = {
            "threshold": 0.5,
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
        }
    return best


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    p, r, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(r, p)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def plot_curves(y_true: np.ndarray, y_score: np.ndarray, fig_prefix: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax[0])
    ax[0].set_title("PR Curve")
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax[1])
    ax[1].set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(f"{fig_prefix}_pr_roc.png", dpi=150)
    plt.close(fig)


def plot_confusion(cm_dict: Dict[str, int], save_path: str) -> None:
    mat = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
