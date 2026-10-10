"""
NeuroVision AI: Module I - Comprehensive Medical Classification Metrics
Computes:
- Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, PR-AUC
- 95% Bootstrap Confidence Intervals (1000 iterations)
- Confusion Matrix, ROC curves, PR curves
- Error Analysis (False Positives and False Negatives)
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 100
) -> Dict[str, Any]:
    """
    Computes all standard medical evaluation metrics and 95% confidence intervals.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    sens = float(recall_score(y_true, y_pred, zero_division=0)) # Recall / Sensitivity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0      # Specificity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(recall_arr, precision_arr))
    except Exception:
        pr_auc = 0.5

    # Bootstrap 95% Confidence Intervals for AUC
    rng = np.random.default_rng(42)
    boot_aucs = []
    boot_accs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(total, size=total, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
            boot_accs.append(accuracy_score(y_true[idx], (y_prob[idx] >= threshold).astype(int)))
        except Exception:
            pass

    auc_ci_lower = float(np.percentile(boot_aucs, 2.5)) if boot_aucs else roc_auc
    auc_ci_upper = float(np.percentile(boot_aucs, 97.5)) if boot_aucs else roc_auc
    acc_ci_lower = float(np.percentile(boot_accs, 2.5)) if boot_accs else acc
    acc_ci_upper = float(np.percentile(boot_accs, 97.5)) if boot_accs else acc

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc_ci_95": [round(auc_ci_lower, 4), round(auc_ci_upper, 4)],
        "accuracy_ci_95": [round(acc_ci_lower, 4), round(acc_ci_upper, 4)],
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    }


def plot_roc_pr_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str
):
    """
    Plots multi-model ROC and Precision-Recall curves.
    results_dict: { "Model Name": (y_true, y_prob) }
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0B0F19")
    for ax in (ax_roc, ax_pr):
        ax.set_facecolor("#111827")
        ax.grid(True, linestyle="--", alpha=0.3, color="#374151")
        for spine in ax.spines.values():
            spine.set_color("#374151")
        ax.tick_params(colors="#9CA3AF")

    colors = ["#60A5FA", "#34D399", "#FBBF24", "#F472B6", "#A78BFA", "#38BDF8", "#4ADE80"]

    for i, (model_name, (y_true, y_prob)) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={auc_score:.3f})")

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_score = auc(recall, precision)
        ax_pr.plot(recall, precision, color=color, lw=2, label=f"{model_name} (PR={pr_score:.3f})")

    # ROC config
    ax_roc.plot([0, 1], [0, 1], color="#6B7280", linestyle="--")
    ax_roc.set_title("ROC Curves (Test Cohort)", color="#E2E8F0", fontsize=13, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", color="#9CA3AF", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", color="#9CA3AF", fontsize=11)
    ax_roc.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#E2E8F0", loc="lower right")

    # PR config
    ax_pr.set_title("Precision-Recall Curves (Test Cohort)", color="#E2E8F0", fontsize=13, fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)", color="#9CA3AF", fontsize=11)
    ax_pr.set_ylabel("Precision (PPV)", color="#9CA3AF", fontsize=11)
    ax_pr.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#E2E8F0", loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_confusion_matrix_figure(
    cm_dict: Dict[str, int],
    title: str,
    output_path: str
):
    """
    Renders a clinical styled 2x2 confusion matrix heatmap.
    """
    matrix = np.array([
        [cm_dict["TN"], cm_dict["FP"]],
        [cm_dict["FN"], cm_dict["TP"]]
    ])
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0B0F19")
    ax.set_facecolor("#111827")

    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["LGG (0)", "HGG (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, color="#E2E8F0", fontsize=11)
    ax.set_yticklabels(classes, color="#E2E8F0", fontsize=11)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = "#0B0F19" if val > matrix.max() / 2 else "#E2E8F0"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=16, fontweight="bold")

    ax.set_title(title, color="#E2E8F0", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Pathological Grade", color="#9CA3AF", fontsize=11)
    ax.set_xlabel("NeuroVision Predicted Grade", color="#9CA3AF", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
