"""
Comprehensive evaluation metrics with bootstrap confidence intervals.

Reports (all on independent test set AND cross-validation):
  - Accuracy, Sensitivity (Recall), Specificity
  - Precision, F1-score, MCC
  - AUC-ROC, AUC-PR (average precision)
  - Bootstrap 95% CI for each metric
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_proba: np.ndarray) -> Dict[str, float]:
    """Compute all classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc  = accuracy_score(y_true, y_pred)
    sn   = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity / recall
    sp   = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    try:
        auc_roc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = average_precision_score(y_true, y_proba)
    except ValueError:
        auc_pr = float("nan")

    return {
        "accuracy":    float(acc),
        "sensitivity": float(sn),
        "specificity": float(sp),
        "precision":   float(prec),
        "f1":          float(f1),
        "mcc":         float(mcc),
        "auc_roc":     float(auc_roc),
        "auc_pr":      float(auc_pr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def bootstrap_ci(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_proba: np.ndarray,
                 n_iter: int = 1000,
                 alpha: float = 0.05,
                 random_state: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap 95% confidence intervals for all metrics.

    Returns dict: {metric_name: (lower, upper)}
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    metric_samples = {k: [] for k in
                      ["accuracy", "sensitivity", "specificity",
                       "precision", "f1", "mcc", "auc_roc", "auc_pr"]}

    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]

        # skip if bootstrap sample has only one class
        if len(np.unique(yt)) < 2:
            continue

        m = compute_all_metrics(yt, yp, ypr)
        for k in metric_samples:
            metric_samples[k].append(m[k])

    ci = {}
    lo, hi = alpha / 2, 1 - alpha / 2
    for k, vals in metric_samples.items():
        arr = np.array(vals)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            ci[k] = (float("nan"), float("nan"))
        else:
            ci[k] = (float(np.percentile(arr, lo * 100)),
                     float(np.percentile(arr, hi * 100)))
    return ci


def aggregate_cv_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict]:
    """Aggregate per-fold metrics into mean ± std."""
    keys = [k for k in fold_metrics[0] if k not in ("tp", "tn", "fp", "fn")]
    result = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics if not np.isnan(fm[k])])
        result[k] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            "values": vals.tolist()
        }
    return result


def print_metrics_table(metrics: Dict[str, float],
                        ci: Optional[Dict[str, Tuple]] = None,
                        title: str = "Evaluation Results"):
    """Pretty-print metrics table to console."""
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)
    display_keys = ["accuracy", "sensitivity", "specificity",
                    "precision", "f1", "mcc", "auc_roc", "auc_pr"]
    for k in display_keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if ci and k in ci:
            lo, hi = ci[k]
            log.info(f"  {k:<16}: {v:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        else:
            log.info(f"  {k:<16}: {v:.4f}")
    if "tp" in metrics:
        log.info(f"  Confusion: TP={metrics['tp']} TN={metrics['tn']} "
                 f"FP={metrics['fp']} FN={metrics['fn']}")
    log.info("=" * 60)


def save_metrics(metrics: Dict, ci: Optional[Dict],
                 cv_metrics: Optional[Dict],
                 output_dir: str, prefix: str = ""):
    """Save all metrics as JSON and CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = {
        "independent_test": metrics,
        "confidence_intervals_95": ci or {},
        "cross_validation": cv_metrics or {}
    }
    json_path = Path(output_dir) / f"{prefix}report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved metrics report → {json_path}")

    # CSV summary
    rows = []
    display_keys = ["accuracy", "sensitivity", "specificity",
                    "precision", "f1", "mcc", "auc_roc", "auc_pr"]
    for k in display_keys:
        row = {"metric": k}
        row["independent_test"] = metrics.get(k, float("nan"))
        if ci and k in ci:
            row["ci_lower"], row["ci_upper"] = ci[k]
        if cv_metrics and k in cv_metrics:
            row["cv_mean"] = cv_metrics[k]["mean"]
            row["cv_std"]  = cv_metrics[k]["std"]
        rows.append(row)

    csv_path = Path(output_dir) / f"{prefix}metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.4f")
    log.info(f"Saved metrics CSV → {csv_path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   output_dir: str, prefix: str = ""):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(output_dir) / f"{prefix}roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved ROC curve → {path}")


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  output_dir: str, prefix: str = ""):
    """Plot and save Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    path = Path(output_dir) / f"{prefix}pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved PR curve → {path}")


def plot_cv_boxplot(cv_agg: Dict, output_dir: str, prefix: str = ""):
    """Boxplot of CV metric distributions across folds."""
    display_keys = ["accuracy", "sensitivity", "specificity",
                    "precision", "f1", "mcc", "auc_roc", "auc_pr"]
    labels, data = [], []
    for k in display_keys:
        if k in cv_agg:
            labels.append(k)
            data.append(cv_agg[k]["values"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#B5D4F4")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("10-Fold CV Metric Distribution")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = Path(output_dir) / f"{prefix}cv_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved CV boxplot → {path}")


def full_evaluation(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_proba: np.ndarray,
                    cv_fold_metrics: Optional[List[Dict]] = None,
                    output_dir: str = "results/evaluation",
                    prefix: str = "",
                    n_bootstrap: int = 1000):
    """
    Complete evaluation: metrics, CI, CV aggregation, plots, save.
    """
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    ci = bootstrap_ci(y_true, y_pred, y_proba, n_iter=n_bootstrap)

    print_metrics_table(metrics, ci, title=f"Independent Test — {prefix}")

    cv_agg = None
    if cv_fold_metrics:
        cv_agg = aggregate_cv_metrics(cv_fold_metrics)
        log.info("\nCross-Validation Summary:")
        for k, v in cv_agg.items():
            log.info(f"  {k:<16}: {v['mean']:.4f} ± {v['std']:.4f}")

    save_metrics(metrics, ci, cv_agg, output_dir, prefix=prefix)
    plot_roc_curve(y_true, y_proba, output_dir, prefix=prefix)
    plot_pr_curve(y_true, y_proba, output_dir, prefix=prefix)

    if cv_agg:
        plot_cv_boxplot(cv_agg, output_dir, prefix=prefix)

    return metrics, ci, cv_agg
