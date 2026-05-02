#!/usr/bin/env python3
"""
Script 05 — Evaluation on Independent Test Set
================================================
Evaluates the trained ensemble on the held-out test set.
Reports both independent test metrics and optionally CV metrics.

This is the HONEST evaluation script — all numbers here are
from data the model has NEVER seen.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.fusion.selector import FeatureFusionPipeline
from src.models.stacking import StackingEnsemble
from src.evaluation.metrics import full_evaluation
from src.utils.logger import get_logger

log = get_logger("05_evaluate",
                 log_file="results/logs/05_evaluate.log")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on test set")
    parser.add_argument("--features_dir", default="results/features/fused")
    parser.add_argument("--models_dir", default="results/models")
    parser.add_argument("--eval_dir", default="results/evaluation")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap iterations for CIs")
    args = parser.parse_args()

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)

    # Load test features
    log.info("Loading test features...")
    X_test, y_test = FeatureFusionPipeline.load_features(
        f"{args.features_dir}/test_final.npz"
    )
    log.info(f"Test: X={X_test.shape}, pos={y_test.sum()}, "
             f"neg={(y_test==0).sum()}")

    # Load model
    log.info(f"Loading ensemble from {args.models_dir}/ensemble.pkl...")
    ensemble = StackingEnsemble.load(f"{args.models_dir}/ensemble.pkl")

    # Predict
    log.info("Generating predictions on independent test set...")
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= args.threshold).astype(int)

    # Load CV metrics if available
    cv_fold_metrics = None
    cv_path = Path(args.models_dir) / "cv_metrics.json"
    if cv_path.exists():
        log.info(f"Loading CV metrics from {cv_path}")
        with open(cv_path) as f:
            cv_agg = json.load(f)
        # Reconstruct fold metrics as list of dicts from aggregated
        # (we have aggregated not per-fold — pass None for fold list)
        cv_fold_metrics = None
        log.info("CV metrics (from training):")
        for k, v in cv_agg.items():
            if isinstance(v, dict) and "mean" in v:
                log.info(f"  {k:<16}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Full evaluation
    log.info("━" * 50)
    log.info("INDEPENDENT TEST SET EVALUATION")
    log.info("━" * 50)

    metrics, ci, _ = full_evaluation(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        cv_fold_metrics=None,
        output_dir=args.eval_dir,
        prefix="test_",
        n_bootstrap=args.n_bootstrap,
    )

    # Print comparison with published methods
    log.info("━" * 50)
    log.info("COMPARISON WITH PUBLISHED METHODS (independent test set):")
    log.info("━" * 50)
    published = [
        ("AntiInflam",   0.720, None,  0.197, 0.647),
        ("AIPpred",      0.744, None,  0.479, 0.814),
        ("iAIPs",        0.751, 0.471, 0.471, 0.822),
        ("AIPStack",     0.755, None,  0.510, 0.819),
        ("IF-AIP",       0.785, None,  0.540, None),
        ("BertAIP",      0.770, None,  0.448, None),
    ]

    log.info(f"  {'Method':<16} {'Accuracy':>10} {'MCC':>10} {'AUC':>10}")
    log.info(f"  {'─'*48}")
    for name, acc, mcc_v, mcc_v2, auc in published:
        mcc_show = mcc_v if mcc_v is not None else mcc_v2
        auc_show = f"{auc:.3f}" if auc else "  —  "
        log.info(f"  {name:<16} {acc:>10.3f} {mcc_show:>10.3f} {auc_show:>10}")

    log.info(f"  {'─'*48}")
    our_acc = metrics['accuracy']
    our_mcc = metrics['mcc']
    our_auc = metrics['auc_roc']
    log.info(f"  {'[Ours]':<16} {our_acc:>10.3f} {our_mcc:>10.3f} {our_auc:>10.3f}")
    log.info("━" * 50)

    log.info("Evaluation complete.")
    log.info(f"  Full report → {args.eval_dir}/test_report.json")
    log.info(f"  Metrics CSV → {args.eval_dir}/test_metrics.csv")
    log.info(f"  ROC curve   → {args.eval_dir}/test_roc_curve.png")
    log.info(f"  PR curve    → {args.eval_dir}/test_pr_curve.png")


if __name__ == "__main__":
    main()
