#!/usr/bin/env python3
"""
Script 04 — Train Stacking Ensemble
=====================================
Loads the selected feature matrices and trains the full
XGBoost + LightGBM + SVM (RBF) stacking ensemble with Optuna tuning.

Saves:
  results/models/ensemble.pkl   — trained ensemble
  results/models/cv_metrics.json — per-fold CV metrics
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.fusion.selector import FeatureFusionPipeline
from src.models.stacking import StackingEnsemble
from src.evaluation.metrics import aggregate_cv_metrics, print_metrics_table
from src.utils.logger import get_logger

log = get_logger("04_train_model",
                 log_file="results/logs/04_train_model.log")


def main():
    parser = argparse.ArgumentParser(description="Train stacking ensemble")
    parser.add_argument("--features_dir", default="results/features/fused")
    parser.add_argument("--models_dir", default="results/models")
    parser.add_argument("--n_optuna_trials", type=int, default=50,
                        help="Optuna trials per base learner (reduce to 10 for testing)")
    parser.add_argument("--cv_folds", type=int, default=10,
                        help="Folds for CV evaluation")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--run_cv", action="store_true",
                        help="Run full 10-fold CV evaluation (slow but thorough)")
    args = parser.parse_args()

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    # Load features
    log.info("Loading final features...")
    X_train, y_train = FeatureFusionPipeline.load_features(
        f"{args.features_dir}/train_final.npz"
    )
    log.info(f"Train: X={X_train.shape}, y={y_train.shape}, "
             f"pos={y_train.sum()}, neg={(y_train==0).sum()}")

    # Train ensemble
    log.info("━" * 50)
    log.info("Training stacking ensemble...")
    log.info(f"  Base learners: XGBoost, LightGBM, SVM")
    log.info(f"  Meta-learner : Logistic Regression")
    log.info(f"  Optuna trials: {args.n_optuna_trials} per learner")
    log.info("━" * 50)

    ensemble = StackingEnsemble(
        n_optuna_trials=args.n_optuna_trials,
        cv_folds=5,   # inner CV for OOF generation
        random_state=args.random_state
    )
    ensemble.fit(X_train, y_train)

    # Save trained ensemble
    ensemble.save(f"{args.models_dir}/ensemble.pkl")

    # Save best params
    params_path = f"{args.models_dir}/best_params.json"
    with open(params_path, "w") as f:
        json.dump(ensemble._best_params, f, indent=2)
    log.info(f"Best hyperparameters saved → {params_path}")

    # Optional: run full 10-fold CV evaluation
    if args.run_cv:
        log.info("━" * 50)
        log.info("Running 10-fold cross-validation evaluation...")
        log.info("(This trains a fresh ensemble in each fold — will take time)")

        cv_ensemble = StackingEnsemble(
            n_optuna_trials=args.n_optuna_trials,
            cv_folds=5,
            random_state=args.random_state
        )
        fold_agg = cv_ensemble.cv_evaluate(
            X_train, y_train, n_folds=args.cv_folds
        )

        # Print CV summary
        log.info("\n10-fold CV Results:")
        for metric, vals in fold_agg.items():
            log.info(f"  {metric:<16}: {vals['mean']:.4f} ± {vals['std']:.4f}")

        cv_path = f"{args.models_dir}/cv_metrics.json"
        with open(cv_path, "w") as f:
            json.dump(fold_agg, f, indent=2)
        log.info(f"CV metrics saved → {cv_path}")

    log.info("━" * 50)
    log.info("Training complete.")
    log.info(f"  Model saved → {args.models_dir}/ensemble.pkl")
    log.info("Next step: python scripts/05_evaluate.py")


if __name__ == "__main__":
    main()
