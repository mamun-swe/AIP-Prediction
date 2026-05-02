#!/usr/bin/env python3
"""
Script 03 — Feature Fusion & Selection
========================================
Fuses hand-crafted and PLM features, then applies:
  Stage 1: Variance filter + correlation filter (unsupervised)
  Stage 2: XGBoost importance-based top-k selection (supervised, train only)

CRITICAL: All selection is fitted on TRAINING data only.
          Test data is transformed using the fitted pipeline.

Outputs:
  results/features/fused/train_final.npz  (X, y)
  results/features/fused/test_final.npz   (X, y)
  results/features/fused/pipeline.pkl     (fitted pipeline)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json

from src.data.splitter import load_splits
from src.fusion.selector import FeatureFusionPipeline
from src.utils.logger import get_logger

log = get_logger("03_select_features",
                 log_file="results/logs/03_select_features.log")


def load_features(hc_only: bool = False):
    """Load saved hand-crafted and optionally PLM features."""
    log.info("Loading hand-crafted features...")
    X_train_hc = np.load("results/features/handcrafted/train_hc.npy")
    X_test_hc  = np.load("results/features/handcrafted/test_hc.npy")
    log.info(f"  HC train: {X_train_hc.shape}, test: {X_test_hc.shape}")

    if hc_only:
        return X_train_hc, X_test_hc, None, None

    plm_train_path = Path("results/features/plm/train_plm.npy")
    plm_test_path  = Path("results/features/plm/test_plm.npy")

    if not plm_train_path.exists():
        log.warning("PLM features not found — using hand-crafted only.")
        log.warning("Run script 02 without --hc_only to generate PLM features.")
        return X_train_hc, X_test_hc, None, None

    log.info("Loading PLM features...")
    X_train_plm = np.load(plm_train_path)
    X_test_plm  = np.load(plm_test_path)
    log.info(f"  PLM train: {X_train_plm.shape}, test: {X_test_plm.shape}")

    return X_train_hc, X_test_hc, X_train_plm, X_test_plm


def main():
    parser = argparse.ArgumentParser(description="Feature fusion & selection")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--n_features", type=int, default=500,
                        help="Final number of features after selection")
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--correlation_threshold", type=float, default=0.95)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--hc_only", action="store_true",
                        help="Use hand-crafted features only (no PLM)")
    args = parser.parse_args()

    Path("results/logs").mkdir(parents=True, exist_ok=True)

    # Load labels
    train_df, test_df = load_splits(args.splits_dir)
    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    log.info(f"Train: {len(y_train)} | pos={y_train.sum()} | neg={(y_train==0).sum()}")
    log.info(f"Test : {len(y_test)}  | pos={y_test.sum()} | neg={(y_test==0).sum()}")

    # Load features
    X_train_hc, X_test_hc, X_train_plm, X_test_plm = load_features(args.hc_only)

    # Handle case where PLM not available
    if X_train_plm is None:
        log.info("Proceeding with hand-crafted features only.")
        # Create dummy zero PLM array (will be filtered by variance filter)
        X_train_plm = np.zeros((len(y_train), 1))
        X_test_plm  = np.zeros((len(y_test), 1))

    # Check NaN/Inf
    for name, X in [("HC_train", X_train_hc), ("HC_test", X_test_hc),
                    ("PLM_train", X_train_plm), ("PLM_test", X_test_plm)]:
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            log.warning(f"{name}: NaN={nan_count}, Inf={inf_count} — replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Load feature names if available
    hc_names_path = "results/features/handcrafted/feature_names.json"
    feature_names = None
    if Path(hc_names_path).exists():
        with open(hc_names_path) as f:
            hc_names = json.load(f)
        plm_names = [f"PLM_{i}" for i in range(X_train_plm.shape[1])]
        feature_names = hc_names + plm_names
        log.info(f"Total feature names: {len(feature_names)}")

    # Fusion + selection pipeline — fit on TRAIN only
    log.info("─" * 50)
    log.info(f"Fusing and selecting features (target: {args.n_features})...")
    pipeline = FeatureFusionPipeline(
        variance_threshold=args.variance_threshold,
        correlation_threshold=args.correlation_threshold,
        n_features=args.n_features,
        random_state=args.random_state,
    )

    X_train_final = pipeline.fit_transform(
        X_train_hc, X_train_plm, y_train, feature_names=feature_names
    )

    # Transform test using FITTED pipeline (no refit)
    X_test_final = pipeline.transform(X_test_hc, X_test_plm)

    log.info(f"Final shapes — Train: {X_train_final.shape}, Test: {X_test_final.shape}")

    # Save
    pipeline.save_features(X_train_final, y_train,
                            "results/features/fused/train_final.npz", "train")
    pipeline.save_features(X_test_final, y_test,
                            "results/features/fused/test_final.npz", "test")
    pipeline.save("results/features/fused/pipeline.pkl")

    # Save selected feature names if available
    if pipeline.feature_names_after_selection_:
        with open("results/features/fused/selected_feature_names.json", "w") as f:
            json.dump(pipeline.feature_names_after_selection_, f)
        log.info(f"Selected feature names saved.")

    log.info("━" * 50)
    log.info("Feature selection complete.")
    log.info("  Train: results/features/fused/train_final.npz")
    log.info("  Test : results/features/fused/test_final.npz")
    log.info("Next step: python scripts/04_train_model.py")


if __name__ == "__main__":
    main()
