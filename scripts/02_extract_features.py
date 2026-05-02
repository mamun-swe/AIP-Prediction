#!/usr/bin/env python3
"""
Script 02 — Feature Extraction
================================
Extracts:
  1. Hand-crafted features (AAC, DPC, PAAC, GAAC, CTDC, CTDT, CTDD, QSO)
     from BOTH train and test using the SAME extractor (no fitting needed —
     these are deterministic transforms).

  2. PLM embeddings (ESM-2 small + medium by default)
     — frozen model, no fine-tuning, mean-pooled last hidden layer.

CRITICAL RULE:
  Feature extractors are fit ONLY on training data.
  Hand-crafted: no fitting needed (deterministic).
  PLM: no fitting (frozen pre-trained models).
  Scaling/selection: done in script 03.

Outputs (saved to results/features/):
  handcrafted/train_hc.npy, test_hc.npy
  plm/train_plm.npy, test_plm.npy
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.splitter import load_splits
from src.features.handcrafted import HandcraftedFeatureExtractor
from src.features.plm_embeddings import PLMFeatureExtractor
from src.utils.logger import get_logger

log = get_logger("02_extract_features",
                 log_file="results/logs/02_extract_features.log")


def extract_and_save_handcrafted(train_seqs, test_seqs,
                                 paac_lambda: int = 10,
                                 qso_maxlag: int = 20):
    """Extract hand-crafted features for train and test sets."""
    extractor = HandcraftedFeatureExtractor(
        paac_lambda=paac_lambda, qso_maxlag=qso_maxlag
    )
    log.info(f"Hand-crafted feature dim: {extractor.n_features}")
    log.info(f"Feature groups: {[name for name, _ in extractor.extractors]}")

    log.info("Extracting hand-crafted features for TRAIN set...")
    X_train_hc = extractor.extract_batch(train_seqs, show_progress=True)

    log.info("Extracting hand-crafted features for TEST set...")
    X_test_hc = extractor.extract_batch(test_seqs, show_progress=True)

    # Save
    Path("results/features/handcrafted").mkdir(parents=True, exist_ok=True)
    np.save("results/features/handcrafted/train_hc.npy", X_train_hc)
    np.save("results/features/handcrafted/test_hc.npy", X_test_hc)

    # Save feature names
    import json
    with open("results/features/handcrafted/feature_names.json", "w") as f:
        json.dump(extractor.feature_names(), f)

    log.info(f"Train HC: {X_train_hc.shape} | Test HC: {X_test_hc.shape}")
    return X_train_hc, X_test_hc


def extract_and_save_plm(train_seqs, test_seqs,
                         device: str = "cpu",
                         batch_size: int = 16,
                         use_esm2_small: bool = True,
                         use_esm2_medium: bool = True,
                         use_prot_t5: bool = False):
    """Extract PLM embeddings for train and test sets."""
    extractor = PLMFeatureExtractor(
        device=device,
        batch_size=batch_size,
        use_esm2_small=use_esm2_small,
        use_esm2_medium=use_esm2_medium,
        use_prot_t5=use_prot_t5,
    )

    log.info("Extracting PLM embeddings for TRAIN set...")
    X_train_plm = extractor.extract_batch(train_seqs, show_progress=True)

    log.info("Extracting PLM embeddings for TEST set...")
    X_test_plm = extractor.extract_batch(test_seqs, show_progress=True)

    Path("results/features/plm").mkdir(parents=True, exist_ok=True)
    np.save("results/features/plm/train_plm.npy", X_train_plm)
    np.save("results/features/plm/test_plm.npy", X_test_plm)

    log.info(f"Train PLM: {X_train_plm.shape} | Test PLM: {X_test_plm.shape}")
    return X_train_plm, X_test_plm


def main():
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--paac_lambda", type=int, default=10)
    parser.add_argument("--qso_maxlag", type=int, default=20)
    parser.add_argument("--device", default="cpu",
                        help="'cpu' or 'cuda' for PLM inference")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--esm2_small", action="store_true", default=True)
    parser.add_argument("--esm2_medium", action="store_true", default=True)
    parser.add_argument("--prot_t5", action="store_true", default=False,
                        help="Enable ProtT5 (requires transformers + ~5GB RAM)")
    parser.add_argument("--hc_only", action="store_true",
                        help="Extract hand-crafted features only (no PLM)")
    args = parser.parse_args()

    Path("results/logs").mkdir(parents=True, exist_ok=True)

    # Load splits
    train_df, test_df = load_splits(args.splits_dir)
    train_seqs = train_df["sequence"].tolist()
    test_seqs = test_df["sequence"].tolist()

    log.info(f"Train: {len(train_seqs)} sequences")
    log.info(f"Test : {len(test_seqs)} sequences")

    # Hand-crafted features
    log.info("─" * 50)
    log.info("STEP 1: Hand-crafted features")
    extract_and_save_handcrafted(
        train_seqs, test_seqs,
        paac_lambda=args.paac_lambda,
        qso_maxlag=args.qso_maxlag
    )

    if not args.hc_only:
        # PLM embeddings
        log.info("─" * 50)
        log.info("STEP 2: PLM embeddings (this requires fair-esm installed)")
        log.info("  If fair-esm is not installed, run:")
        log.info("  pip install fair-esm")
        try:
            extract_and_save_plm(
                train_seqs, test_seqs,
                device=args.device,
                batch_size=args.batch_size,
                use_esm2_small=args.esm2_small,
                use_esm2_medium=args.esm2_medium,
                use_prot_t5=args.prot_t5
            )
        except ImportError as e:
            log.error(f"PLM extraction failed: {e}")
            log.error("Install fair-esm: pip install fair-esm")
            log.error("Then rerun this script without --hc_only")
            sys.exit(1)

    log.info("━" * 50)
    log.info("Feature extraction complete.")
    log.info("  HC  → results/features/handcrafted/")
    if not args.hc_only:
        log.info("  PLM → results/features/plm/")
    log.info("Next step: python scripts/03_select_features.py")


if __name__ == "__main__":
    main()
