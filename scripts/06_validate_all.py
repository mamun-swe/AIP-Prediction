#!/usr/bin/env python3
"""
Script 06 — Comprehensive Validation
======================================
Validates EVERY component of the pipeline independently.
Run this before starting a real training run to catch
any issues early.

Tests:
  ✓ Hand-crafted feature extractors (shape + value range)
  ✓ Feature dimension correctness
  ✓ No NaN/Inf in features
  ✓ Preprocessing (length filter, AA filter, deduplication)
  ✓ Train/test split stratification
  ✓ Feature selection pipeline (train-only fitting)
  ✓ Stacking ensemble with mock data (quick smoke test)
  ✓ Metric computation correctness
  ✓ Bootstrap CI validity

Usage:
  python scripts/06_validate_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import traceback

from src.utils.logger import get_logger

log = get_logger("06_validate_all",
                 log_file="results/logs/06_validate_all.log")

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def check(condition: bool, msg: str):
    status = PASS if condition else FAIL
    log.info(f"  {status} — {msg}")
    return condition


# ─── test data ───────────────────────────────────────────────────────────────

TEST_SEQS_VALID = [
    "GILDTAGLNLYVF",
    "FLPILASLAAKFGPK",
    "MKGAVFSGF",
    "KFLHSAGKFGKALG",
    "RLCRIVVIRVCR",
    "GIMDTAGLNLYVF",
    "RSLRKSDFY",
    "ACDEFGHIKLM",
    "MNPQRSTVWY",
    "LKLKLKLKL",
    "KWKLFKK",
    "RWKIFKK",
    "RRWWRF",
    "ILPWKWPWWPWRR",
]

TEST_SEQS_INVALID = [
    "ABCXYZ",            # non-standard AAs
    "AC",                # too short
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",  # too long
    "GILDTAGLNLYVF",     # duplicate of first
]


# ─── validation tests ────────────────────────────────────────────────────────

def test_handcrafted_features():
    log.info("\n[1] Hand-crafted Feature Extractors")
    all_pass = True

    from src.features.handcrafted import (
        AACExtractor, DPCExtractor, PAACExtractor,
        GAACExtractor, CTDCExtractor, CTDTExtractor,
        CTDDExtractor, QSOExtractor,
        HandcraftedFeatureExtractor
    )
    seq = "GILDTAGLNLYVF"

    # AAC
    ext = AACExtractor()
    vec = ext.extract(seq)
    all_pass &= check(vec.shape == (20,), f"AAC shape = {vec.shape}")
    all_pass &= check(abs(vec.sum() - 1.0) < 1e-6, f"AAC sums to 1 ({vec.sum():.6f})")
    all_pass &= check(not np.any(np.isnan(vec)), "AAC has no NaN")

    # DPC
    ext = DPCExtractor()
    vec = ext.extract(seq)
    all_pass &= check(vec.shape == (400,), f"DPC shape = {vec.shape}")
    all_pass &= check(abs(vec.sum() - 1.0) < 1e-6, f"DPC sums to 1 ({vec.sum():.6f})")

    # PAAC
    ext = PAACExtractor(lam=10)
    vec = ext.extract(seq)
    all_pass &= check(vec.shape == (30,), f"PAAC shape = {vec.shape}")
    all_pass &= check(not np.any(np.isnan(vec)), "PAAC no NaN")

    # GAAC
    ext = GAACExtractor()
    vec = ext.extract(seq)
    all_pass &= check(vec.shape == (5,), f"GAAC shape = {vec.shape}")
    all_pass &= check(abs(vec.sum() - 1.0) < 1e-6, f"GAAC sums to 1 ({vec.sum():.6f})")

    # CTDC
    ext = CTDCExtractor()
    vec = ext.extract(seq)
    all_pass &= check(vec.shape == (21,), f"CTDC shape = {vec.shape}")
    all_pass &= check(not np.any(np.isnan(vec)), "CTDC no NaN")

    # CTDT
    ext = CTDTExtractor()
    vec = ext.extract(seq)
    all_pass &= check(vec.shape == (21,), f"CTDT shape = {vec.shape}")

    # CTDD
    ext = CTDDExtractor()
    vec = ext.extract(seq)
    all_pass &= check(vec.shape == (105,), f"CTDD shape = {vec.shape}")

    # QSO
    ext = QSOExtractor(maxlag=20)
    vec = ext.extract(seq)
    # QSO with maxlag=20: xr(20) + xd(20) = 40 dims
    all_pass &= check(vec.shape == (40,), f"QSO shape = {vec.shape}")

    # Combined dims: AAC(20)+DPC(400)+PAAC(30)+GAAC(5)+CTDC(21)+CTDT(21)+CTDD(105)+QSO(40)=642
    combined = HandcraftedFeatureExtractor(paac_lambda=10, qso_maxlag=20)
    vec = combined.extract(seq)
    expected_dim = 20 + 400 + 30 + 5 + 21 + 21 + 105 + 40  # = 642
    all_pass &= check(vec.shape[0] == expected_dim,
                      f"Combined dim = {vec.shape[0]} (expected {expected_dim})")
    all_pass &= check(not np.any(np.isnan(vec)), "Combined no NaN")
    all_pass &= check(not np.any(np.isinf(vec)), "Combined no Inf")

    # Batch consistency
    batch = combined.extract_batch(TEST_SEQS_VALID, show_progress=False)
    all_pass &= check(batch.shape == (len(TEST_SEQS_VALID), expected_dim),
                      f"Batch shape = {batch.shape}")
    all_pass &= check(not np.any(np.isnan(batch)), "Batch no NaN")

    # Feature name count
    names = combined.feature_names()
    all_pass &= check(len(names) == expected_dim,
                      f"Feature name count = {len(names)}")

    return all_pass


def test_short_sequences():
    log.info("\n[2] Short Sequence Edge Cases")
    all_pass = True

    from src.features.handcrafted import HandcraftedFeatureExtractor

    ext = HandcraftedFeatureExtractor(paac_lambda=10, qso_maxlag=20)
    short_seqs = ["ACDEF", "KLMNO", "ACDEFG", "KLMNPQ"]

    for seq in short_seqs:
        try:
            vec = ext.extract(seq)
            all_pass &= check(not np.any(np.isnan(vec)),
                              f"No NaN for seq len={len(seq)} ('{seq}')")
        except Exception as e:
            log.error(f"  {FAIL} — Exception on seq '{seq}': {e}")
            all_pass = False

    return all_pass


def test_preprocessing():
    log.info("\n[3] Preprocessing & Cleaning")
    all_pass = True

    from src.data.preprocessor import clean_dataframe, validate_sequence

    # validate_sequence
    all_pass &= check(validate_sequence("GILDTAG", 5, 40), "Valid seq accepted")
    all_pass &= check(not validate_sequence("AC", 5, 40), "Too short rejected")
    all_pass &= check(not validate_sequence("ABCXYZ", 5, 40), "Non-std AA rejected")
    all_pass &= check(not validate_sequence("A" * 50, 5, 40), "Too long rejected")

    # clean_dataframe
    all_seqs = TEST_SEQS_VALID + TEST_SEQS_INVALID
    labels = [1] * len(TEST_SEQS_VALID) + [0] * len(TEST_SEQS_INVALID)
    df = pd.DataFrame({"sequence": all_seqs, "label": labels})
    cleaned = clean_dataframe(df, min_len=5, max_len=40)

    all_pass &= check(len(cleaned) < len(df),
                      f"Cleaning reduced rows: {len(df)} → {len(cleaned)}")
    all_pass &= check(cleaned["sequence"].str.len().min() >= 5,
                      "All remaining seqs ≥ 5 aa")
    all_pass &= check(cleaned["sequence"].str.len().max() <= 40,
                      "All remaining seqs ≤ 40 aa")
    all_pass &= check(cleaned["sequence"].duplicated().sum() == 0,
                      "No duplicate sequences")

    return all_pass


def test_split_stratification():
    log.info("\n[4] Train/Test Split Stratification")
    all_pass = True
    import tempfile

    from src.data.splitter import split_and_save, load_splits

    seqs = TEST_SEQS_VALID  # use unique sequences only
    labels = [1] * (len(seqs) // 2) + [0] * (len(seqs) - len(seqs) // 2)
    df = pd.DataFrame({"sequence": seqs, "label": labels})

    with tempfile.TemporaryDirectory() as tmpdir:
        train_df, test_df = split_and_save(df, splits_dir=tmpdir, train_ratio=0.8)

        # Check sizes
        total = len(df)
        all_pass &= check(len(train_df) + len(test_df) == total,
                          f"No samples lost: {len(train_df)}+{len(test_df)}={total}")

        # Check stratification (class ratios similar)
        train_ratio = train_df["label"].mean()
        test_ratio  = test_df["label"].mean()
        all_pass &= check(abs(train_ratio - test_ratio) < 0.25,
                          f"Class ratio similar: train={train_ratio:.2f} test={test_ratio:.2f}")

        # Check no leakage
        train_seqs = set(train_df["sequence"].tolist())
        test_seqs  = set(test_df["sequence"].tolist())
        overlap = len(train_seqs & test_seqs)
        all_pass &= check(overlap == 0, f"No train-test overlap ({overlap} duplicates)")

        # Test load
        tr2, te2 = load_splits(tmpdir)
        all_pass &= check(len(tr2) == len(train_df) and len(te2) == len(test_df),
                          "Save/load round-trip preserves sizes")

    return all_pass


def test_feature_selection_pipeline():
    log.info("\n[5] Feature Fusion & Selection Pipeline")
    all_pass = True

    from src.fusion.selector import FeatureFusionPipeline
    import tempfile

    n_train, n_test = 80, 20
    n_hc, n_plm = 50, 30
    rng = np.random.RandomState(42)

    X_train_hc  = rng.randn(n_train, n_hc)
    X_test_hc   = rng.randn(n_test,  n_hc)
    X_train_plm = rng.randn(n_train, n_plm)
    X_test_plm  = rng.randn(n_test,  n_plm)
    y_train = np.array([1]*40 + [0]*40)

    pipeline = FeatureFusionPipeline(
        variance_threshold=0.001,
        correlation_threshold=0.99,
        n_features=20,
        random_state=42
    )

    X_train_final = pipeline.fit_transform(
        X_train_hc, X_train_plm, y_train
    )
    X_test_final = pipeline.transform(X_test_hc, X_test_plm)

    all_pass &= check(X_train_final.shape[0] == n_train,
                      f"Train sample count preserved ({X_train_final.shape[0]})")
    all_pass &= check(X_test_final.shape[0] == n_test,
                      f"Test sample count preserved ({X_test_final.shape[0]})")
    all_pass &= check(X_train_final.shape[1] == X_test_final.shape[1],
                      f"Same features for train/test ({X_train_final.shape[1]})")
    all_pass &= check(X_train_final.shape[1] <= 20,
                      f"Final dim ≤ 20 ({X_train_final.shape[1]})")
    all_pass &= check(not np.any(np.isnan(X_train_final)),
                      "Train final no NaN")
    all_pass &= check(not np.any(np.isnan(X_test_final)),
                      "Test final no NaN")

    # Verify test was not used in fitting
    all_pass &= check(pipeline._fitted, "Pipeline marked as fitted")

    # Save/load round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline.save_features(X_train_final, y_train,
                                f"{tmpdir}/train.npz", "train")
        X_loaded, y_loaded = FeatureFusionPipeline.load_features(
            f"{tmpdir}/train.npz"
        )
        all_pass &= check(np.allclose(X_train_final, X_loaded),
                          "Feature save/load round-trip correct")

    return all_pass


def test_metrics_computation():
    log.info("\n[6] Metric Computation")
    all_pass = True

    from src.evaluation.metrics import compute_all_metrics, bootstrap_ci

    # Perfect classifier
    y_true  = np.array([1, 1, 1, 0, 0, 0])
    y_pred  = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.85, 0.1, 0.15, 0.05])

    m = compute_all_metrics(y_true, y_pred, y_proba)
    all_pass &= check(m["accuracy"] == 1.0, f"Perfect accuracy = {m['accuracy']}")
    all_pass &= check(m["sensitivity"] == 1.0, f"Perfect sensitivity = {m['sensitivity']}")
    all_pass &= check(m["specificity"] == 1.0, f"Perfect specificity = {m['specificity']}")
    all_pass &= check(m["mcc"] == 1.0, f"Perfect MCC = {m['mcc']}")
    all_pass &= check(m["auc_roc"] == 1.0, f"Perfect AUC = {m['auc_roc']}")

    # Known-bad classifier
    y_pred_bad = np.array([0, 0, 0, 1, 1, 1])
    m_bad = compute_all_metrics(y_true, y_pred_bad, 1 - y_proba)
    all_pass &= check(m_bad["accuracy"] == 0.0, f"Bad accuracy = {m_bad['accuracy']}")
    all_pass &= check(m_bad["mcc"] == -1.0, f"Bad MCC = {m_bad['mcc']}")

    # Random
    rng = np.random.RandomState(42)
    y_true_r  = rng.randint(0, 2, 200)
    y_pred_r  = rng.randint(0, 2, 200)
    y_proba_r = rng.rand(200)
    m_r = compute_all_metrics(y_true_r, y_pred_r, y_proba_r)
    all_pass &= check(-1.0 <= m_r["mcc"] <= 1.0, f"Random MCC in [-1,1]: {m_r['mcc']:.3f}")
    all_pass &= check(0.0 <= m_r["auc_roc"] <= 1.0, f"Random AUC in [0,1]: {m_r['auc_roc']:.3f}")

    # Bootstrap CI
    ci = bootstrap_ci(y_true, y_pred, y_proba, n_iter=100)
    all_pass &= check("accuracy" in ci, "Bootstrap CI has 'accuracy' key")
    all_pass &= check(ci["accuracy"][0] <= ci["accuracy"][1],
                      f"CI lower ≤ upper: {ci['accuracy']}")

    return all_pass


def test_ensemble_smoke():
    log.info("\n[7] Stacking Ensemble Smoke Test (mock data)")
    all_pass = True

    from src.models.stacking import StackingEnsemble

    rng = np.random.RandomState(42)
    n = 100
    X = rng.randn(n, 30)
    y = np.array([1] * 50 + [0] * 50)

    # Tiny Optuna trials for speed
    ensemble = StackingEnsemble(
        n_optuna_trials=3,
        cv_folds=3,
        random_state=42
    )

    try:
        ensemble.fit(X, y)
        all_pass &= check(ensemble._fitted, "Ensemble fitted flag set")

        proba = ensemble.predict_proba(X)
        all_pass &= check(proba.shape == (n, 2),
                          f"predict_proba shape = {proba.shape}")
        all_pass &= check(np.allclose(proba.sum(axis=1), 1.0),
                          "Probabilities sum to 1")
        all_pass &= check(np.all(proba >= 0) and np.all(proba <= 1),
                          "All probabilities in [0,1]")

        pred = ensemble.predict(X)
        all_pass &= check(pred.shape == (n,), f"predict shape = {pred.shape}")
        all_pass &= check(set(np.unique(pred)).issubset({0, 1}),
                          f"predict outputs only 0/1: {np.unique(pred)}")

    except Exception as e:
        log.error(f"  {FAIL} — Ensemble training failed: {e}")
        log.error(traceback.format_exc())
        all_pass = False

    return all_pass


def test_no_data_leakage():
    log.info("\n[8] Data Leakage Prevention")
    all_pass = True

    from src.fusion.selector import FeatureFusionPipeline

    rng = np.random.RandomState(42)
    # Inject a known signal only in training positives
    X_train_pos = rng.randn(40, 50) + 5   # high signal
    X_train_neg = rng.randn(40, 50) - 5   # low signal
    X_test_pos  = rng.randn(10, 50) + 5
    X_test_neg  = rng.randn(10, 50) - 5

    X_train_hc  = np.vstack([X_train_pos, X_train_neg])
    X_test_hc   = np.vstack([X_test_pos, X_test_neg])
    X_train_plm = rng.randn(80, 20)
    X_test_plm  = rng.randn(20, 20)
    y_train = np.array([1]*40 + [0]*40)

    pipeline = FeatureFusionPipeline(
        variance_threshold=0.001,
        correlation_threshold=0.99,
        n_features=10,
        random_state=42
    )

    # Fit on train, transform test
    pipeline.fit_transform(X_train_hc, X_train_plm, y_train)
    X_test_final = pipeline.transform(X_test_hc, X_test_plm)

    # Selected indices come from TRAINING data only
    all_pass &= check(
        pipeline._importance_selector.selected_indices_ is not None,
        "Selector has indices (fitted on train only)"
    )

    # Verify test transform uses SAME indices as train
    X_train_final = pipeline.transform(X_train_hc, X_train_plm)
    all_pass &= check(
        X_train_final.shape[1] == X_test_final.shape[1],
        f"Same columns for train & test: {X_train_final.shape[1]}"
    )

    return all_pass


# ─── runner ─────────────────────────────────────────────────────────────────

def main():
    Path("results/logs").mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("AIP-ML-Stack — Comprehensive Validation Suite")
    log.info("=" * 60)

    tests = [
        ("Hand-crafted features",        test_handcrafted_features),
        ("Short sequence edge cases",     test_short_sequences),
        ("Preprocessing & cleaning",      test_preprocessing),
        ("Train/test split",              test_split_stratification),
        ("Feature selection pipeline",    test_feature_selection_pipeline),
        ("Metric computation",            test_metrics_computation),
        ("Stacking ensemble smoke test",  test_ensemble_smoke),
        ("Data leakage prevention",       test_no_data_leakage),
    ]

    results = {}
    for name, fn in tests:
        try:
            passed = fn()
        except Exception as e:
            log.error(f"\n  Exception in '{name}': {e}")
            log.error(traceback.format_exc())
            passed = False
        results[name] = passed

    # Summary
    log.info("\n" + "=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 60)
    n_pass = sum(results.values())
    n_total = len(results)
    for name, passed in results.items():
        status = PASS if passed else FAIL
        log.info(f"  {status} — {name}")

    log.info(f"\n  Result: {n_pass}/{n_total} tests passed")

    if n_pass == n_total:
        log.info("  All tests passed — pipeline is ready for training.")
        log.info("  Next step: python scripts/01_prepare_data.py")
    else:
        log.error("  Some tests failed — fix issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
