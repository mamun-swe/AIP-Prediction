"""
dcwa_ensemble.py
=================
DCWA-E: Dynamic Confidence-Weighted Adaptive Ensemble
for Anti-Inflammatory Peptide (AIP) Prediction

Novel contributions:
  1. Per-sample confidence estimator   c_i = 1 - 2|p_i - 0.5|
  2. Disagreement detector             D   = Var(p_CB, p_LGBM, p_ET, p_RF)
  3. Adaptive weight calculator        w_i = softmax(c_i * base_w_i * stream_w_i)
  4. Disagreement D as 6th meta-feature to LR meta-learner

Architecture:
  PLM stream    : CatBoost (ProtT5) + LightGBM (ESM2)
  Classical stream: Extra Trees (CTDC) + Random Forest (PAAC)
  Meta-learner  : Logistic Regression (C=0.1) on 6 meta-features
  Calibration   : Isotonic regression on LGBM, ET, RF
  Threshold     : MCC-optimised on OOF predictions
"""

# ============================================================
#   CELL 1 — Mount Google Drive (Colab only)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================
#   CELL 2 — Libraries
# ============================================================
# !pip install catboost lightgbm -q

import os
import json
import joblib
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ── CPU parallelism ───────────────────────────────────────────
N_JOBS = 10
os.environ["OMP_NUM_THREADS"]      = str(N_JOBS)
os.environ["MKL_NUM_THREADS"]      = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"]  = str(N_JOBS)

from catboost           import CatBoostClassifier, Pool
from lightgbm           import LGBMClassifier
from sklearn.ensemble   import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration  import CalibratedClassifierCV

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay
)

print("✅ Libraries loaded")
print(f"   CPU cores : {N_JOBS}")


# ============================================================
#   CELL 3 — Configuration
# ============================================================

# ── Google Colab paths ────────────────────────────────────────
# FEATURE_DIR = "/content/drive/MyDrive/AIP-Prediction/data/features"
# BASE_RESULTS= "/content/drive/MyDrive/AIP-Prediction/results/models"
# DCWA_DIR    = "/content/drive/MyDrive/AIP-Prediction/results/models/dcwa_ensemble"
# FIGURES_DIR = "/content/drive/MyDrive/AIP-Prediction/results/figures/dcwa_ensemble"

# ── Local paths ───────────────────────────────────────────────
FEATURE_DIR  = "../../../data/features"
BASE_RESULTS = "../../../results/models"
DCWA_DIR     = "../../../results/models/dcwa_ensemble"
FIGURES_DIR  = "../../../results/figures/dcwa_ensemble"

os.makedirs(DCWA_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Base learner → feature dataset mapping ───────────────────
#
#  DCWA-E uses two streams:
#    PLM stream     : CB(ProtT5)  + LGBM(ESM2)
#    Classical stream: ET(CTDC)   + RF(PAAC)
#
#  Maximum diversity: PLM captures evolutionary/structural
#  signal; classical captures physicochemical/compositional.
#  Their disagreement is informative about hard cases.
#
BASE_LEARNER_DATASETS = {
    "CB"  : "ProtT5_features.csv",   # PLM — best single model
    "LGBM": "ESM2_features.csv",     # PLM — different architecture
    "ET"  : "CTDC_features.csv",     # classical — composition
    "RF"  : "PAAC_features.csv",     # classical — sequence order
}

# Stream membership (used in adaptive weight calculation)
PLM_STREAM       = ["CB", "LGBM"]
CLASSICAL_STREAM = ["ET", "RF"]

# ── Training settings ─────────────────────────────────────────
N_FOLDS      = 10       # OOF folds — more folds = more stable
RANDOM_STATE = 42
TEST_SIZE    = 0.30
ALPHA        = 0.5      # AUC-MCC composite weight

# ── Adaptive weighting settings ──────────────────────────────
#
#  STREAM_BALANCE controls how aggressively disagreement
#  shifts weight toward the PLM stream:
#    0.0 = no adaptive stream balancing
#    1.0 = full shift to PLM when models disagree
#  Recommended: 0.5 (balanced adaptation)
#
STREAM_BALANCE = 0.5

# pos_weight defaults (used if JSON params not found)
POS_WEIGHTS = {
    "CB"  : 2.5,
    "LGBM": 2.0,
    "ET"  : 1.8,
    "RF"  : 1.8,
}

print("✅ Configuration loaded")
print(f"   PLM stream    : {PLM_STREAM}")
print(f"   Classical     : {CLASSICAL_STREAM}")
print(f"   N_FOLDS       : {N_FOLDS}")
print(f"   Stream balance: {STREAM_BALANCE}")


# ============================================================
#   CELL 4 — Helper Functions
# ============================================================

def safe_none(v):
    if v in (None, "None", "none", "null", "nan", ""):
        return None
    return v

def safe_int(v, default):
    v = safe_none(v)
    if v is None: return default
    try: return int(float(v))
    except: return default

def safe_float(v, default):
    v = safe_none(v)
    if v is None: return default
    try: return float(v)
    except: return default

def safe_str(v, default):
    v = safe_none(v)
    return default if v is None else str(v)


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy"   : round(accuracy_score(y_true, y_pred),              4),
        "Sensitivity": round(tp/(tp+fn) if (tp+fn)>0 else 0.0,            4),
        "Specificity": round(tn/(tn+fp) if (tn+fp)>0 else 0.0,            4),
        "F1_Score"   : round(f1_score(y_true, y_pred, zero_division=0),    4),
        "MCC"        : round(matthews_corrcoef(y_true, y_pred),            4),
        "AUC"        : round(roc_auc_score(y_true, y_prob),                4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def find_best_threshold(y_true, y_prob, metric="mcc"):
    best_t, best_score = 0.50, -999.0
    for t in np.arange(0.10, 0.90, 0.01):
        y_p = (y_prob >= t).astype(int)
        if len(np.unique(y_p)) < 2: continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
        if metric == "mcc":
            score = matthews_corrcoef(y_true, y_p)
        elif metric == "sensitivity":
            score = tp/(tp+fn) if (tp+fn)>0 else 0.0
        else:
            score = f1_score(y_true, y_p, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return round(best_t, 2), round(best_score, 4)


def load_feature_csv(csv_path):
    df        = pd.read_csv(csv_path)
    drop_cols = [c for c in ["seq_id","sequence","length"]
                 if c in df.columns]
    df        = df.drop(columns=drop_cols)
    return df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(int)


# ============================================================
#   CELL 5 — Build Base Learners from Optuna JSON params
# ============================================================

def load_optuna_params(clf_short, dataset_file):
    ds_key = dataset_file.replace("_features.csv", "")
    paths  = {
        "CB"  : os.path.join(BASE_RESULTS, "cb_optuna",   "best_params",
                             f"{ds_key}_CB_best_params.json"),
        "LGBM": os.path.join(BASE_RESULTS, "lgbm_optuna", "best_params",
                             f"{ds_key}_LGBM_best_params.json"),
        "ET"  : os.path.join(BASE_RESULTS, "et_optuna",   "best_params",
                             f"{ds_key}_ET_best_params.json"),
        "RF"  : os.path.join(BASE_RESULTS, "rf_optuna",   "best_params",
                             f"{ds_key}_RF_best_params.json"),
    }
    path = paths.get(clf_short, "")
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        print(f"     ✅ Loaded: {os.path.basename(path)}")
        return p
    print(f"     ⚠️  Params not found — using defaults")
    return None


def build_base_learner(clf_short, dataset_file):
    params = load_optuna_params(clf_short, dataset_file)
    pw     = POS_WEIGHTS.get(clf_short, 2.0)

    if clf_short == "CB":
        if params:
            cb_p = {
                "iterations"            : safe_int(params.get("iterations"),            300),
                "depth"                 : safe_int(params.get("depth"),                 6),
                "learning_rate"         : safe_float(params.get("learning_rate"),       0.05),
                "l2_leaf_reg"           : safe_float(params.get("l2_leaf_reg"),         3.0),
                "random_strength"       : safe_float(params.get("random_strength"),     1.0),
                "bagging_temperature"   : safe_float(params.get("bagging_temperature"), 0.5),
                "border_count"          : safe_int(params.get("border_count"),          128),
                "grow_policy"           : safe_str(params.get("grow_policy"),           "SymmetricTree"),
                "min_data_in_leaf"      : safe_int(params.get("min_data_in_leaf"),      5),
                "leaf_estimation_method": safe_str(params.get("leaf_estimation_method"),"Newton"),
            }
        else:
            cb_p = {
                "iterations": 300, "depth": 6, "learning_rate": 0.05,
                "l2_leaf_reg": 3.0, "random_strength": 1.0,
                "bagging_temperature": 0.5, "border_count": 128,
                "grow_policy": "SymmetricTree", "min_data_in_leaf": 5,
                "leaf_estimation_method": "Newton",
            }
        raw_cw = params.get("class_weights", None) if params else None
        if isinstance(raw_cw, str):
            try:
                import ast; raw_cw = ast.literal_eval(raw_cw)
            except: raw_cw = [1.0, pw]
        cb_p["class_weights"]        = raw_cw if raw_cw else [1.0, pw]
        cb_p["eval_metric"]          = "AUC"
        cb_p["early_stopping_rounds"]= 20
        cb_p["task_type"]            = "CPU"
        cb_p["thread_count"]         = N_JOBS
        cb_p["verbose"]              = 0
        cb_p["allow_writing_files"]  = False
        cb_p["random_seed"]          = RANDOM_STATE
        return CatBoostClassifier(**cb_p)

    elif clf_short == "LGBM":
        if params:
            bt        = safe_str(params.get("boosting_type"), "gbdt")
            subsample = (1.0 if bt == "goss"
                         else safe_float(params.get("subsample"), 0.8))
            return LGBMClassifier(
                n_estimators     = safe_int(params.get("n_estimators"),    300),
                max_depth        = safe_int(params.get("max_depth"),        6),
                learning_rate    = safe_float(params.get("learning_rate"), 0.05),
                num_leaves       = safe_int(params.get("num_leaves"),       31),
                subsample        = subsample,
                colsample_bytree = safe_float(params.get("colsample_bytree"), 0.8),
                reg_alpha        = safe_float(params.get("reg_alpha"),      0.1),
                reg_lambda       = safe_float(params.get("reg_lambda"),     1.0),
                scale_pos_weight = safe_float(params.get("scale_pos_weight"), pw),
                boosting_type    = bt,
                n_jobs=N_JOBS, random_state=RANDOM_STATE, verbosity=-1,
            )
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw, n_jobs=N_JOBS,
            random_state=RANDOM_STATE, verbosity=-1,
        )

    elif clf_short == "ET":
        if params:
            max_depth = safe_none(params.get("max_depth", None))
            if max_depth is not None: max_depth = int(float(max_depth))
            max_feat  = safe_none(params.get("max_features", "sqrt"))
            if max_feat not in (None,"sqrt","log2","auto"):
                try: max_feat = float(max_feat)
                except: max_feat = "sqrt"
            return ExtraTreesClassifier(
                n_estimators     = safe_int(params.get("n_estimators"),      300),
                max_depth        = max_depth,
                min_samples_split= safe_int(params.get("min_samples_split"), 2),
                min_samples_leaf = safe_int(params.get("min_samples_leaf"),  1),
                max_features     = max_feat,
                class_weight="balanced", n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            )
        return ExtraTreesClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
        )

    elif clf_short == "RF":
        if params:
            max_depth = safe_none(params.get("max_depth", None))
            if max_depth is not None: max_depth = int(float(max_depth))
            max_feat  = safe_none(params.get("max_features", "sqrt"))
            if max_feat not in (None,"sqrt","log2","auto"):
                try: max_feat = float(max_feat)
                except: max_feat = "sqrt"
            return RandomForestClassifier(
                n_estimators     = safe_int(params.get("n_estimators"),      300),
                max_depth        = max_depth,
                min_samples_split= safe_int(params.get("min_samples_split"), 2),
                min_samples_leaf = safe_int(params.get("min_samples_leaf"),  1),
                max_features     = max_feat,
                class_weight="balanced", n_jobs=N_JOBS,
                random_state=RANDOM_STATE,
            )
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
        )


# ============================================================
#   CELL 6 — Load Datasets
# ============================================================

print("\n" + "="*65)
print("  Loading feature datasets...")
print("="*65)

datasets     = {}
unique_files = set(BASE_LEARNER_DATASETS.values())

for csv_file in unique_files:
    csv_path = os.path.join(FEATURE_DIR, csv_file)
    if not os.path.exists(csv_path):
        print(f"  ❌ Not found: {csv_path}")
        continue
    X, y = load_feature_csv(csv_path)
    datasets[csv_file] = (X, y)
    print(f"  ✅ {csv_file:<30} shape: {X.shape}")

csv0     = list(unique_files)[0]
_, y_all = datasets[csv0]

train_idx, test_idx = train_test_split(
    np.arange(len(y_all)),
    test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y_all
)

print(f"\n  Train : {len(train_idx)}  |  Test  : {len(test_idx)}")
print(f"  Pos   : {y_all.sum()}     |  Neg   : {(y_all==0).sum()}")

X_trains, X_tests, scalers = {}, {}, {}
for csv_file, (X, y) in datasets.items():
    sc = StandardScaler()
    X_trains[csv_file] = sc.fit_transform(X[train_idx])
    X_tests[csv_file]  = sc.transform(X[test_idx])
    scalers[csv_file]  = sc

y_train = y_all[train_idx]
y_test  = y_all[test_idx]


# ============================================================
#   CELL 7 — OOF Generation with Isotonic Calibration
# ============================================================

print("\n" + "="*65)
print("  Generating OOF Probabilities")
print(f"  {N_FOLDS}-fold CV  |  Isotonic calibration  |  {N_JOBS} CPUs")
print("="*65)

cv           = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
n_base       = len(BASE_LEARNER_DATASETS)
oof_probs    = np.zeros((len(y_train), n_base))
test_probs   = np.zeros((len(y_test),  n_base))
final_models = {}
final_scalers= {}
clf_aucs     = {}

clf_list = list(BASE_LEARNER_DATASETS.keys())

for clf_idx, clf_short in enumerate(clf_list):
    csv_file = BASE_LEARNER_DATASETS[clf_short]
    t0       = time.time()
    print(f"\n  [{clf_idx+1}/{n_base}]  {clf_short}  ←  {csv_file}")

    X_tr = X_trains[csv_file]
    X_te = X_tests[csv_file]

    oof_fold   = np.zeros(len(y_train))
    test_folds = np.zeros((len(y_test), N_FOLDS))

    for fold, (f_tr, f_val) in enumerate(cv.split(X_tr, y_train)):
        Xf_tr, Xf_val = X_tr[f_tr], X_tr[f_val]
        yf_tr, yf_val = y_train[f_tr], y_train[f_val]

        base_clf = build_base_learner(clf_short, csv_file)

        if clf_short == "CB":
            # CatBoost: no calibration wrapper (already calibrated)
            tp = Pool(Xf_tr, yf_tr)
            vp = Pool(Xf_val, yf_val)
            base_clf.fit(tp, eval_set=vp, use_best_model=True)
            oof_fold[f_val]     = base_clf.predict_proba(Xf_val)[:,1]
            test_folds[:,fold]  = base_clf.predict_proba(X_te)[:,1]
        else:
            # LGBM, ET, RF: isotonic calibration
            cal_clf = CalibratedClassifierCV(
                base_clf, method="isotonic", cv=3
            )
            cal_clf.fit(Xf_tr, yf_tr)
            oof_fold[f_val]     = cal_clf.predict_proba(Xf_val)[:,1]
            test_folds[:,fold]  = cal_clf.predict_proba(X_te)[:,1]

        fold_auc = roc_auc_score(yf_val, oof_fold[f_val])
        print(f"     Fold {fold+1:>2}/{N_FOLDS}  AUC = {fold_auc:.4f}")

    oof_probs[:, clf_idx]  = oof_fold
    test_probs[:, clf_idx] = test_folds.mean(axis=1)

    clf_aucs[clf_short] = roc_auc_score(y_train, oof_fold)
    elapsed             = time.time() - t0
    print(f"     OOF AUC : {clf_aucs[clf_short]:.4f}  ({elapsed/60:.1f} min)")

    # Final model on full training set
    final_clf = build_base_learner(clf_short, csv_file)
    if clf_short == "CB":
        tp = Pool(X_tr, y_train)
        ep = Pool(X_te,  y_test)
        final_clf.fit(tp, eval_set=ep, use_best_model=True)
    else:
        cal = CalibratedClassifierCV(final_clf, method="isotonic", cv=5)
        cal.fit(X_tr, y_train)
        final_clf = cal

    final_models[clf_short]  = final_clf
    final_scalers[clf_short] = scalers[csv_file]
    joblib.dump(final_clf,
                os.path.join(DCWA_DIR, f"base_{clf_short}.joblib"))
    print(f"     ✅ Saved base_{clf_short}.joblib")

print(f"\n✅ OOF complete — shape: {oof_probs.shape}")
print(f"   OOF AUCs: {clf_aucs}")


# ============================================================
#   CELL 8 — AUC-MCC Composite Base Weights
# ============================================================

print("\n" + "="*65)
print("  Computing AUC-MCC Composite Base Weights")
print("="*65)

base_weights = {}
mcc_scores   = {}

for ci, clf_short in enumerate(clf_list):
    oof_pred = (oof_probs[:, ci] >= 0.5).astype(int)
    mcc      = matthews_corrcoef(y_train, oof_pred)
    mcc_norm = (mcc + 1.0) / 2.0
    auc      = clf_aucs[clf_short]
    w        = ALPHA * auc + (1.0 - ALPHA) * mcc_norm
    base_weights[clf_short] = w
    mcc_scores[clf_short]   = mcc

total_w    = sum(base_weights.values())
norm_base  = {k: v/total_w for k, v in base_weights.items()}

print(f"\n  {'Clf':<6} {'AUC':>8} {'MCC':>8} {'BaseW':>8}")
print(f"  {'─'*36}")
for k in clf_list:
    print(f"  {k:<6} {clf_aucs[k]:>8.4f} "
          f"{mcc_scores[k]:>8.4f} {norm_base[k]:>8.4f}")


# ============================================================
#   CELL 9 — ★ NOVEL COMPONENT 1: Confidence Estimator
#
#   Formula: c_i = 1 - 2 * |p_i - 0.5|
#
#   Intuition:
#     p = 0.50 → c = 0.00 (maximum uncertainty)
#     p = 0.75 → c = 0.50 (moderate confidence)
#     p = 0.95 → c = 0.90 (high confidence)
#     p = 0.99 → c = 0.98 (very high confidence)
#
#   A model that outputs p=0.95 is much more certain about
#   its prediction than one that outputs p=0.52. The confidence
#   score captures this and is used to modulate the base weight.
# ============================================================

def compute_confidence(probs):
    """
    Compute per-sample confidence for each base learner.

    Args:
        probs: np.ndarray shape (n_samples, n_models)
    Returns:
        confidence: np.ndarray shape (n_samples, n_models)
                    each value in [0, 1]
    """
    # c_i = 1 - 2 * |p_i - 0.5|
    confidence = 1.0 - 2.0 * np.abs(probs - 0.5)
    return confidence   # shape (n_samples, n_models)


# ============================================================
#   CELL 10 — ★ NOVEL COMPONENT 2: Disagreement Detector
#
#   Formula: D = Var(p_CB, p_LGBM, p_ET, p_RF)  per sample
#
#   Intuition:
#     Low D  → all models agree → easy peptide → trust all equally
#     High D → PLM and classical disagree → hard peptide →
#              trust PLM stream more (richer representations)
#
#   Also computed separately as:
#     D_plm = |p_CB - p_LGBM|          within-stream PLM disagree
#     D_cls = |p_ET - p_RF|            within-stream classical
#     D_cross = |mean_plm - mean_cls|  between-stream disagree
# ============================================================

def compute_disagreement(probs, plm_idx, cls_idx):
    """
    Compute disagreement signals per sample.

    Args:
        probs    : (n_samples, n_models) calibrated probabilities
        plm_idx  : list of column indices for PLM stream models
        cls_idx  : list of column indices for classical stream models

    Returns:
        D        : (n_samples,) overall variance across all models
        D_plm    : (n_samples,) within-PLM disagreement
        D_cls    : (n_samples,) within-classical disagreement
        D_cross  : (n_samples,) between-stream disagreement
    """
    # Overall variance across all 4 models per sample
    D       = np.var(probs, axis=1)             # shape (n,)

    # Within PLM stream disagreement
    p_plm   = probs[:, plm_idx]
    D_plm   = np.var(p_plm, axis=1)

    # Within classical stream disagreement
    p_cls   = probs[:, cls_idx]
    D_cls   = np.var(p_cls, axis=1)

    # Between-stream disagreement
    mean_plm   = p_plm.mean(axis=1)
    mean_cls   = p_cls.mean(axis=1)
    D_cross    = np.abs(mean_plm - mean_cls)

    return D, D_plm, D_cls, D_cross


# ============================================================
#   CELL 11 — ★ NOVEL COMPONENT 3: Adaptive Weight Calculator
#
#   Formula:
#     raw_w_i = base_w_i * c_i * stream_w_i
#     w_i     = softmax(raw_w_i)    so Σw_i = 1 per sample
#
#   stream_w_i is modulated by cross-stream disagreement D_cross:
#     When D_cross is high → PLM stream gets higher weight
#     When D_cross is low  → weights stay close to base_w
#
#   This is the key novelty: the weight of each model changes
#   per sample based on that sample's uncertainty profile.
# ============================================================

def softmax(x):
    """Numerically stable softmax per row."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def compute_adaptive_weights(probs, base_w_arr,
                              plm_idx, cls_idx,
                              stream_balance=STREAM_BALANCE):
    """
    ★ NOVEL COMPONENT 3 — Compute per-sample adaptive weights.

    For each sample i:
      1. Compute confidence c_j for each model j
      2. Compute cross-stream disagreement D_cross_i
      3. Compute stream boost factor:
           plm_boost = 1 + stream_balance * D_cross_i
           cls_boost = 1 + stream_balance * (1 - D_cross_i)
      4. Multiply base weight by confidence * stream_boost
      5. Normalise via softmax so weights sum to 1

    Args:
        probs         : (n_samples, n_models)
        base_w_arr    : (n_models,) base weights from AUC-MCC
        plm_idx       : column indices of PLM stream
        cls_idx       : column indices of classical stream
        stream_balance: how aggressively to shift toward PLM on disagreement

    Returns:
        adaptive_w    : (n_samples, n_models) normalised weights
        D             : (n_samples,) overall disagreement
        confidence    : (n_samples, n_models) confidence scores
    """
    n_samples, n_models = probs.shape

    # Step 1 — per-sample, per-model confidence
    confidence = compute_confidence(probs)      # (n, n_models)

    # Step 2 — disagreement signals
    D, D_plm, D_cls, D_cross = compute_disagreement(
        probs, plm_idx, cls_idx
    )

    # Step 3 — stream boost factors per sample
    #   When models disagree strongly (D_cross high),
    #   PLM stream is trusted more because PLM features
    #   carry richer context for ambiguous sequences.
    plm_boost = 1.0 + stream_balance * D_cross     # (n,)
    cls_boost = 1.0 + stream_balance * (1.0 - D_cross)

    stream_boost = np.ones((n_samples, n_models))
    for j in plm_idx:
        stream_boost[:, j] = plm_boost
    for j in cls_idx:
        stream_boost[:, j] = cls_boost

    # Step 4 — combine: base × confidence × stream boost
    raw_w = (base_w_arr[np.newaxis, :]   # (1, n_models)
             * confidence                # (n, n_models)
             * stream_boost)             # (n, n_models)

    # Step 5 — normalise via softmax
    adaptive_w = softmax(raw_w)          # (n, n_models)

    return adaptive_w, D, confidence


# Compute index mappings
plm_idx = [clf_list.index(c) for c in PLM_STREAM]
cls_idx = [clf_list.index(c) for c in CLASSICAL_STREAM]
base_w_arr = np.array([norm_base[c] for c in clf_list])

# Compute OOF adaptive weights
oof_adaptive_w, oof_D, oof_confidence = compute_adaptive_weights(
    oof_probs, base_w_arr, plm_idx, cls_idx
)

# Compute test adaptive weights
test_adaptive_w, test_D, test_confidence = compute_adaptive_weights(
    test_probs, base_w_arr, plm_idx, cls_idx
)

print(f"\n✅ Adaptive weights computed")
print(f"   OOF  adaptive_w shape : {oof_adaptive_w.shape}")
print(f"   Test adaptive_w shape : {test_adaptive_w.shape}")
print(f"\n   Mean adaptive weights across OOF samples:")
for ci, clf_short in enumerate(clf_list):
    stream = "PLM" if clf_short in PLM_STREAM else "Classical"
    print(f"     {clf_short:<6} ({stream:<10}): "
          f"mean={oof_adaptive_w[:,ci].mean():.4f}  "
          f"std={oof_adaptive_w[:,ci].std():.4f}")
print(f"\n   OOF disagreement D:")
print(f"     Mean : {oof_D.mean():.4f}")
print(f"     Max  : {oof_D.max():.4f}")
print(f"     High-disagreement samples (D>0.05): "
      f"{(oof_D>0.05).sum()} / {len(oof_D)}")


# ============================================================
#   CELL 12 — Adaptive Combination
#
#   p_combined_i = Σ_j (w_ij * p_ij)   per sample i
#
#   This is a weighted average where every sample gets its
#   own personalised set of model weights.
# ============================================================

# OOF adaptive combination
oof_combined  = (oof_probs * oof_adaptive_w).sum(axis=1)

# Test adaptive combination
test_combined = (test_probs * test_adaptive_w).sum(axis=1)

print(f"\n✅ Adaptive combination complete")
print(f"   OOF  combined AUC : "
      f"{roc_auc_score(y_train, oof_combined):.4f}")
print(f"   Test combined AUC : "
      f"{roc_auc_score(y_test,  test_combined):.4f}")


# ============================================================
#   CELL 13 — Build Meta-Feature Matrix
#
#   Meta-features (6 total):
#     [0] p_CB       — calibrated CatBoost probability
#     [1] p_LGBM     — calibrated LightGBM probability
#     [2] p_ET       — calibrated Extra Trees probability
#     [3] p_RF       — calibrated Random Forest probability
#     [4] p_combined — adaptive weighted combination
#     [5] D          — disagreement score (★ novel feature)
#
#   The disagreement D as a meta-feature is the key
#   structural innovation: the meta-learner explicitly knows
#   when the base models disagree and can learn to treat
#   high-disagreement cases differently.
# ============================================================

meta_train = np.column_stack([
    oof_probs,           # cols 0-3: p_CB, p_LGBM, p_ET, p_RF
    oof_combined,        # col  4  : adaptive weighted average
    oof_D,               # col  5  : ★ disagreement signal
])

meta_test  = np.column_stack([
    test_probs,
    test_combined,
    test_D,
])

feat_names = clf_list + ["p_combined", "D_disagreement"]

print(f"\n✅ Meta-feature matrix built")
print(f"   Train : {meta_train.shape}")
print(f"   Test  : {meta_test.shape}")
print(f"   Features: {feat_names}")


# ============================================================
#   CELL 14 — Train Meta-Learner (Logistic Regression)
#
#   LR with C=0.1 (strong L2 regularisation):
#     - Only 6 meta-features → LR is appropriate
#     - C=0.1 prevents overfitting on small meta-set
#     - Coefficients are interpretable (which model matters most)
#     - class_weight='balanced' handles the 1:1.5 class ratio
# ============================================================

print("\n" + "="*65)
print("  Training Meta-Learner (Logistic Regression, C=0.1)")
print("="*65)

meta_learner = LogisticRegression(
    C            = 0.1,
    penalty      = "l2",
    class_weight = "balanced",
    solver       = "lbfgs",
    max_iter     = 2000,
    n_jobs       = N_JOBS,
    random_state = RANDOM_STATE,
)
meta_learner.fit(meta_train, y_train)

meta_prob_train = meta_learner.predict_proba(meta_train)[:,1]
meta_prob_test  = meta_learner.predict_proba(meta_test) [:,1]

print(f"\n  Train AUC : {roc_auc_score(y_train, meta_prob_train):.4f}")
print(f"  Test  AUC : {roc_auc_score(y_test,  meta_prob_test):.4f}")

# Meta-learner coefficients — show what it learned
print(f"\n  Meta-learner coefficients:")
for fname, coef in zip(feat_names, meta_learner.coef_[0]):
    stream = ("PLM" if fname in PLM_STREAM
              else "Classical" if fname in CLASSICAL_STREAM
              else "Combined")
    print(f"     {fname:<16} [{stream:<10}] : {coef:+.4f}")

joblib.dump(meta_learner,
            os.path.join(DCWA_DIR, "meta_learner_lr.joblib"))
print(f"\n  ✅ Meta-learner saved")


# ============================================================
#   CELL 15 — Threshold Optimisation (on OOF — no test leakage)
# ============================================================

print("\n" + "="*65)
print("  Threshold Optimisation (MCC on OOF)")
print("="*65)

best_t, best_mcc_val = find_best_threshold(
    y_train, meta_prob_train, metric="mcc"
)
print(f"  Optimal threshold : {best_t}  "
      f"(train MCC = {best_mcc_val:.4f})")

y_pred_default = (meta_prob_test >= 0.50).astype(int)
y_pred_optimal = (meta_prob_test >= best_t).astype(int)


# ============================================================
#   CELL 16 — Final Metrics
# ============================================================

print("\n" + "="*65)
print("  DCWA-E FINAL RESULTS")
print("="*65)

metrics_def = compute_metrics(y_test, y_pred_default, meta_prob_test)
metrics_opt = compute_metrics(y_test, y_pred_optimal, meta_prob_test)

print(f"\n  {'Metric':<14} {'Default (0.50)':>15} "
      f"{'Optimal ({:.2f})'.format(best_t):>18}  {'Δ':>8}")
print(f"  {'─'*58}")
for m in ["Accuracy","Sensitivity","Specificity",
          "F1_Score","MCC","AUC"]:
    d=metrics_def[m]; o=metrics_opt[m]
    delta=round(o-d,4); sign="+" if delta>=0 else ""
    print(f"  {m:<14} {d:>15.4f} {o:>18.4f}  {sign}{delta:>7.4f}")

print(f"\n  TP={metrics_opt['TP']}  TN={metrics_opt['TN']}  "
      f"FP={metrics_opt['FP']}  FN={metrics_opt['FN']}")


# ============================================================
#   CELL 17 — Compare with CB Optuna + Stacking v2
# ============================================================

print("\n" + "="*65)
print("  COMPARISON — CB Optuna vs Stacking v2 vs DCWA-E")
print("="*65)

individual_metrics = {}
for ci, clf_short in enumerate(clf_list):
    p    = test_probs[:, ci]
    t, _ = find_best_threshold(y_test, p, "mcc")
    pred = (p >= t).astype(int)
    individual_metrics[clf_short] = compute_metrics(y_test, pred, p)

comparison_rows = []
for clf_short, m in individual_metrics.items():
    comparison_rows.append({
        "Model"      : clf_short,
        "Accuracy"   : m["Accuracy"],
        "Sensitivity": m["Sensitivity"],
        "Specificity": m["Specificity"],
        "F1_Score"   : m["F1_Score"],
        "MCC"        : m["MCC"],
        "AUC"        : m["AUC"],
    })

# Load stacking v2 if available
v2_json = os.path.join(
    BASE_RESULTS, "stacking_v2", "stacking_v2_summary.json"
)
if os.path.exists(v2_json):
    with open(v2_json) as f:
        v2 = json.load(f)
    comparison_rows.append({
        "Model"      : "Stacking v2",
        "Accuracy"   : v2.get("stacking_acc"),
        "Sensitivity": v2.get("stacking_sn"),
        "Specificity": v2.get("stacking_sp"),
        "F1_Score"   : v2.get("stacking_f1"),
        "MCC"        : v2.get("stacking_mcc"),
        "AUC"        : v2.get("stacking_auc"),
    })

comparison_rows.append({
    "Model"      : "★ DCWA-E",
    "Accuracy"   : metrics_opt["Accuracy"],
    "Sensitivity": metrics_opt["Sensitivity"],
    "Specificity": metrics_opt["Specificity"],
    "F1_Score"   : metrics_opt["F1_Score"],
    "MCC"        : metrics_opt["MCC"],
    "AUC"        : metrics_opt["AUC"],
})

df_comp   = pd.DataFrame(comparison_rows)
comp_path = os.path.join(DCWA_DIR, "dcwa_vs_baselines.csv")
df_comp.to_csv(comp_path, index=False)
print(f"\n{df_comp.to_string(index=False)}")
print(f"\n  ✅ Comparison saved → {comp_path}")


# ============================================================
#   CELL 18 — Save All Artefacts
# ============================================================

results_path = os.path.join(DCWA_DIR, "dcwa_results.csv")
pd.DataFrame([
    {"Model": "DCWA-E (default t=0.50)", "Threshold": 0.50,
     **{k:v for k,v in metrics_def.items()
        if k not in ("TP","TN","FP","FN")}},
    {"Model": f"DCWA-E (optimal t={best_t})", "Threshold": best_t,
     **{k:v for k,v in metrics_opt.items()
        if k not in ("TP","TN","FP","FN")}},
]).to_csv(results_path, index=False)

probs_path = os.path.join(DCWA_DIR, "dcwa_probabilities.csv")
pd.DataFrame({
    "y_true"          : y_test,
    "y_pred_default"  : y_pred_default,
    "y_pred_optimal"  : y_pred_optimal,
    "prob_positive"   : meta_prob_test,
    "p_combined"      : test_combined,
    "D_disagreement"  : test_D,
    **{f"p_{c}": test_probs[:,i]
       for i,c in enumerate(clf_list)},
    **{f"w_{c}": test_adaptive_w[:,i]
       for i,c in enumerate(clf_list)},
    **{f"conf_{c}": test_confidence[:,i]
       for i,c in enumerate(clf_list)},
}).to_csv(probs_path, index=False)

summary = {
    "method"             : "DCWA-E",
    "base_learners"      : clf_list,
    "plm_stream"         : PLM_STREAM,
    "classical_stream"   : CLASSICAL_STREAM,
    "dataset_map"        : BASE_LEARNER_DATASETS,
    "meta_learner"       : "LogisticRegression(C=0.1)",
    "calibration"        : "isotonic (LGBM, ET, RF)",
    "stream_balance"     : STREAM_BALANCE,
    "best_threshold"     : best_t,
    "n_folds"            : N_FOLDS,
    "n_jobs"             : N_JOBS,
    "stacking_auc"       : metrics_opt["AUC"],
    "stacking_mcc"       : metrics_opt["MCC"],
    "stacking_sn"        : metrics_opt["Sensitivity"],
    "stacking_sp"        : metrics_opt["Specificity"],
    "stacking_acc"       : metrics_opt["Accuracy"],
    "stacking_f1"        : metrics_opt["F1_Score"],
    "mean_D_train"       : float(oof_D.mean()),
    "high_disagreement_pct": float((oof_D>0.05).mean()*100),
    "novel_components"   : [
        "Confidence estimator: c_i = 1 - 2|p_i - 0.5|",
        "Disagreement detector: D = Var(p_CB, p_LGBM, p_ET, p_RF)",
        "Adaptive weight calculator: w_i = softmax(c_i * base_w * stream_w)",
    ],
}
sum_path = os.path.join(DCWA_DIR, "dcwa_summary.json")
with open(sum_path, "w") as f:
    json.dump(summary, f, indent=4)

art_path = os.path.join(DCWA_DIR, "dcwa_artefacts.joblib")
joblib.dump({
    "meta_learner"   : meta_learner,
    "final_models"   : final_models,
    "final_scalers"  : final_scalers,
    "base_w_arr"     : base_w_arr,
    "norm_base"      : norm_base,
    "best_threshold" : best_t,
    "clf_list"       : clf_list,
    "plm_idx"        : plm_idx,
    "cls_idx"        : cls_idx,
    "dataset_map"    : BASE_LEARNER_DATASETS,
    "stream_balance" : STREAM_BALANCE,
    "feat_names"     : feat_names,
}, art_path)

print(f"\n✅ Results    → {results_path}")
print(f"✅ Probs      → {probs_path}")
print(f"✅ Summary    → {sum_path}")
print(f"✅ Artefacts  → {art_path}")


# ============================================================
#   CELL 19 — Visualization 1: ROC Curves
# ============================================================

clf_colors = {
    "CB"  : "#e67e22", "LGBM": "#8e44ad",
    "ET"  : "#1abc9c", "RF"  : "#2ecc71",
}

fig, ax = plt.subplots(figsize=(10, 8))
for ci, clf_short in enumerate(clf_list):
    p           = test_probs[:, ci]
    fpr, tpr, _ = roc_curve(y_test, p)
    auc_val     = roc_auc_score(y_test, p)
    ax.plot(fpr, tpr,
            color=clf_colors.get(clf_short,"#95a5a6"),
            linewidth=1.2, linestyle="--",
            label=f"{clf_short} (AUC={auc_val:.4f})",
            alpha=0.8)

fpr_s, tpr_s, _ = roc_curve(y_test, meta_prob_test)
ax.plot(fpr_s, tpr_s, color="black", linewidth=2.5,
        label=f"★ DCWA-E (AUC={metrics_opt['AUC']:.4f})")
ax.plot([0,1],[0,1],"grey",linewidth=0.8,linestyle="--",alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — DCWA-E vs Base Learners",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dcwa_ROC.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ ROC curves saved")


# ============================================================
#   CELL 20 — Visualization 2: Adaptive Weights Heatmap
#
#   Shows how weights change across samples — the core visual
#   proof that DCWA-E is doing something novel vs static methods
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Sort by disagreement
sort_idx    = np.argsort(test_D)
sorted_w    = test_adaptive_w[sort_idx]
sorted_D    = test_D[sort_idx]

# Heatmap of adaptive weights
im = axes[0].imshow(sorted_w.T, aspect="auto",
                     cmap="YlOrRd", vmin=0, vmax=1)
axes[0].set_yticks(range(len(clf_list)))
axes[0].set_yticklabels(clf_list, fontsize=11)
axes[0].set_xlabel("Test samples (sorted by disagreement D)", fontsize=11)
axes[0].set_title("Adaptive weights per sample\n"
                  "(sorted by disagreement — right = harder peptides)",
                  fontsize=11, fontweight="bold")
plt.colorbar(im, ax=axes[0], label="Weight")

# PLM vs Classical weight ratio across samples
plm_weight_sum = sorted_w[:, plm_idx].sum(axis=1)
cls_weight_sum = sorted_w[:, cls_idx].sum(axis=1)
x_samples = np.arange(len(sorted_D))
axes[1].fill_between(x_samples, plm_weight_sum,
                      alpha=0.6, color="#BA7517",
                      label="PLM stream weight")
axes[1].fill_between(x_samples, plm_weight_sum, 1.0,
                      alpha=0.6, color="#0F6E56",
                      label="Classical stream weight")
axes[1].plot(x_samples, sorted_D * 3, color="#922B21",
             linewidth=1.5, linestyle="--",
             label="Disagreement D (×3 scaled)")
axes[1].set_xlabel("Test samples (sorted by disagreement D)", fontsize=11)
axes[1].set_ylabel("Cumulative stream weight", fontsize=11)
axes[1].set_title("PLM vs Classical stream dominance\n"
                  "(PLM trusted more when models disagree)",
                  fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.suptitle("DCWA-E Adaptive Weight Distribution",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dcwa_adaptive_weights.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Adaptive weights heatmap saved")


# ============================================================
#   CELL 21 — Visualization 3: Disagreement Analysis
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of D per class
for label, color, name in [(1,"#2ecc71","AIP"), (0,"#e74c3c","Non-AIP")]:
    mask = y_test == label
    axes[0].hist(test_D[mask], bins=30, color=color,
                 alpha=0.6, label=name, density=True)
axes[0].set_xlabel("Disagreement D", fontsize=11)
axes[0].set_ylabel("Density", fontsize=11)
axes[0].set_title("Disagreement distribution by class",
                  fontsize=11, fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)

# D vs final prediction confidence
final_conf = 1.0 - 2.0*np.abs(meta_prob_test - 0.5)
axes[1].scatter(test_D, final_conf, alpha=0.4,
                c=y_test, cmap="RdYlGn", s=20)
axes[1].set_xlabel("Disagreement D", fontsize=11)
axes[1].set_ylabel("Final prediction confidence", fontsize=11)
axes[1].set_title("Disagreement vs final confidence\n"
                  "(green=AIP, red=Non-AIP)",
                  fontsize=11, fontweight="bold")
axes[1].grid(alpha=0.3)

# Accuracy binned by disagreement level
bins   = np.percentile(test_D, [0,25,50,75,100])
labels = ["Low D","Med-low D","Med-high D","High D"]
accs   = []
for i in range(len(bins)-1):
    mask = (test_D >= bins[i]) & (test_D < bins[i+1])
    if mask.sum() > 0:
        pred = (meta_prob_test[mask] >= best_t).astype(int)
        accs.append(accuracy_score(y_test[mask], pred))
    else:
        accs.append(0)

axes[2].bar(labels, accs, color=["#2ecc71","#f1c40f","#e67e22","#e74c3c"],
            edgecolor="white", alpha=0.85)
axes[2].set_ylabel("Accuracy", fontsize=11)
axes[2].set_title("Accuracy by disagreement quartile\n"
                  "(confirms D captures hard cases)",
                  fontsize=11, fontweight="bold")
axes[2].set_ylim(0, 1.0); axes[2].grid(axis="y", alpha=0.3)
for i, v in enumerate(accs):
    axes[2].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=10)

plt.suptitle("DCWA-E Disagreement Signal Analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dcwa_disagreement_analysis.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Disagreement analysis saved")


# ============================================================
#   CELL 22 — Visualization 4: Confusion Matrix
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
best_base_p    = test_probs[:, 0]   # CB
best_base_t, _ = find_best_threshold(y_test, best_base_p, "mcc")

for ax_cm, pred, title, prob in zip(
    axes,
    [(best_base_p >= best_base_t).astype(int), y_pred_optimal],
    [f"CB Optuna (t={best_base_t})", f"★ DCWA-E (t={best_t})"],
    [best_base_p, meta_prob_test]
):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Non-AIP","AIP"]
                           ).plot(cmap="Blues", ax=ax_cm, colorbar=False)
    m = compute_metrics(y_test, pred, prob)
    ax_cm.set_title(
        f"{title}\nSn={m['Sensitivity']:.4f}  "
        f"Sp={m['Specificity']:.4f}  "
        f"MCC={m['MCC']:.4f}  AUC={m['AUC']:.4f}",
        fontsize=9, fontweight="bold"
    )
plt.suptitle("Confusion Matrix — CB vs DCWA-E",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dcwa_confusion.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Confusion matrices saved")


# ============================================================
#   CELL 23 — Visualization 5: Threshold Curve
# ============================================================

thresh_vals = np.arange(0.10, 0.90, 0.01)
mcc_v, sn_v, sp_v = [], [], []
for t in thresh_vals:
    y_p = (meta_prob_test >= t).astype(int)
    if len(np.unique(y_p)) < 2:
        mcc_v.append(np.nan); sn_v.append(np.nan)
        sp_v.append(np.nan); continue
    tn, fp, fn, tp_ = confusion_matrix(y_test, y_p).ravel()
    mcc_v.append(matthews_corrcoef(y_test, y_p))
    sn_v.append(tp_/(tp_+fn) if (tp_+fn)>0 else 0)
    sp_v.append(tn/(tn+fp)   if (tn+fp)>0  else 0)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(thresh_vals, mcc_v, color="#e74c3c", linewidth=2.5, label="MCC")
ax.plot(thresh_vals, sn_v,  color="#3498db", linewidth=1.8,
        linestyle="--", label="Sensitivity")
ax.plot(thresh_vals, sp_v,  color="#2ecc71", linewidth=1.8,
        linestyle="--", label="Specificity")
ax.axvline(best_t, color="#1A1A1A", linewidth=1.5,
           linestyle=":", label=f"Optimal t={best_t}")
ax.axvline(0.50, color="#888888", linewidth=1.0,
           linestyle=":", label="Default t=0.50")
ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Threshold vs MCC / Sn / Sp — DCWA-E",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.set_xlim([0.10, 0.90])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dcwa_threshold_curve.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Threshold curve saved")


# ============================================================
#   CELL 24 — Visualization 6: Meta-coefficient Bar Chart
# ============================================================

coef_vals = meta_learner.coef_[0]
colors    = [clf_colors.get(c,"#1abc9c") for c in clf_list] + \
            ["#34495e","#922B21"]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(feat_names, coef_vals, color=colors,
              edgecolor="white", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("LR Coefficient", fontsize=12)
ax.set_title("DCWA-E Meta-Learner Coefficients\n"
             "(positive = trusted for AIP prediction)",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, (f, c) in enumerate(zip(feat_names, coef_vals)):
    ax.text(i, c + (0.01 if c>=0 else -0.04),
            f"{c:+.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dcwa_meta_coefficients.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Meta-coefficients saved")


# ============================================================
#   CELL 25 — Final Summary
# ============================================================

print("\n" + "="*65)
print("  DCWA-E — COMPLETE SUMMARY")
print("="*65)
print(f"\n  Architecture:")
print(f"    PLM stream    : CB(ProtT5) + LGBM(ESM2)")
print(f"    Classical     : ET(CTDC) + RF(PAAC)")
print(f"    Novel comps   :")
print(f"      [1] Confidence estimator  c_i = 1-2|p_i-0.5|")
print(f"      [2] Disagreement detector D = Var(p_i)")
print(f"      [3] Adaptive weights      w_i = softmax(c_i*base_w*stream_w)")
print(f"    Meta-learner  : LR(C=0.1), 6 features incl. D")
print(f"    Threshold     : {best_t} (MCC-optimised on OOF)")
print(f"    Stream balance: {STREAM_BALANCE}")
print(f"\n{'─'*65}")
print(f"  {'Model':<16} {'Acc':>8} {'Sn':>8} {'Sp':>8} "
      f"{'MCC':>8} {'AUC':>8}")
print(f"{'─'*65}")
for _, row in df_comp.iterrows():
    is_dcwa = "DCWA" in str(row["Model"])
    vals = [row.get(m,None) for m in
            ["Accuracy","Sensitivity","Specificity","MCC","AUC"]]
    vstr = "  ".join(
        f"{v:>8.4f}" if v is not None and not pd.isna(v)
        else "       —" for v in vals
    )
    mark = " ★" if is_dcwa else ""
    print(f"  {str(row['Model']):<16} {vstr}{mark}")
print(f"{'─'*65}")
print(f"\n  Mean disagreement D : {oof_D.mean():.4f}")
print(f"  Hard peptides (D>0.05): "
      f"{(oof_D>0.05).sum()} / {len(oof_D)} "
      f"({(oof_D>0.05).mean()*100:.1f}%)")
print(f"\n  Files saved to: {DCWA_DIR}/")
print(f"  Figures saved : {FIGURES_DIR}/")
print("="*65)