"""
stacking_ensemble.py
====================
AIP Prediction — Stacking Ensemble
Base learners : CB (ProtT5), XGB (ESM2), LGBM (ProtT5),
                RF (ESM2), LR (DPC)
Meta-learner  : LightGBM (depth=3, 50 trees)
Objective     : AUC-MCC composite + threshold optimisation

Usage:
    python stacking_ensemble.py
"""

# ============================================================
#   CELL 1 — Mount Google Drive (Colab only)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================
#   CELL 2 — Install & Import Libraries
# ============================================================
# !pip install optuna catboost lightgbm xgboost imbalanced-learn -q

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from catboost           import CatBoostClassifier, Pool
from xgboost            import XGBClassifier
from lightgbm           import LGBMClassifier
from sklearn.ensemble   import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection  import (
    train_test_split, StratifiedKFold
)
from sklearn.preprocessing    import StandardScaler
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay
)

print("✅ Libraries loaded")


# ============================================================
#   CELL 3 — Configuration
# ============================================================

# ── Paths ────────────────────────────────────────────────────
# Google Colab:
# FEATURE_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/data/features"
# BASE_RESULTS= "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models"
# STACK_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/stacking"
# FIGURES_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/figures/stacking"

# Local:
FEATURE_DIR = "../../../data/features"
BASE_RESULTS= "../../../results/models"
STACK_DIR   = "../../../results/models/stacking"
FIGURES_DIR = "../../../results/figures/stacking"

os.makedirs(STACK_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Base learner dataset assignments ─────────────────────────
#
#  Each base learner uses the feature dataset that gave it
#  the highest Optuna AUC. You can update these if your
#  best datasets differ.
#
BASE_LEARNER_DATASETS = {
    "CB"  : "ProtT5_features.csv",    # AUC 0.8468 on ProtT5
    "XGB" : "ESM2_features.csv",      # AUC 0.8317 on ESM2
    "LGBM": "ProtT5_features.csv",    # AUC 0.8304 on ProtT5
    "RF"  : "ESM2_features.csv",      # AUC 0.8296 on ESM2
    "LR"  : "DPC_features.csv",       # AUC 0.7320 on DPC
}

# ── Stacking settings ────────────────────────────────────────
N_FOLDS      = 5       # folds for OOF generation
RANDOM_STATE = 42
TEST_SIZE    = 0.30

# ── AUC-MCC composite weight ─────────────────────────────────
ALPHA = 0.5            # 0.5 = equal AUC + MCC weight

# ── pos_weight per base learner (from Optuna best params) ────
#  Update these from your saved JSON best-params files.
#  Using sensible defaults if not yet known.
POS_WEIGHTS = {
    "CB"  : 2.5,
    "XGB" : 3.0,       # higher — XGB is the Sn specialist
    "LGBM": 2.0,
    "RF"  : 1.8,
    "LR"  : 2.0,
}

print("✅ Configuration loaded")
print(f"   Base learners   : {list(BASE_LEARNER_DATASETS.keys())}")
print(f"   CV folds        : {N_FOLDS}")
print(f"   Train/Test      : {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
print(f"   Composite alpha : {ALPHA}")


# ============================================================
#   CELL 4 — Metric Helpers
# ============================================================

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
        if len(np.unique(y_p)) < 2:
            continue
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
    X         = df.iloc[:, :-1].values
    y         = df.iloc[:, -1].values.astype(int)
    return X, y


# ============================================================
#   CELL 5 — Build Base Learner Instances
#
#  Hyperparameters below are reasonable defaults.
#  For best results, replace with your Optuna best params
#  loaded from the JSON files saved by CB/XGB/LGBM/RF/LR
#  Optuna scripts.
# ============================================================

def load_optuna_params(clf_short, dataset_name):
    """
    Try to load best params from saved JSON.
    Falls back to defaults if file not found.
    """
    param_paths = {
        "CB"  : os.path.join(BASE_RESULTS, "cb_optuna",
                             "best_params",
                             f"{dataset_name}_CB_best_params.json"),
        "XGB" : os.path.join(BASE_RESULTS, "xgb_optuna",
                             "best_params",
                             f"{dataset_name}_XGB_best_params.json"),
        "LGBM": os.path.join(BASE_RESULTS, "lgbm_optuna",
                             "best_params",
                             f"{dataset_name}_LGBM_best_params.json"),
        "RF"  : os.path.join(BASE_RESULTS, "rf_optuna",
                             "best_params",
                             f"{dataset_name}_RF_best_params.json"),
        "LR"  : os.path.join(BASE_RESULTS, "lr_optuna",
                             "best_params",
                             f"{dataset_name}_LR_best_params.json"),
    }
    path = param_paths.get(clf_short, "")
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"     ✅ Loaded params from {os.path.basename(path)}")
        return params
    print(f"     ⚠️  Params not found for {clf_short}/{dataset_name}"
          f" — using defaults")
    return None


def build_base_learner(clf_short, dataset_name):
    """
    Build each base learner using saved Optuna params if available,
    or sensible defaults if not.
    """
    params = load_optuna_params(clf_short, dataset_name)
    pw     = POS_WEIGHTS.get(clf_short, 2.0)

    # ── CatBoost ─────────────────────────────────────────────
    if clf_short == "CB":
        if params:
            cb_params = {
                "iterations"           : params.get("iterations", 300),
                "depth"                : params.get("depth", 6),
                "learning_rate"        : params.get("learning_rate", 0.05),
                "l2_leaf_reg"          : params.get("l2_leaf_reg", 3.0),
                "random_strength"      : params.get("random_strength", 1.0),
                "bagging_temperature"  : params.get("bagging_temperature", 0.5),
                "border_count"         : params.get("border_count", 128),
                "grow_policy"          : params.get("grow_policy", "SymmetricTree"),
                "min_data_in_leaf"     : params.get("min_data_in_leaf", 5),
                "leaf_estimation_method": params.get("leaf_estimation_method", "Newton"),
            }
        else:
            cb_params = {
                "iterations": 300, "depth": 6,
                "learning_rate": 0.05, "l2_leaf_reg": 3.0,
                "random_strength": 1.0, "bagging_temperature": 0.5,
                "border_count": 128, "grow_policy": "SymmetricTree",
                "min_data_in_leaf": 5, "leaf_estimation_method": "Newton",
            }
        cb_params.update({
            "class_weights"      : [1.0, pw],
            "eval_metric"        : "AUC",
            "early_stopping_rounds": 20,
            "task_type"          : "CPU",
            "verbose"            : 0,
            "allow_writing_files": False,
            "random_seed"        : RANDOM_STATE,
        })
        return CatBoostClassifier(**cb_params)

    # ── XGBoost ──────────────────────────────────────────────
    elif clf_short == "XGB":
        if params:
            return XGBClassifier(
                n_estimators      = params.get("n_estimators", 300),
                max_depth         = params.get("max_depth", 6),
                learning_rate     = params.get("learning_rate", 0.05),
                subsample         = params.get("subsample", 0.8),
                colsample_bytree  = params.get("colsample_bytree", 0.8),
                reg_alpha         = params.get("reg_alpha", 0.1),
                reg_lambda        = params.get("reg_lambda", 1.0),
                scale_pos_weight  = pw,
                use_label_encoder = False,
                eval_metric       = "auc",
                tree_method       = "hist",
                random_state      = RANDOM_STATE,
                verbosity         = 0,
            )
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw, use_label_encoder=False,
            eval_metric="auc", tree_method="hist",
            random_state=RANDOM_STATE, verbosity=0,
        )

    # ── LightGBM ─────────────────────────────────────────────
    elif clf_short == "LGBM":
        if params:
            return LGBMClassifier(
                n_estimators     = params.get("n_estimators", 300),
                max_depth        = params.get("max_depth", 6),
                learning_rate    = params.get("learning_rate", 0.05),
                num_leaves       = params.get("num_leaves", 31),
                subsample        = params.get("subsample", 0.8),
                colsample_bytree = params.get("colsample_bytree", 0.8),
                reg_alpha        = params.get("reg_alpha", 0.1),
                reg_lambda       = params.get("reg_lambda", 1.0),
                scale_pos_weight = pw,
                n_jobs           = -1,
                random_state     = RANDOM_STATE,
                verbosity        = -1,
            )
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw, n_jobs=-1,
            random_state=RANDOM_STATE, verbosity=-1,
        )

    # ── Random Forest ─────────────────────────────────────────
    elif clf_short == "RF":
        if params:
            return RandomForestClassifier(
                n_estimators  = params.get("n_estimators", 300),
                max_depth     = params.get("max_depth", None),
                min_samples_split = params.get("min_samples_split", 2),
                min_samples_leaf  = params.get("min_samples_leaf", 1),
                max_features  = params.get("max_features", "sqrt"),
                class_weight  = "balanced",
                n_jobs        = -1,
                random_state  = RANDOM_STATE,
            )
        return RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    # ── Logistic Regression ───────────────────────────────────
    elif clf_short == "LR":
        if params:
            return LogisticRegression(
                C            = params.get("C", 1.0),
                penalty      = params.get("penalty", "l2"),
                solver       = params.get("solver", "lbfgs"),
                class_weight = "balanced",
                max_iter     = 2000,
                random_state = RANDOM_STATE,
            )
        return LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=2000,
            random_state=RANDOM_STATE,
        )


# ============================================================
#   CELL 6 — Load All Datasets for Base Learners
# ============================================================

print("\n" + "="*65)
print("  Loading feature datasets for base learners...")
print("="*65)

datasets      = {}
scalers       = {}
unique_files  = set(BASE_LEARNER_DATASETS.values())

for csv_file in unique_files:
    csv_path = os.path.join(FEATURE_DIR, csv_file)
    if not os.path.exists(csv_path):
        print(f"  ❌ File not found: {csv_path}")
        continue
    X, y = load_feature_csv(csv_path)
    datasets[csv_file] = (X, y)
    print(f"  ✅ {csv_file:<30} shape: {X.shape}")

# All classifiers share same y — use one to split indices
csv0     = list(unique_files)[0]
_, y_all = datasets[csv0]

# ── Stratified train/test split ───────────────────────────────
train_idx, test_idx = train_test_split(
    np.arange(len(y_all)),
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y_all
)

print(f"\n  Train samples : {len(train_idx)}")
print(f"  Test  samples : {len(test_idx)}")
print(f"  Pos (AIP)     : {y_all.sum()}")
print(f"  Neg (non-AIP) : {(y_all==0).sum()}")

# ── Scale and split each dataset ─────────────────────────────
X_trains, X_tests, scalers = {}, {}, {}

for csv_file, (X, y) in datasets.items():
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X[train_idx])
    X_te    = scaler.transform(X[test_idx])
    X_trains[csv_file] = X_tr
    X_tests[csv_file]  = X_te
    scalers[csv_file]  = scaler

y_train = y_all[train_idx]
y_test  = y_all[test_idx]


# ============================================================
#   CELL 7 — Generate Out-of-Fold Probabilities
#
#  For each base learner:
#    1. Split training data into N_FOLDS folds
#    2. Train on (N-1) folds, predict on 1 fold
#    3. Collect all fold predictions → OOF probabilities
#    4. Final model trained on full training set
#
#  OOF probabilities are unbiased — the model never saw
#  the sample it is predicting → safe for meta-learner training
# ============================================================

print("\n" + "="*65)
print("  Generating Out-of-Fold Probabilities")
print(f"  {N_FOLDS}-fold Stratified CV per base learner")
print("="*65)

cv              = StratifiedKFold(
    n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
)

oof_probs       = np.zeros((len(y_train), len(BASE_LEARNER_DATASETS)))
test_probs      = np.zeros((len(y_test),  len(BASE_LEARNER_DATASETS)))
final_models    = {}
final_scalers   = {}
clf_aucs        = {}

for clf_idx, (clf_short, csv_file) in enumerate(
    BASE_LEARNER_DATASETS.items()
):
    print(f"\n  [{clf_idx+1}/5]  {clf_short}  ←  {csv_file}")

    X_tr = X_trains[csv_file]
    X_te = X_tests[csv_file]

    oof_fold   = np.zeros(len(y_train))
    test_folds = np.zeros((len(y_test), N_FOLDS))

    fold_aucs = []

    for fold, (f_train, f_val) in enumerate(
        cv.split(X_tr, y_train)
    ):
        X_f_tr, X_f_val = X_tr[f_train], X_tr[f_val]
        y_f_tr, y_f_val = y_train[f_train], y_train[f_val]

        clf = build_base_learner(clf_short, csv_file.split("_")[0])

        # ── CatBoost uses Pool objects ────────────────────────
        if clf_short == "CB":
            train_pool = Pool(X_f_tr, y_f_tr)
            val_pool   = Pool(X_f_val, y_f_val)
            clf.fit(train_pool, eval_set=val_pool,
                    use_best_model=True)
        else:
            clf.fit(X_f_tr, y_f_tr)

        oof_fold[f_val]     = clf.predict_proba(X_f_val)[:, 1]
        test_folds[:, fold] = clf.predict_proba(X_te)[:, 1]

        fold_auc = roc_auc_score(y_f_val, oof_fold[f_val])
        fold_aucs.append(fold_auc)
        print(f"     Fold {fold+1}/{N_FOLDS}  AUC = {fold_auc:.4f}")

    # ── Average test predictions across folds ────────────────
    oof_probs[:, clf_idx]  = oof_fold
    test_probs[:, clf_idx] = test_folds.mean(axis=1)

    mean_oof_auc = roc_auc_score(y_train, oof_fold)
    clf_aucs[clf_short] = mean_oof_auc
    print(f"     Mean OOF AUC : {mean_oof_auc:.4f}")

    # ── Train final model on full training set ────────────────
    print(f"     Training final {clf_short} on full train set...")
    final_clf = build_base_learner(clf_short,
                                    csv_file.split("_")[0])
    if clf_short == "CB":
        train_pool = Pool(X_tr, y_train)
        eval_pool  = Pool(X_te, y_test)
        final_clf.fit(train_pool, eval_set=eval_pool,
                      use_best_model=True)
    else:
        final_clf.fit(X_tr, y_train)

    final_models[clf_short]  = final_clf
    final_scalers[clf_short] = scalers[csv_file]

    # ── Save base model ───────────────────────────────────────
    model_path = os.path.join(
        STACK_DIR, f"base_{clf_short}.joblib"
    )
    joblib.dump(final_clf, model_path)
    print(f"     ✅ Saved → {model_path}")

print(f"\n✅ OOF generation complete")
print(f"   OOF matrix shape  : {oof_probs.shape}")
print(f"   Test matrix shape : {test_probs.shape}")
print(f"\n   Base learner OOF AUCs:")
for clf_short, auc in clf_aucs.items():
    print(f"     {clf_short:<6} : {auc:.4f}")


# ============================================================
#   CELL 8 — Compute AUC-MCC Composite Weights
#
#  W_i = ALPHA × AUC_i + (1 - ALPHA) × MCC_norm_i
#
#  These weights represent how much each base learner's
#  prediction contributes to the weighted ensemble probability,
#  which is also added as a 6th feature to the meta-learner.
# ============================================================

print("\n" + "="*65)
print("  Computing AUC-MCC Composite Weights")
print("="*65)

weights    = {}
mcc_scores = {}

for clf_idx, clf_short in enumerate(BASE_LEARNER_DATASETS):
    oof_pred      = (oof_probs[:, clf_idx] >= 0.5).astype(int)
    mcc           = matthews_corrcoef(y_train, oof_pred)
    mcc_norm      = (mcc + 1.0) / 2.0
    auc           = clf_aucs[clf_short]
    w             = ALPHA * auc + (1.0 - ALPHA) * mcc_norm
    weights[clf_short]    = w
    mcc_scores[clf_short] = mcc

# Normalise weights so they sum to 1
total_w = sum(weights.values())
norm_weights = {k: round(v / total_w, 4)
                for k, v in weights.items()}

print(f"\n  {'Clf':<6} {'OOF AUC':>9} {'OOF MCC':>9} "
      f"{'Raw W':>9} {'Norm W':>9}")
print(f"  {'─'*46}")
for clf_short in BASE_LEARNER_DATASETS:
    print(f"  {clf_short:<6} "
          f"{clf_aucs[clf_short]:>9.4f} "
          f"{mcc_scores[clf_short]:>9.4f} "
          f"{weights[clf_short]:>9.4f} "
          f"{norm_weights[clf_short]:>9.4f}")

# ── Weighted ensemble probability ────────────────────────────
w_array      = np.array([norm_weights[c]
                          for c in BASE_LEARNER_DATASETS])
oof_weighted = oof_probs  @ w_array
test_weighted= test_probs @ w_array

print(f"\n  Normalised weights : {norm_weights}")

# ── Add weighted probability as 6th meta-feature ─────────────
meta_train = np.column_stack([oof_probs,  oof_weighted])
meta_test  = np.column_stack([test_probs, test_weighted])

print(f"\n  Meta-train shape  : {meta_train.shape}")
print(f"  Meta-test  shape  : {meta_test.shape}")
print(f"  Columns: [p_CB, p_XGB, p_LGBM, p_RF, p_LR, p_weighted]")


# ============================================================
#   CELL 9 — Train Meta-Learner (LightGBM)
# ============================================================

print("\n" + "="*65)
print("  Training Meta-Learner (LightGBM)")
print("="*65)

meta_learner = LGBMClassifier(
    n_estimators      = 50,
    max_depth         = 3,       # shallow — avoid overfit on small set
    learning_rate     = 0.05,
    num_leaves        = 7,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = 1.5,     # class ratio 1:1.5
    n_jobs            = -1,
    random_state      = RANDOM_STATE,
    verbosity         = -1,
)

meta_learner.fit(meta_train, y_train)

# ── Meta-learner probabilities ────────────────────────────────
meta_prob_train = meta_learner.predict_proba(meta_train)[:, 1]
meta_prob_test  = meta_learner.predict_proba(meta_test) [:, 1]

train_auc = roc_auc_score(y_train, meta_prob_train)
test_auc  = roc_auc_score(y_test,  meta_prob_test)
print(f"\n  Meta-learner train AUC : {train_auc:.4f}")
print(f"  Meta-learner test  AUC : {test_auc:.4f}")

# ── Save meta-learner ─────────────────────────────────────────
meta_path = os.path.join(STACK_DIR, "meta_learner_lgbm.joblib")
joblib.dump(meta_learner, meta_path)
print(f"\n  ✅ Meta-learner saved → {meta_path}")


# ============================================================
#   CELL 10 — Threshold Optimisation on Train Set
#
#  Find the threshold that maximises MCC on the training
#  OOF predictions. This avoids test set leakage.
# ============================================================

print("\n" + "="*65)
print("  Threshold Optimisation (MCC on OOF predictions)")
print("="*65)

best_t, best_mcc_val = find_best_threshold(
    y_train, meta_prob_train, metric="mcc"
)

print(f"  Optimal threshold : {best_t}  "
      f"(train MCC = {best_mcc_val:.4f})")
print(f"  Default threshold : 0.50")

# ── Final predictions ─────────────────────────────────────────
y_pred_default = (meta_prob_test >= 0.50).astype(int)
y_pred_optimal = (meta_prob_test >= best_t).astype(int)


# ============================================================
#   CELL 11 — Final Metrics
# ============================================================

print("\n" + "="*65)
print("  STACKING ENSEMBLE — FINAL RESULTS")
print("="*65)

metrics_def = compute_metrics(y_test, y_pred_default, meta_prob_test)
metrics_opt = compute_metrics(y_test, y_pred_optimal, meta_prob_test)

print(f"\n  {'Metric':<14} {'Default (0.50)':>15} "
      f"{'Optimal ({:.2f})'.format(best_t):>18}  {'Δ':>8}")
print(f"  {'─'*58}")
for m in ["Accuracy","Sensitivity","Specificity","F1_Score","MCC","AUC"]:
    d     = metrics_def[m]
    o     = metrics_opt[m]
    delta = round(o - d, 4)
    sign  = "+" if delta >= 0 else ""
    print(f"  {m:<14} {d:>15.4f} {o:>18.4f}  {sign}{delta:>7.4f}")

print(f"\n  TP={metrics_opt['TP']}  TN={metrics_opt['TN']}  "
      f"FP={metrics_opt['FP']}  FN={metrics_opt['FN']}")

# ── Save results ──────────────────────────────────────────────
results_df = pd.DataFrame([
    {
        "Model"         : "Stacking (default t=0.50)",
        "Threshold"     : 0.50,
        **{k: v for k, v in metrics_def.items()
           if k not in ("TP","TN","FP","FN")}
    },
    {
        "Model"         : f"Stacking (optimal t={best_t})",
        "Threshold"     : best_t,
        **{k: v for k, v in metrics_opt.items()
           if k not in ("TP","TN","FP","FN")}
    },
])
results_path = os.path.join(STACK_DIR, "stacking_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\n  ✅ Results saved → {results_path}")

# ── Save probabilities ────────────────────────────────────────
probs_df = pd.DataFrame({
    "y_true"         : y_test,
    "y_pred_default" : y_pred_default,
    "y_pred_optimal" : y_pred_optimal,
    "prob_positive"  : meta_prob_test,
    "prob_negative"  : 1 - meta_prob_test,
    **{f"p_{c}": test_probs[:, i]
       for i, c in enumerate(BASE_LEARNER_DATASETS)},
    "p_weighted"     : test_weighted,
})
probs_path = os.path.join(STACK_DIR, "stacking_probabilities.csv")
probs_df.to_csv(probs_path, index=False)
print(f"  ✅ Probs  saved → {probs_path}")


# ============================================================
#   CELL 12 — Compare with Individual Base Learners
# ============================================================

print("\n" + "="*65)
print("  COMPARISON — Stack vs Individual Base Learners")
print("="*65)

individual_metrics = {}
for clf_idx, clf_short in enumerate(BASE_LEARNER_DATASETS):
    p    = test_probs[:, clf_idx]
    t, _ = find_best_threshold(y_test, p, metric="mcc")
    pred = (p >= t).astype(int)
    m    = compute_metrics(y_test, pred, p)
    individual_metrics[clf_short] = m

comparison_rows = []
for clf_short, m in individual_metrics.items():
    comparison_rows.append({
        "Model"      : clf_short,
        "Threshold"  : "opt",
        "Accuracy"   : m["Accuracy"],
        "Sensitivity": m["Sensitivity"],
        "Specificity": m["Specificity"],
        "F1_Score"   : m["F1_Score"],
        "MCC"        : m["MCC"],
        "AUC"        : m["AUC"],
    })

comparison_rows.append({
    "Model"      : "★ Stacking",
    "Threshold"  : best_t,
    "Accuracy"   : metrics_opt["Accuracy"],
    "Sensitivity": metrics_opt["Sensitivity"],
    "Specificity": metrics_opt["Specificity"],
    "F1_Score"   : metrics_opt["F1_Score"],
    "MCC"        : metrics_opt["MCC"],
    "AUC"        : metrics_opt["AUC"],
})

df_comp = pd.DataFrame(comparison_rows)
print(f"\n  {df_comp.to_string(index=False)}")

comp_path = os.path.join(STACK_DIR, "stacking_vs_individual.csv")
df_comp.to_csv(comp_path, index=False)
print(f"\n  ✅ Comparison saved → {comp_path}")


# ============================================================
#   CELL 13 — Visualization 1: ROC Curves
#             Stack vs all base learners
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))

clf_colors = {
    "CB"  : "#e67e22", "XGB" : "#3498db",
    "LGBM": "#8e44ad", "RF"  : "#2ecc71",
    "LR"  : "#c0392b",
}

for clf_idx, clf_short in enumerate(BASE_LEARNER_DATASETS):
    p       = test_probs[:, clf_idx]
    fpr, tpr, _ = roc_curve(y_test, p)
    auc_val = roc_auc_score(y_test, p)
    ax.plot(fpr, tpr,
            color    = clf_colors.get(clf_short, "#95a5a6"),
            linewidth= 1.2,
            linestyle= "--",
            label    = f"{clf_short} (AUC={auc_val:.4f})",
            alpha    = 0.8)

# Stack ROC
fpr_s, tpr_s, _ = roc_curve(y_test, meta_prob_test)
ax.plot(fpr_s, tpr_s,
        color="black", linewidth=2.5,
        label=f"★ Stacking (AUC={metrics_opt['AUC']:.4f})")

ax.plot([0,1],[0,1],"grey",linewidth=0.8,linestyle="--",alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — Stacking Ensemble vs Base Learners",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_ROC_curves.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ ROC curves saved")


# ============================================================
#   CELL 14 — Visualization 2: Metrics Bar Chart
# ============================================================

metric_list = ["Accuracy","Sensitivity","Specificity",
               "F1_Score","MCC","AUC"]

fig, ax = plt.subplots(figsize=(14, 6))
x_     = np.arange(len(metric_list))
n_clf  = len(df_comp)
bar_w  = 0.8 / n_clf

clrs   = list(clf_colors.values()) + ["#1A1A1A"]

for i, row in df_comp.iterrows():
    offset = (i - n_clf / 2 + 0.5) * bar_w
    vals   = [row[m] for m in metric_list]
    is_stack = "Stacking" in str(row["Model"])
    ax.bar(x_ + offset, vals, bar_w,
           label    = row["Model"],
           color    = clrs[i] if i < len(clrs) else "#555555",
           alpha    = 1.0 if is_stack else 0.70,
           edgecolor= "white",
           linewidth= 1.5 if is_stack else 0.5)

ax.set_xticks(x_)
ax.set_xticklabels(metric_list, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_title("Stacking Ensemble vs Base Learners — All Metrics",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.5, color="grey", linestyle="--",
           linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_metrics_bar.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Metrics bar chart saved")


# ============================================================
#   CELL 15 — Visualization 3: Confusion Matrices
#             Best base learner (CB) vs Stacking side-by-side
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

best_base_p    = test_probs[:, 0]   # CB is index 0
best_base_t, _ = find_best_threshold(y_test, best_base_p, "mcc")
best_base_pred = (best_base_p >= best_base_t).astype(int)

for ax, pred, title in zip(
    axes,
    [best_base_pred, y_pred_optimal],
    [f"CB (Best Base Learner, t={best_base_t})",
     f"★ Stacking Ensemble (t={best_t})"]
):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-AIP","AIP"]
    ).plot(cmap="Blues", ax=ax, colorbar=False)
    m = compute_metrics(y_test, pred,
                        best_base_p if "Base" in title
                        else meta_prob_test)
    ax.set_title(
        f"{title}\n"
        f"Sn={m['Sensitivity']:.4f}  "
        f"Sp={m['Specificity']:.4f}  "
        f"MCC={m['MCC']:.4f}  "
        f"AUC={m['AUC']:.4f}",
        fontsize=9, fontweight="bold"
    )

plt.suptitle("Confusion Matrix Comparison",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_confusion_matrices.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Confusion matrices saved")


# ============================================================
#   CELL 16 — Visualization 4: OOF Probability Heatmap
#             Shows how each base learner votes per sample
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5))

df_oof = pd.DataFrame(
    oof_probs,
    columns=list(BASE_LEARNER_DATASETS.keys())
)
df_oof["Weighted"] = oof_weighted
df_oof["Label"]    = y_train

# Sort by weighted probability for cleaner visualisation
df_oof = df_oof.sort_values("Weighted").reset_index(drop=True)

sns.heatmap(
    df_oof[list(BASE_LEARNER_DATASETS.keys()) + ["Weighted"]].T,
    cmap       = "RdYlGn",
    vmin       = 0, vmax = 1,
    ax         = ax,
    cbar_kws   = {"label": "Predicted Probability (AIP)"},
    xticklabels= False,
)
ax.set_title(
    "OOF Probability Heatmap — Base Learner Agreement\n"
    "(sorted by weighted ensemble score, "
    "green=high AIP prob, red=low)",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Training Samples (sorted by ensemble score)",
              fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_oof_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ OOF heatmap saved")


# ============================================================
#   CELL 17 — Visualization 5: Meta-feature Importance
# ============================================================

feat_names = list(BASE_LEARNER_DATASETS.keys()) + ["Weighted"]
importances = meta_learner.feature_importances_

fig, ax = plt.subplots(figsize=(9, 4))
colors  = [clf_colors.get(c, "#1abc9c") for c in feat_names]
ax.bar(feat_names, importances, color=colors,
       edgecolor="white", alpha=0.85)
ax.set_ylabel("Feature Importance", fontsize=12)
ax.set_title("Meta-Learner Feature Importance\n"
             "(which base learner the meta-LightGBM trusts most)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, (f, imp) in enumerate(zip(feat_names, importances)):
    ax.text(i, imp + 0.2, f"{imp:.1f}",
            ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_meta_importance.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Meta-feature importance saved")


# ============================================================
#   CELL 18 — Visualization 6: Threshold vs MCC curve
# ============================================================

thresh_vals = np.arange(0.10, 0.90, 0.01)
mcc_vals    = []
sn_vals     = []
sp_vals     = []

for t in thresh_vals:
    y_p = (meta_prob_test >= t).astype(int)
    if len(np.unique(y_p)) < 2:
        mcc_vals.append(np.nan); sn_vals.append(np.nan)
        sp_vals.append(np.nan); continue
    tn, fp, fn, tp = confusion_matrix(y_test, y_p).ravel()
    mcc_vals.append(matthews_corrcoef(y_test, y_p))
    sn_vals.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    sp_vals.append(tn/(tn+fp) if (tn+fp)>0 else 0)

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(thresh_vals, mcc_vals, color="#e74c3c",
        linewidth=2.5, label="MCC")
ax.plot(thresh_vals, sn_vals,  color="#3498db",
        linewidth=1.8, linestyle="--", label="Sensitivity")
ax.plot(thresh_vals, sp_vals,  color="#2ecc71",
        linewidth=1.8, linestyle="--", label="Specificity")
ax.axvline(best_t, color="#1A1A1A", linewidth=1.5,
           linestyle=":", label=f"Optimal t={best_t}")
ax.axvline(0.50, color="#888888", linewidth=1.0,
           linestyle=":", label="Default t=0.50")
ax.set_xlabel("Classification Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Threshold vs MCC / Sensitivity / Specificity\n"
             "Stacking Ensemble (test set)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0.10, 0.90])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "stacking_threshold_curve.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Threshold curve saved")


# ============================================================
#   CELL 19 — Save All Stacking Artefacts
# ============================================================

artefacts = {
    "meta_learner"   : meta_learner,
    "final_models"   : final_models,
    "final_scalers"  : final_scalers,
    "norm_weights"   : norm_weights,
    "best_threshold" : best_t,
    "clf_order"      : list(BASE_LEARNER_DATASETS.keys()),
    "dataset_map"    : BASE_LEARNER_DATASETS,
}
artefacts_path = os.path.join(STACK_DIR, "stacking_artefacts.joblib")
joblib.dump(artefacts, artefacts_path)

summary = {
    "base_learners"      : list(BASE_LEARNER_DATASETS.keys()),
    "dataset_map"        : BASE_LEARNER_DATASETS,
    "norm_weights"       : norm_weights,
    "best_threshold"     : best_t,
    "composite_alpha"    : ALPHA,
    "n_folds"            : N_FOLDS,
    "stacking_auc"       : metrics_opt["AUC"],
    "stacking_mcc"       : metrics_opt["MCC"],
    "stacking_sn"        : metrics_opt["Sensitivity"],
    "stacking_sp"        : metrics_opt["Specificity"],
    "stacking_acc"       : metrics_opt["Accuracy"],
    "stacking_f1"        : metrics_opt["F1_Score"],
    "cb_auc"             : individual_metrics["CB"]["AUC"],
    "cb_mcc"             : individual_metrics["CB"]["MCC"],
    "improvement_auc"    : round(metrics_opt["AUC"]
                                 - individual_metrics["CB"]["AUC"], 4),
    "improvement_mcc"    : round(metrics_opt["MCC"]
                                 - individual_metrics["CB"]["MCC"], 4),
    "improvement_sn"     : round(metrics_opt["Sensitivity"]
                                 - individual_metrics["CB"]["Sensitivity"], 4),
}
summary_path = os.path.join(STACK_DIR, "stacking_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print(f"\n✅ Artefacts saved → {artefacts_path}")
print(f"✅ Summary  saved → {summary_path}")


# ============================================================
#   CELL 20 — Final Summary
# ============================================================

print("\n" + "="*65)
print("  STACKING ENSEMBLE — COMPLETE SUMMARY")
print("="*65)
print(f"\n  Architecture:")
print(f"    Level 0  : CB(ProtT5) + XGB(ESM2) + LGBM(ProtT5) "
      f"+ RF(ESM2) + LR(DPC)")
print(f"    Weights  : {norm_weights}")
print(f"    Level 1  : LightGBM (depth=3, 50 trees)")
print(f"    Threshold: {best_t} (MCC-optimised on OOF)")
print(f"\n{'─'*65}")
print(f"  {'Model':<22} {'Sn':>8} {'Sp':>8} "
      f"{'MCC':>8} {'AUC':>8}")
print(f"{'─'*65}")
for clf_short in BASE_LEARNER_DATASETS:
    m = individual_metrics[clf_short]
    print(f"  {clf_short:<22} "
          f"{m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f} "
          f"{m['MCC']:>8.4f} {m['AUC']:>8.4f}")
print(f"{'─'*65}")
print(f"  {'★ Stacking':<22} "
      f"{metrics_opt['Sensitivity']:>8.4f} "
      f"{metrics_opt['Specificity']:>8.4f} "
      f"{metrics_opt['MCC']:>8.4f} "
      f"{metrics_opt['AUC']:>8.4f}")
print(f"{'─'*65}")
print(f"\n  Improvement over best single model (CB):")
print(f"     AUC  : {summary['improvement_auc']:+.4f}")
print(f"     MCC  : {summary['improvement_mcc']:+.4f}")
print(f"     Sn   : {summary['improvement_sn']:+.4f}")
print(f"\n  Saved files:")
print(f"     {artefacts_path}")
print(f"     {summary_path}")
print(f"     {results_path}")
print(f"     {comp_path}")
print(f"     {probs_path}")
print(f"\n  Figures saved to: {FIGURES_DIR}/")
print("="*65)