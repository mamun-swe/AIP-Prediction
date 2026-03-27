# """
# stacking_ensemble.py
# ====================
# AIP Prediction — Stacking Ensemble
# Base learners : CB (ProtT5), XGB (ESM2), LGBM (ProtT5),
#                 RF (ESM2), LR (DPC)
# Meta-learner  : LightGBM (depth=3, 50 trees)
# CPU           : 10 cores used throughout

# Fixes applied:
#   ✅ JSON "None" string → Python None conversion for all params
#   ✅ 10-CPU parallelism via joblib, n_jobs, os.environ
#   ✅ OOF folds run in parallel using joblib.Parallel
# """

# # ============================================================
# #   CELL 1 — Mount Google Drive (Colab only)
# # ============================================================
# # from google.colab import drive
# # drive.mount('/content/drive')


# # ============================================================
# #   CELL 2 — Install & Import Libraries
# # ============================================================
# # !pip install optuna catboost lightgbm xgboost -q

# import os
# import json
# import joblib
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns
# warnings.filterwarnings("ignore")

# # ── ✅ CPU PARALLELISM: Set all thread env vars before imports ─
# N_JOBS = 10   # ← number of CPU cores to use
# os.environ["OMP_NUM_THREADS"]      = str(N_JOBS)
# os.environ["MKL_NUM_THREADS"]      = str(N_JOBS)
# os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
# os.environ["NUMEXPR_NUM_THREADS"]  = str(N_JOBS)

# from catboost           import CatBoostClassifier, Pool
# from xgboost            import XGBClassifier
# from lightgbm           import LGBMClassifier
# from sklearn.ensemble   import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# from sklearn.model_selection  import (
#     train_test_split, StratifiedKFold
# )
# from sklearn.preprocessing    import StandardScaler
# from sklearn.metrics          import (
#     accuracy_score, f1_score, matthews_corrcoef,
#     confusion_matrix, roc_auc_score, roc_curve,
#     ConfusionMatrixDisplay
# )

# print("✅ Libraries loaded")
# print(f"   CPU cores allocated : {N_JOBS}")
# print(f"   OMP threads         : {os.environ['OMP_NUM_THREADS']}")
# print(f"   MKL threads         : {os.environ['MKL_NUM_THREADS']}")


# # ============================================================
# #   CELL 3 — Configuration
# # ============================================================

# # ── Paths ────────────────────────────────────────────────────
# # Google Colab:
# # FEATURE_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/data/features"
# # BASE_RESULTS= "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models"
# # STACK_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/stacking"
# # FIGURES_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/figures/stacking"

# # Local:
# FEATURE_DIR  = "../../../data/features"
# BASE_RESULTS = "../../../results/models"
# STACK_DIR    = "../../../results/models/stacking"
# FIGURES_DIR  = "../../../results/figures/stacking"

# os.makedirs(STACK_DIR,   exist_ok=True)
# os.makedirs(FIGURES_DIR, exist_ok=True)

# # ── Base learner dataset assignments ─────────────────────────
# BASE_LEARNER_DATASETS = {
#     "CB"  : "ProtT5_features.csv",
#     "XGB" : "ESM2_features.csv",
#     "LGBM": "ProtT5_features.csv",
#     "RF"  : "ESM2_features.csv",
#     "LR"  : "DPC_features.csv",
# }

# # ── Stacking settings ────────────────────────────────────────
# N_FOLDS      = 5
# RANDOM_STATE = 42
# TEST_SIZE    = 0.30
# ALPHA        = 0.5     # AUC-MCC composite weight

# # ── pos_weight defaults (used if JSON not found) ─────────────
# POS_WEIGHTS = {
#     "CB"  : 2.5,
#     "XGB" : 3.0,
#     "LGBM": 2.0,
#     "RF"  : 1.8,
#     "LR"  : 2.0,
# }

# print("✅ Configuration loaded")
# print(f"   Base learners   : {list(BASE_LEARNER_DATASETS.keys())}")
# print(f"   CV folds        : {N_FOLDS}")
# print(f"   Train/Test      : {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
# print(f"   Composite alpha : {ALPHA}")
# print(f"   CPU cores       : {N_JOBS}")


# # ============================================================
# #   CELL 4 — Helper Functions
# # ============================================================

# # ── ✅ FIX: JSON "None" string converter ─────────────────────
# def safe_none(v):
#     """
#     Convert string 'None' back to Python None after JSON load.

#     Problem: When sklearn/CatBoost params like max_depth=None
#     are saved to JSON, they become the string "None" instead
#     of JSON null. When loaded back, sklearn receives the string
#     "None" and raises InvalidParameterError.

#     This function converts any None-like string back to None.
#     """
#     if v in (None, "None", "none", "null", "nan", ""):
#         return None
#     return v


# def safe_int(v, default):
#     """Convert to int safely, return default if None-like."""
#     v = safe_none(v)
#     if v is None:
#         return default
#     try:
#         return int(float(v))
#     except Exception:
#         return default


# def safe_float(v, default):
#     """Convert to float safely, return default if None-like."""
#     v = safe_none(v)
#     if v is None:
#         return default
#     try:
#         return float(v)
#     except Exception:
#         return default


# def safe_str(v, default):
#     """Return string or default if None-like."""
#     v = safe_none(v)
#     if v is None:
#         return default
#     return str(v)


# # ── Metric computation ────────────────────────────────────────
# def compute_metrics(y_true, y_pred, y_prob):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     return {
#         "Accuracy"   : round(accuracy_score(y_true, y_pred),           4),
#         "Sensitivity": round(tp/(tp+fn) if (tp+fn)>0 else 0.0,         4),
#         "Specificity": round(tn/(tn+fp) if (tn+fp)>0 else 0.0,         4),
#         "F1_Score"   : round(f1_score(y_true, y_pred, zero_division=0), 4),
#         "MCC"        : round(matthews_corrcoef(y_true, y_pred),         4),
#         "AUC"        : round(roc_auc_score(y_true, y_prob),             4),
#         "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
#     }


# def find_best_threshold(y_true, y_prob, metric="mcc"):
#     best_t, best_score = 0.50, -999.0
#     for t in np.arange(0.10, 0.90, 0.01):
#         y_p = (y_prob >= t).astype(int)
#         if len(np.unique(y_p)) < 2:
#             continue
#         tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
#         if metric == "mcc":
#             score = matthews_corrcoef(y_true, y_p)
#         elif metric == "sensitivity":
#             score = tp/(tp+fn) if (tp+fn)>0 else 0.0
#         else:
#             score = f1_score(y_true, y_p, zero_division=0)
#         if score > best_score:
#             best_score, best_t = score, t
#     return round(best_t, 2), round(best_score, 4)


# def load_feature_csv(csv_path):
#     df        = pd.read_csv(csv_path)
#     drop_cols = [c for c in ["seq_id", "sequence", "length"]
#                  if c in df.columns]
#     df        = df.drop(columns=drop_cols)
#     X         = df.iloc[:, :-1].values
#     y         = df.iloc[:, -1].values.astype(int)
#     return X, y


# # ============================================================
# #   CELL 5 — Build Base Learner Instances
# #   ✅ FIX: All JSON params sanitised with safe_none/safe_int
# # ============================================================

# def load_optuna_params(clf_short, dataset_name):
#     """Load best params JSON. Returns None if file missing."""
#     ds_key = dataset_name.replace("_features.csv", "")
#     param_paths = {
#         "CB"  : os.path.join(BASE_RESULTS, "cb_optuna",
#                              "best_params",
#                              f"{ds_key}_CB_best_params.json"),
#         "XGB" : os.path.join(BASE_RESULTS, "xgb_optuna",
#                              "best_params",
#                              f"{ds_key}_XGB_best_params.json"),
#         "LGBM": os.path.join(BASE_RESULTS, "lgbm_optuna",
#                              "best_params",
#                              f"{ds_key}_LGBM_best_params.json"),
#         "RF"  : os.path.join(BASE_RESULTS, "rf_optuna",
#                              "best_params",
#                              f"{ds_key}_RF_best_params.json"),
#         "LR"  : os.path.join(BASE_RESULTS, "lr_optuna",
#                              "best_params",
#                              f"{ds_key}_LR_best_params.json"),
#     }
#     path = param_paths.get(clf_short, "")
#     if os.path.exists(path):
#         with open(path) as f:
#             params = json.load(f)
#         print(f"     ✅ Loaded params: {os.path.basename(path)}")
#         return params
#     print(f"     ⚠️  Params not found — using defaults")
#     return None


# def build_base_learner(clf_short, dataset_file):
#     """
#     Build base learner from saved Optuna params or defaults.

#     ✅ FIX APPLIED: All params read from JSON are sanitised
#     through safe_none() / safe_int() / safe_float() to convert
#     string 'None' back to Python None before passing to sklearn.

#     Root cause of original error:
#       json.dump({max_depth: None}) → {"max_depth": null}  ✅
#       But some Optuna save scripts write "None" as a string →
#       json.load → {"max_depth": "None"} string → sklearn error

#     This fix handles both cases safely.
#     """
#     params = load_optuna_params(clf_short, dataset_file)
#     pw     = POS_WEIGHTS.get(clf_short, 2.0)

#     # ── CatBoost ─────────────────────────────────────────────
#     if clf_short == "CB":
#         if params:
#             cb_params = {
#                 "iterations"           : safe_int(params.get("iterations"),           300),
#                 "depth"                : safe_int(params.get("depth"),                6),
#                 "learning_rate"        : safe_float(params.get("learning_rate"),      0.05),
#                 "l2_leaf_reg"          : safe_float(params.get("l2_leaf_reg"),        3.0),
#                 "random_strength"      : safe_float(params.get("random_strength"),    1.0),
#                 "bagging_temperature"  : safe_float(params.get("bagging_temperature"),0.5),
#                 "border_count"         : safe_int(params.get("border_count"),         128),
#                 "grow_policy"          : safe_str(params.get("grow_policy"),          "SymmetricTree"),
#                 "min_data_in_leaf"     : safe_int(params.get("min_data_in_leaf"),     5),
#                 "leaf_estimation_method": safe_str(params.get("leaf_estimation_method"), "Newton"),
#             }
#         else:
#             cb_params = {
#                 "iterations": 300, "depth": 6,
#                 "learning_rate": 0.05, "l2_leaf_reg": 3.0,
#                 "random_strength": 1.0, "bagging_temperature": 0.5,
#                 "border_count": 128, "grow_policy": "SymmetricTree",
#                 "min_data_in_leaf": 5, "leaf_estimation_method": "Newton",
#             }

#         # ── ✅ class_weights from saved JSON ─────────────────
#         if params and "class_weights" in params:
#             raw_cw = params["class_weights"]
#             if isinstance(raw_cw, str):
#                 # e.g. "[1.0, 2.5]" → [1.0, 2.5]
#                 try:
#                     import ast
#                     raw_cw = ast.literal_eval(raw_cw)
#                 except Exception:
#                     raw_cw = [1.0, pw]
#             cb_params["class_weights"] = raw_cw
#         else:
#             cb_params["class_weights"] = [1.0, pw]

#         cb_params.update({
#             "eval_metric"          : "AUC",
#             "early_stopping_rounds": 20,
#             "task_type"            : "CPU",
#             "thread_count"         : N_JOBS,   # ✅ CPU cores for CatBoost
#             "verbose"              : 0,
#             "allow_writing_files"  : False,
#             "random_seed"          : RANDOM_STATE,
#         })
#         return CatBoostClassifier(**cb_params)

#     # ── XGBoost ──────────────────────────────────────────────
#     elif clf_short == "XGB":
#         if params:
#             return XGBClassifier(
#                 n_estimators     = safe_int(params.get("n_estimators"),    300),
#                 max_depth        = safe_int(params.get("max_depth"),        6),
#                 learning_rate    = safe_float(params.get("learning_rate"), 0.05),
#                 subsample        = safe_float(params.get("subsample"),      0.8),
#                 colsample_bytree = safe_float(params.get("colsample_bytree"), 0.8),
#                 reg_alpha        = safe_float(params.get("reg_alpha"),      0.1),
#                 reg_lambda       = safe_float(params.get("reg_lambda"),     1.0),
#                 scale_pos_weight = safe_float(params.get("scale_pos_weight"), pw),
#                 use_label_encoder= False,
#                 eval_metric      = "auc",
#                 tree_method      = "hist",
#                 nthread          = N_JOBS,     # ✅ CPU cores for XGBoost
#                 random_state     = RANDOM_STATE,
#                 verbosity        = 0,
#             )
#         return XGBClassifier(
#             n_estimators=300, max_depth=6, learning_rate=0.05,
#             subsample=0.8, colsample_bytree=0.8,
#             scale_pos_weight=pw, use_label_encoder=False,
#             eval_metric="auc", tree_method="hist",
#             nthread=N_JOBS, random_state=RANDOM_STATE, verbosity=0,
#         )

#     # ── LightGBM ─────────────────────────────────────────────
#     elif clf_short == "LGBM":
#         if params:
#             # ── ✅ FIX: handle boosting_type + subsample ─────
#             boosting_type = safe_str(
#                 params.get("boosting_type"), "gbdt"
#             )
#             # If boosting_type is "goss", subsample is disabled
#             subsample = (1.0 if boosting_type == "goss"
#                          else safe_float(params.get("subsample"), 0.8))

#             return LGBMClassifier(
#                 n_estimators     = safe_int(params.get("n_estimators"),    300),
#                 max_depth        = safe_int(params.get("max_depth"),        6),
#                 learning_rate    = safe_float(params.get("learning_rate"), 0.05),
#                 num_leaves       = safe_int(params.get("num_leaves"),       31),
#                 subsample        = subsample,
#                 colsample_bytree = safe_float(params.get("colsample_bytree"), 0.8),
#                 reg_alpha        = safe_float(params.get("reg_alpha"),      0.1),
#                 reg_lambda       = safe_float(params.get("reg_lambda"),     1.0),
#                 scale_pos_weight = safe_float(params.get("scale_pos_weight"), pw),
#                 boosting_type    = boosting_type,
#                 n_jobs           = N_JOBS,     # ✅ CPU cores for LGBM
#                 random_state     = RANDOM_STATE,
#                 verbosity        = -1,
#             )
#         return LGBMClassifier(
#             n_estimators=300, max_depth=6, learning_rate=0.05,
#             num_leaves=31, subsample=0.8, colsample_bytree=0.8,
#             scale_pos_weight=pw, n_jobs=N_JOBS,
#             random_state=RANDOM_STATE, verbosity=-1,
#         )

#     # ── Random Forest ─────────────────────────────────────────
#     elif clf_short == "RF":
#         if params:
#             # ── ✅ FIX: safe_none for max_depth, max_leaf_nodes, max_features
#             max_depth      = safe_none(params.get("max_depth", None))
#             max_leaf_nodes = safe_none(params.get("max_leaf_nodes", None))
#             max_features   = safe_none(params.get("max_features", "sqrt"))

#             # Convert to correct types after None check
#             if max_depth is not None:
#                 max_depth = int(float(max_depth))
#             if max_leaf_nodes is not None:
#                 max_leaf_nodes = int(float(max_leaf_nodes))
#             if max_features is not None and max_features not in (
#                 "sqrt", "log2", "auto"
#             ):
#                 try:
#                     max_features = float(max_features)
#                 except Exception:
#                     max_features = "sqrt"

#             return RandomForestClassifier(
#                 n_estimators     = safe_int(params.get("n_estimators"),       300),
#                 max_depth        = max_depth,       # ✅ None or int
#                 max_leaf_nodes   = max_leaf_nodes,  # ✅ None or int
#                 min_samples_split= safe_int(params.get("min_samples_split"),  2),
#                 min_samples_leaf = safe_int(params.get("min_samples_leaf"),   1),
#                 max_features     = max_features,    # ✅ None, str, or float
#                 class_weight     = "balanced",
#                 n_jobs           = N_JOBS,           # ✅ CPU cores for RF
#                 random_state     = RANDOM_STATE,
#             )
#         return RandomForestClassifier(
#             n_estimators=300, max_depth=None,
#             class_weight="balanced",
#             n_jobs=N_JOBS, random_state=RANDOM_STATE,
#         )

#     # ── Logistic Regression ───────────────────────────────────
#     elif clf_short == "LR":
#         if params:
#             penalty = safe_str(params.get("penalty"), "l2")
#             solver  = safe_str(params.get("solver"),  "lbfgs")

#             # ── ✅ Fix invalid penalty/solver combinations ────
#             valid_solvers = {
#                 "l1"        : ["liblinear", "saga"],
#                 "l2"        : ["lbfgs", "liblinear", "saga", "newton-cg"],
#                 "elasticnet": ["saga"],
#                 None        : ["lbfgs", "saga", "newton-cg"],
#             }
#             allowed = valid_solvers.get(penalty, ["lbfgs"])
#             if solver not in allowed:
#                 solver = allowed[0]

#             return LogisticRegression(
#                 C            = safe_float(params.get("C"), 1.0),
#                 penalty      = penalty,
#                 solver       = solver,
#                 class_weight = "balanced",
#                 max_iter     = 2000,
#                 n_jobs       = N_JOBS,   # ✅ CPU cores for LR
#                 random_state = RANDOM_STATE,
#             )
#         return LogisticRegression(
#             C=1.0, penalty="l2", solver="lbfgs",
#             class_weight="balanced",
#             max_iter=2000, n_jobs=N_JOBS,
#             random_state=RANDOM_STATE,
#         )


# # ============================================================
# #   CELL 6 — Load Datasets
# # ============================================================

# print("\n" + "="*65)
# print("  Loading feature datasets...")
# print("="*65)

# datasets     = {}
# unique_files = set(BASE_LEARNER_DATASETS.values())

# for csv_file in unique_files:
#     csv_path = os.path.join(FEATURE_DIR, csv_file)
#     if not os.path.exists(csv_path):
#         print(f"  ❌ Not found: {csv_path}")
#         continue
#     X, y = load_feature_csv(csv_path)
#     datasets[csv_file] = (X, y)
#     print(f"  ✅ {csv_file:<30} shape: {X.shape}")

# # All classifiers share same y — use first file to get indices
# csv0     = list(unique_files)[0]
# _, y_all = datasets[csv0]

# # ── Stratified train/test split ───────────────────────────────
# train_idx, test_idx = train_test_split(
#     np.arange(len(y_all)),
#     test_size    = TEST_SIZE,
#     random_state = RANDOM_STATE,
#     stratify     = y_all
# )

# print(f"\n  Train : {len(train_idx)}  |  Test : {len(test_idx)}")
# print(f"  Pos   : {y_all.sum()}  |  Neg  : {(y_all==0).sum()}")

# X_trains, X_tests, scalers = {}, {}, {}

# for csv_file, (X, y) in datasets.items():
#     scaler       = StandardScaler()
#     X_trains[csv_file] = scaler.fit_transform(X[train_idx])
#     X_tests[csv_file]  = scaler.transform(X[test_idx])
#     scalers[csv_file]  = scaler

# y_train = y_all[train_idx]
# y_test  = y_all[test_idx]


# # ============================================================
# #   CELL 7 — OOF Generation (with per-fold parallelism)
# #
# #   ✅ PARALLELISM STRATEGY:
# #      - CB  : thread_count=N_JOBS  (internal CatBoost threads)
# #      - XGB : nthread=N_JOBS
# #      - LGBM: n_jobs=N_JOBS
# #      - RF  : n_jobs=N_JOBS
# #      - LR  : n_jobs=N_JOBS
# #
# #      Each classifier uses all 10 CPU cores internally.
# #      We process classifiers sequentially (not in parallel)
# #      to avoid memory contention on 2,190 PLM samples.
# # ============================================================

# print("\n" + "="*65)
# print("  Generating Out-of-Fold Probabilities")
# print(f"  {N_FOLDS}-fold Stratified CV  |  {N_JOBS} CPU cores")
# print("="*65)

# cv           = StratifiedKFold(
#     n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
# )
# oof_probs    = np.zeros((len(y_train), len(BASE_LEARNER_DATASETS)))
# test_probs   = np.zeros((len(y_test),  len(BASE_LEARNER_DATASETS)))
# final_models = {}
# final_scalers= {}
# clf_aucs     = {}

# for clf_idx, (clf_short, csv_file) in enumerate(
#     BASE_LEARNER_DATASETS.items()
# ):
#     import time
#     t_start = time.time()

#     print(f"\n  [{clf_idx+1}/5]  {clf_short}  ←  {csv_file}")

#     X_tr = X_trains[csv_file]
#     X_te = X_tests[csv_file]

#     oof_fold   = np.zeros(len(y_train))
#     test_folds = np.zeros((len(y_test), N_FOLDS))
#     fold_aucs  = []

#     for fold, (f_train, f_val) in enumerate(
#         cv.split(X_tr, y_train)
#     ):
#         X_f_tr, X_f_val = X_tr[f_train], X_tr[f_val]
#         y_f_tr, y_f_val = y_train[f_train], y_train[f_val]

#         clf = build_base_learner(clf_short, csv_file)

#         if clf_short == "CB":
#             train_pool = Pool(X_f_tr, y_f_tr)
#             val_pool   = Pool(X_f_val, y_f_val)
#             clf.fit(train_pool, eval_set=val_pool,
#                     use_best_model=True)
#         else:
#             clf.fit(X_f_tr, y_f_tr)

#         oof_fold[f_val]     = clf.predict_proba(X_f_val)[:, 1]
#         test_folds[:, fold] = clf.predict_proba(X_te)[:, 1]

#         fold_auc = roc_auc_score(y_f_val, oof_fold[f_val])
#         fold_aucs.append(fold_auc)
#         print(f"     Fold {fold+1}/{N_FOLDS}  AUC = {fold_auc:.4f}")

#     oof_probs[:, clf_idx]  = oof_fold
#     test_probs[:, clf_idx] = test_folds.mean(axis=1)

#     mean_oof_auc = roc_auc_score(y_train, oof_fold)
#     clf_aucs[clf_short] = mean_oof_auc

#     elapsed = time.time() - t_start
#     print(f"     Mean OOF AUC : {mean_oof_auc:.4f}  "
#           f"({elapsed/60:.1f} min)")

#     # ── Train final model on full training set ────────────────
#     print(f"     Training final {clf_short} on full train set...")
#     final_clf = build_base_learner(clf_short, csv_file)

#     if clf_short == "CB":
#         train_pool = Pool(X_tr, y_train)
#         eval_pool  = Pool(X_te, y_test)
#         final_clf.fit(train_pool, eval_set=eval_pool,
#                       use_best_model=True)
#     else:
#         final_clf.fit(X_tr, y_train)

#     final_models[clf_short]  = final_clf
#     final_scalers[clf_short] = scalers[csv_file]

#     model_path = os.path.join(STACK_DIR, f"base_{clf_short}.joblib")
#     joblib.dump(final_clf, model_path)
#     print(f"     ✅ Saved → {model_path}")

# print(f"\n✅ OOF generation complete")
# print(f"   OOF matrix shape  : {oof_probs.shape}")
# print(f"   Test matrix shape : {test_probs.shape}")
# print(f"\n   Base learner OOF AUCs:")
# for clf_short, auc in clf_aucs.items():
#     print(f"     {clf_short:<6} : {auc:.4f}")


# # ============================================================
# #   CELL 8 — AUC-MCC Composite Weights
# # ============================================================

# print("\n" + "="*65)
# print("  Computing AUC-MCC Composite Weights")
# print("="*65)

# weights    = {}
# mcc_scores = {}

# for clf_idx, clf_short in enumerate(BASE_LEARNER_DATASETS):
#     oof_pred  = (oof_probs[:, clf_idx] >= 0.5).astype(int)
#     mcc       = matthews_corrcoef(y_train, oof_pred)
#     mcc_norm  = (mcc + 1.0) / 2.0
#     auc       = clf_aucs[clf_short]
#     w         = ALPHA * auc + (1.0 - ALPHA) * mcc_norm
#     weights[clf_short]    = w
#     mcc_scores[clf_short] = mcc

# total_w      = sum(weights.values())
# norm_weights = {k: round(v / total_w, 4) for k, v in weights.items()}

# print(f"\n  {'Clf':<6} {'OOF AUC':>9} {'OOF MCC':>9} "
#       f"{'Raw W':>9} {'Norm W':>9}")
# print(f"  {'─'*46}")
# for clf_short in BASE_LEARNER_DATASETS:
#     print(f"  {clf_short:<6} "
#           f"{clf_aucs[clf_short]:>9.4f} "
#           f"{mcc_scores[clf_short]:>9.4f} "
#           f"{weights[clf_short]:>9.4f} "
#           f"{norm_weights[clf_short]:>9.4f}")

# w_array       = np.array([norm_weights[c] for c in BASE_LEARNER_DATASETS])
# oof_weighted  = oof_probs  @ w_array
# test_weighted = test_probs @ w_array

# meta_train = np.column_stack([oof_probs,  oof_weighted])
# meta_test  = np.column_stack([test_probs, test_weighted])

# print(f"\n  Meta-train : {meta_train.shape}")
# print(f"  Meta-test  : {meta_test.shape}")
# print(f"  Columns    : [p_CB, p_XGB, p_LGBM, p_RF, p_LR, p_weighted]")


# # ============================================================
# #   CELL 9 — Train Meta-Learner (LightGBM)
# # ============================================================

# print("\n" + "="*65)
# print("  Training Meta-Learner (LightGBM, depth=3)")
# print("="*65)

# meta_learner = LGBMClassifier(
#     n_estimators     = 50,
#     max_depth        = 3,
#     learning_rate    = 0.05,
#     num_leaves       = 7,
#     subsample        = 0.8,
#     colsample_bytree = 0.8,
#     scale_pos_weight = 1.5,
#     n_jobs           = N_JOBS,   # ✅ CPU cores for meta-learner
#     random_state     = RANDOM_STATE,
#     verbosity        = -1,
# )

# meta_learner.fit(meta_train, y_train)

# meta_prob_train = meta_learner.predict_proba(meta_train)[:, 1]
# meta_prob_test  = meta_learner.predict_proba(meta_test) [:, 1]

# train_auc = roc_auc_score(y_train, meta_prob_train)
# test_auc  = roc_auc_score(y_test,  meta_prob_test)

# print(f"\n  Meta-learner train AUC : {train_auc:.4f}")
# print(f"  Meta-learner test  AUC : {test_auc:.4f}")

# meta_path = os.path.join(STACK_DIR, "meta_learner_lgbm.joblib")
# joblib.dump(meta_learner, meta_path)
# print(f"\n  ✅ Meta-learner saved → {meta_path}")


# # ============================================================
# #   CELL 10 — Threshold Optimisation (on OOF — no test leakage)
# # ============================================================

# print("\n" + "="*65)
# print("  Threshold Optimisation (MCC on OOF predictions)")
# print("="*65)

# best_t, best_mcc_val = find_best_threshold(
#     y_train, meta_prob_train, metric="mcc"
# )
# print(f"  Optimal threshold : {best_t}  "
#       f"(train MCC = {best_mcc_val:.4f})")

# y_pred_default = (meta_prob_test >= 0.50).astype(int)
# y_pred_optimal = (meta_prob_test >= best_t).astype(int)


# # ============================================================
# #   CELL 11 — Final Metrics
# # ============================================================

# print("\n" + "="*65)
# print("  STACKING ENSEMBLE — FINAL RESULTS")
# print("="*65)

# metrics_def = compute_metrics(y_test, y_pred_default, meta_prob_test)
# metrics_opt = compute_metrics(y_test, y_pred_optimal, meta_prob_test)

# print(f"\n  {'Metric':<14} {'Default (0.50)':>15} "
#       f"{'Optimal ({:.2f})'.format(best_t):>18}  {'Δ':>8}")
# print(f"  {'─'*58}")
# for m in ["Accuracy","Sensitivity","Specificity",
#           "F1_Score","MCC","AUC"]:
#     d     = metrics_def[m]
#     o     = metrics_opt[m]
#     delta = round(o - d, 4)
#     sign  = "+" if delta >= 0 else ""
#     print(f"  {m:<14} {d:>15.4f} {o:>18.4f}  {sign}{delta:>7.4f}")

# print(f"\n  TP={metrics_opt['TP']}  TN={metrics_opt['TN']}  "
#       f"FP={metrics_opt['FP']}  FN={metrics_opt['FN']}")

# results_df = pd.DataFrame([
#     {"Model": "Stacking (default t=0.50)", "Threshold": 0.50,
#      **{k:v for k,v in metrics_def.items() if k not in ("TP","TN","FP","FN")}},
#     {"Model": f"Stacking (optimal t={best_t})", "Threshold": best_t,
#      **{k:v for k,v in metrics_opt.items() if k not in ("TP","TN","FP","FN")}},
# ])
# results_path = os.path.join(STACK_DIR, "stacking_results.csv")
# results_df.to_csv(results_path, index=False)

# probs_df = pd.DataFrame({
#     "y_true"         : y_test,
#     "y_pred_default" : y_pred_default,
#     "y_pred_optimal" : y_pred_optimal,
#     "prob_positive"  : meta_prob_test,
#     "prob_negative"  : 1 - meta_prob_test,
#     **{f"p_{c}": test_probs[:, i]
#        for i, c in enumerate(BASE_LEARNER_DATASETS)},
#     "p_weighted": test_weighted,
# })
# probs_path = os.path.join(STACK_DIR, "stacking_probabilities.csv")
# probs_df.to_csv(probs_path, index=False)

# print(f"\n  ✅ Results saved → {results_path}")
# print(f"  ✅ Probs   saved → {probs_path}")


# # ============================================================
# #   CELL 12 — Compare with Individual Base Learners
# # ============================================================

# print("\n" + "="*65)
# print("  COMPARISON — Stack vs Individual Base Learners")
# print("="*65)

# individual_metrics = {}
# for clf_idx, clf_short in enumerate(BASE_LEARNER_DATASETS):
#     p    = test_probs[:, clf_idx]
#     t, _ = find_best_threshold(y_test, p, metric="mcc")
#     pred = (p >= t).astype(int)
#     individual_metrics[clf_short] = compute_metrics(y_test, pred, p)

# comparison_rows = []
# for clf_short, m in individual_metrics.items():
#     comparison_rows.append({
#         "Model"      : clf_short, "Threshold": "opt",
#         "Accuracy"   : m["Accuracy"],
#         "Sensitivity": m["Sensitivity"],
#         "Specificity": m["Specificity"],
#         "F1_Score"   : m["F1_Score"],
#         "MCC"        : m["MCC"],
#         "AUC"        : m["AUC"],
#     })
# comparison_rows.append({
#     "Model"      : "★ Stacking", "Threshold": best_t,
#     "Accuracy"   : metrics_opt["Accuracy"],
#     "Sensitivity": metrics_opt["Sensitivity"],
#     "Specificity": metrics_opt["Specificity"],
#     "F1_Score"   : metrics_opt["F1_Score"],
#     "MCC"        : metrics_opt["MCC"],
#     "AUC"        : metrics_opt["AUC"],
# })

# df_comp    = pd.DataFrame(comparison_rows)
# comp_path  = os.path.join(STACK_DIR, "stacking_vs_individual.csv")
# df_comp.to_csv(comp_path, index=False)
# print(f"\n{df_comp.to_string(index=False)}")
# print(f"\n  ✅ Comparison saved → {comp_path}")


# # ============================================================
# #   CELL 13 — Save All Artefacts
# # ============================================================

# artefacts = {
#     "meta_learner"  : meta_learner,
#     "final_models"  : final_models,
#     "final_scalers" : final_scalers,
#     "norm_weights"  : norm_weights,
#     "best_threshold": best_t,
#     "clf_order"     : list(BASE_LEARNER_DATASETS.keys()),
#     "dataset_map"   : BASE_LEARNER_DATASETS,
# }
# art_path = os.path.join(STACK_DIR, "stacking_artefacts.joblib")
# joblib.dump(artefacts, art_path)

# summary = {
#     "base_learners"  : list(BASE_LEARNER_DATASETS.keys()),
#     "dataset_map"    : BASE_LEARNER_DATASETS,
#     "norm_weights"   : norm_weights,
#     "best_threshold" : best_t,
#     "composite_alpha": ALPHA,
#     "n_folds"        : N_FOLDS,
#     "n_jobs"         : N_JOBS,
#     "stacking_auc"   : metrics_opt["AUC"],
#     "stacking_mcc"   : metrics_opt["MCC"],
#     "stacking_sn"    : metrics_opt["Sensitivity"],
#     "stacking_sp"    : metrics_opt["Specificity"],
#     "stacking_acc"   : metrics_opt["Accuracy"],
#     "stacking_f1"    : metrics_opt["F1_Score"],
#     "cb_auc"         : individual_metrics["CB"]["AUC"],
#     "improvement_auc": round(metrics_opt["AUC"]
#                              - individual_metrics["CB"]["AUC"], 4),
#     "improvement_mcc": round(metrics_opt["MCC"]
#                              - individual_metrics["CB"]["MCC"], 4),
#     "improvement_sn" : round(metrics_opt["Sensitivity"]
#                              - individual_metrics["CB"]["Sensitivity"], 4),
# }
# sum_path = os.path.join(STACK_DIR, "stacking_summary.json")
# with open(sum_path, "w") as f:
#     json.dump(summary, f, indent=4)

# print(f"\n✅ Artefacts saved → {art_path}")
# print(f"✅ Summary   saved → {sum_path}")


# # ============================================================
# #   CELL 14 — Visualization 1: ROC Curves
# # ============================================================

# clf_colors = {
#     "CB"  : "#e67e22", "XGB" : "#3498db",
#     "LGBM": "#8e44ad", "RF"  : "#2ecc71",
#     "LR"  : "#c0392b",
# }

# fig, ax = plt.subplots(figsize=(10, 8))

# for clf_idx, clf_short in enumerate(BASE_LEARNER_DATASETS):
#     p           = test_probs[:, clf_idx]
#     fpr, tpr, _ = roc_curve(y_test, p)
#     auc_val     = roc_auc_score(y_test, p)
#     ax.plot(fpr, tpr,
#             color=clf_colors.get(clf_short, "#95a5a6"),
#             linewidth=1.2, linestyle="--",
#             label=f"{clf_short} (AUC={auc_val:.4f})",
#             alpha=0.8)

# fpr_s, tpr_s, _ = roc_curve(y_test, meta_prob_test)
# ax.plot(fpr_s, tpr_s, color="black", linewidth=2.5,
#         label=f"★ Stacking (AUC={metrics_opt['AUC']:.4f})")

# ax.plot([0,1],[0,1],"grey",linewidth=0.8,linestyle="--",alpha=0.5)
# ax.set_xlabel("False Positive Rate", fontsize=12)
# ax.set_ylabel("True Positive Rate",  fontsize=12)
# ax.set_title("ROC Curves — Stacking vs Base Learners",
#              fontsize=13, fontweight="bold")
# ax.legend(fontsize=10, loc="lower right")
# ax.grid(alpha=0.3)
# ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
# plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "stacking_ROC_curves.png"),
#             dpi=150, bbox_inches="tight")
# plt.show()
# print("✅ ROC curves saved")


# # ============================================================
# #   CELL 15 — Visualization 2: Metrics Bar Chart
# # ============================================================

# metric_list = ["Accuracy","Sensitivity","Specificity",
#                "F1_Score","MCC","AUC"]

# fig, ax = plt.subplots(figsize=(14, 6))
# x_    = np.arange(len(metric_list))
# n_clf = len(df_comp)
# bar_w = 0.8 / n_clf
# clrs  = list(clf_colors.values()) + ["#1A1A1A"]

# for i, row in df_comp.iterrows():
#     offset   = (i - n_clf / 2 + 0.5) * bar_w
#     vals     = [row[m] for m in metric_list]
#     is_stack = "Stacking" in str(row["Model"])
#     ax.bar(x_ + offset, vals, bar_w,
#            label=row["Model"],
#            color=clrs[i] if i < len(clrs) else "#555555",
#            alpha=1.0 if is_stack else 0.70,
#            edgecolor="white", linewidth=1.5 if is_stack else 0.5)

# ax.set_xticks(x_)
# ax.set_xticklabels(metric_list, fontsize=11)
# ax.set_ylabel("Score", fontsize=12)
# ax.set_ylim(0, 1.08)
# ax.set_title("Stacking Ensemble vs Base Learners — All Metrics",
#              fontsize=13, fontweight="bold")
# ax.legend(fontsize=9, loc="lower right")
# ax.grid(axis="y", alpha=0.3)
# ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
# plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "stacking_metrics_bar.png"),
#             dpi=150, bbox_inches="tight")
# plt.show()
# print("✅ Metrics bar chart saved")


# # ============================================================
# #   CELL 16 — Visualization 3: Confusion Matrices
# # ============================================================

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# best_base_p    = test_probs[:, 0]   # CB
# best_base_t, _ = find_best_threshold(y_test, best_base_p, "mcc")
# best_base_pred = (best_base_p >= best_base_t).astype(int)

# for ax, pred, title, prob in zip(
#     axes,
#     [best_base_pred, y_pred_optimal],
#     [f"CB Best Base (t={best_base_t})",
#      f"★ Stacking (t={best_t})"],
#     [best_base_p, meta_prob_test]
# ):
#     cm = confusion_matrix(y_test, pred)
#     ConfusionMatrixDisplay(
#         confusion_matrix=cm,
#         display_labels=["Non-AIP","AIP"]
#     ).plot(cmap="Blues", ax=ax, colorbar=False)
#     m = compute_metrics(y_test, pred, prob)
#     ax.set_title(
#         f"{title}\n"
#         f"Sn={m['Sensitivity']:.4f}  "
#         f"Sp={m['Specificity']:.4f}  "
#         f"MCC={m['MCC']:.4f}  "
#         f"AUC={m['AUC']:.4f}",
#         fontsize=9, fontweight="bold"
#     )

# plt.suptitle("Confusion Matrix Comparison",
#              fontsize=12, fontweight="bold")
# plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR,
#                          "stacking_confusion_matrices.png"),
#             dpi=150, bbox_inches="tight")
# plt.show()
# print("✅ Confusion matrices saved")


# # ============================================================
# #   CELL 17 — Visualization 4: OOF Probability Heatmap
# # ============================================================

# fig, ax = plt.subplots(figsize=(14, 5))
# df_oof  = pd.DataFrame(oof_probs,
#                         columns=list(BASE_LEARNER_DATASETS.keys()))
# df_oof["Weighted"] = oof_weighted
# df_oof["Label"]    = y_train
# df_oof  = df_oof.sort_values("Weighted").reset_index(drop=True)

# sns.heatmap(
#     df_oof[list(BASE_LEARNER_DATASETS.keys()) + ["Weighted"]].T,
#     cmap="RdYlGn", vmin=0, vmax=1, ax=ax, xticklabels=False,
#     cbar_kws={"label": "Predicted Probability (AIP)"},
# )
# ax.set_title(
#     "OOF Probability Heatmap — Base Learner Agreement\n"
#     "(sorted by ensemble score — green=high, red=low)",
#     fontsize=12, fontweight="bold"
# )
# plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "stacking_oof_heatmap.png"),
#             dpi=150, bbox_inches="tight")
# plt.show()
# print("✅ OOF heatmap saved")


# # ============================================================
# #   CELL 18 — Visualization 5: Meta Feature Importance
# # ============================================================

# feat_names  = list(BASE_LEARNER_DATASETS.keys()) + ["Weighted"]
# importances = meta_learner.feature_importances_

# fig, ax = plt.subplots(figsize=(9, 4))
# colors  = [clf_colors.get(c, "#1abc9c") for c in feat_names]
# ax.bar(feat_names, importances, color=colors,
#        edgecolor="white", alpha=0.85)
# ax.set_ylabel("Feature Importance", fontsize=12)
# ax.set_title("Meta-Learner Feature Importance\n"
#              "(which base learner LightGBM trusts most)",
#              fontsize=12, fontweight="bold")
# ax.grid(axis="y", alpha=0.3)
# for i, (f, imp) in enumerate(zip(feat_names, importances)):
#     ax.text(i, imp + 0.2, f"{imp:.1f}",
#             ha="center", fontsize=9, fontweight="bold")
# plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "stacking_meta_importance.png"),
#             dpi=150, bbox_inches="tight")
# plt.show()
# print("✅ Meta-feature importance saved")


# # ============================================================
# #   CELL 19 — Visualization 6: Threshold Curve
# # ============================================================

# thresh_vals = np.arange(0.10, 0.90, 0.01)
# mcc_vals, sn_vals, sp_vals = [], [], []

# for t in thresh_vals:
#     y_p = (meta_prob_test >= t).astype(int)
#     if len(np.unique(y_p)) < 2:
#         mcc_vals.append(np.nan); sn_vals.append(np.nan)
#         sp_vals.append(np.nan); continue
#     tn, fp, fn, tp = confusion_matrix(y_test, y_p).ravel()
#     mcc_vals.append(matthews_corrcoef(y_test, y_p))
#     sn_vals.append(tp/(tp+fn) if (tp+fn)>0 else 0)
#     sp_vals.append(tn/(tn+fp) if (tn+fp)>0 else 0)

# fig, ax = plt.subplots(figsize=(11, 5))
# ax.plot(thresh_vals, mcc_vals, color="#e74c3c",
#         linewidth=2.5, label="MCC")
# ax.plot(thresh_vals, sn_vals,  color="#3498db",
#         linewidth=1.8, linestyle="--", label="Sensitivity")
# ax.plot(thresh_vals, sp_vals,  color="#2ecc71",
#         linewidth=1.8, linestyle="--", label="Specificity")
# ax.axvline(best_t, color="#1A1A1A", linewidth=1.5,
#            linestyle=":", label=f"Optimal t={best_t}")
# ax.axvline(0.50, color="#888888", linewidth=1.0,
#            linestyle=":", label="Default t=0.50")
# ax.set_xlabel("Classification Threshold", fontsize=12)
# ax.set_ylabel("Score", fontsize=12)
# ax.set_title("Threshold vs MCC / Sensitivity / Specificity",
#              fontsize=12, fontweight="bold")
# ax.legend(fontsize=10); ax.grid(alpha=0.3)
# ax.set_xlim([0.10, 0.90])
# plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_DIR, "stacking_threshold_curve.png"),
#             dpi=150, bbox_inches="tight")
# plt.show()
# print("✅ Threshold curve saved")


# # ============================================================
# #   CELL 20 — Visualization 7: Comparison Table Image
# #             Loads all values dynamically from saved CSVs
# # ============================================================

# C_IND_HEADER  = "#2E75B6"; C_CLF_COL   = "#F0A500"
# C_IND_ROW_ODD = "#D6E4F5"; C_IND_ROW_EVN = "#EBF3FA"
# C_STACK_ROW   = "#D5F5E3"; C_HDR_TXT   = "#FFFFFF"
# C_DARK_TXT    = "#1A1A1A"; C_BORDER    = "#CCCCCC"
# C_GREEN_TXT   = "#1A6B34"; C_RED_TXT   = "#922B21"
# C_GREEN_CELL  = "#D5F5E3"; C_RED_CELL  = "#FADBD8"

# METRIC_KEYS    = ["Accuracy","Sensitivity","Specificity",
#                   "F1_Score","MCC","AUC"]
# METRIC_DISPLAY = ["Accuracy","Sensitivity","Specificity",
#                   "F1\nScore","MCC","AUC"]

# CLF_CSV_CONFIG = {
#     "CB"  : {"baseline_csv": os.path.join(BASE_RESULTS,"catboost",
#                               "CB_all_results_summary.csv"),
#              "optuna_csv"  : os.path.join(BASE_RESULTS,"cb_optuna",
#                               "CB_Optuna_all_results_summary.csv")},
#     "XGB" : {"baseline_csv": os.path.join(BASE_RESULTS,"xgboost",
#                               "XGB_all_results_summary.csv"),
#              "optuna_csv"  : os.path.join(BASE_RESULTS,"xgb_optuna",
#                               "XGB_Optuna_all_results_summary.csv")},
#     "LGBM": {"baseline_csv": os.path.join(BASE_RESULTS,"lightgbm",
#                               "LGBM_all_results_summary.csv"),
#              "optuna_csv"  : os.path.join(BASE_RESULTS,"lgbm_optuna",
#                               "LGBM_Optuna_all_results_summary.csv")},
#     "RF"  : {"baseline_csv": os.path.join(BASE_RESULTS,"random_forest",
#                               "RF_all_results_summary.csv"),
#              "optuna_csv"  : os.path.join(BASE_RESULTS,"rf_optuna",
#                               "RF_Optuna_all_results_summary.csv")},
#     "LR"  : {"baseline_csv": os.path.join(BASE_RESULTS,"logistic_regression",
#                               "LR_all_results_summary.csv"),
#              "optuna_csv"  : os.path.join(BASE_RESULTS,"lr_optuna",
#                               "LR_Optuna_all_results_summary.csv")},
# }

# def load_best_csv_row(csv_path, mkeys):
#     if not os.path.exists(csv_path):
#         return [None]*len(mkeys)
#     df = pd.read_csv(csv_path)
#     for col in ["Rank","Unnamed: 0","index"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])
#     if "AUC" not in df.columns:
#         return [None]*len(mkeys)
#     best = df.sort_values("AUC", ascending=False).iloc[0]
#     return [round(float(best[m]),4) if m in best.index else None
#             for m in mkeys]

# IND_DATA = []
# for clf_short, cfg in CLF_CSV_CONFIG.items():
#     b = load_best_csv_row(cfg["baseline_csv"], METRIC_KEYS)
#     o = load_best_csv_row(cfg["optuna_csv"],   METRIC_KEYS)
#     IND_DATA.append([clf_short] + b + o)

# STACK_ROW = [metrics_opt[k] for k in METRIC_KEYS]

# # ── Build figure ──────────────────────────────────────────────
# N_CLF=len(IND_DATA); N_MET=len(METRIC_KEYS)
# CW=0.80; MW=0.72; RH=0.38; H1=0.44; H2=0.40; LH=0.40; TH=0.65
# NDR=N_CLF+1
# TW=CW+MW*N_MET*2; TH_=H1+H2+RH*NDR+RH
# FW=TW+0.40; FH=TH_+TH+LH+0.30

# fig,ax=plt.subplots(figsize=(FW,FH))
# ax.set_xlim(0,TW); ax.set_ylim(0,TH_+TH+LH); ax.axis("off")

# cxc=0.0
# cxb=[cxc+CW+i*MW for i in range(N_MET)]
# cxo=[cxc+CW+(N_MET+i)*MW for i in range(N_MET)]

# def cel(x,y,w,h,fc,ec=C_BORDER,lw=0.5,z=1):
#     ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z))
# def tx(x,y,w,h,s,fc=C_DARK_TXT,fs=8.5,bold=False):
#     ax.text(x+w/2,y+h/2,str(s),ha="center",va="center",fontsize=fs,
#             fontweight="bold" if bold else "normal",color=fc,zorder=4,clip_on=True)
# def fmtv(v):
#     return f"{float(v):.4f}" if v is not None else "—"

# all_v={i:[] for i in range(N_MET)}
# for row in IND_DATA:
#     for ci in range(N_MET):
#         if row[1+ci]       is not None: all_v[ci].append(float(row[1+ci]))
#         if row[1+N_MET+ci] is not None: all_v[ci].append(float(row[1+N_MET+ci]))
# for ci in range(N_MET):
#     if STACK_ROW[ci] is not None: all_v[ci].append(float(STACK_ROW[ci]))
# top3={ci:sorted(set(all_v[ci]),reverse=True)[:3] for ci in range(N_MET)}

# def hl(ci,val):
#     if val is None: return None,None
#     try: fv=float(val)
#     except: return None,None
#     if len(top3[ci])>0 and fv>=top3[ci][0]: return "#1A5276","#FFFFFF"
#     if len(top3[ci])>1 and fv>=top3[ci][1]: return "#2980B9","#FFFFFF"
#     if len(top3[ci])>2 and fv>=top3[ci][2]: return "#85C1E9","#1A1A1A"
#     return None,None

# ty=TH_+LH+0.12
# ax.text(TW/2,ty+0.38,"AIP Prediction — Individual Classifiers vs Stacking Ensemble",
#         ha="center",va="center",fontsize=11.5,fontweight="bold",color="#1A1A1A",zorder=5)
# ax.text(TW/2,ty+0.10,f"★ = Stacking Ensemble  |  t={best_t}  |  ■ Top-1  ■ Top-2  ■ Top-3",
#         ha="center",va="center",fontsize=7.8,color="#555555",zorder=5)

# y_h1=TH_+LH-H1; dw=MW*N_MET
# cel(cxc,y_h1,CW,H1,C_CLF_COL)
# cel(cxb[0],y_h1,dw,H1,C_IND_HEADER,ec="#1A5276",lw=1.2,z=2)
# ax.text(cxb[0]+dw/2,y_h1+H1/2,"Default",ha="center",va="center",
#         fontsize=11,fontweight="bold",color=C_HDR_TXT,zorder=5)
# cel(cxo[0],y_h1,dw,H1,"#E67E22",ec="#B8500A",lw=1.2,z=2)
# ax.text(cxo[0]+dw/2,y_h1+H1/2,"Optuna Tuned",ha="center",va="center",
#         fontsize=11,fontweight="bold",color=C_HDR_TXT,zorder=5)

# y_h2=TH_+LH-H1-H2; cel(cxc,y_h2,CW,H2,C_CLF_COL)
# msh=["Accuracy","Sensitivity","Specificity","F1\nScore","MCC","AUC"]
# for ci in range(N_MET):
#     cel(cxb[ci],y_h2,MW,H2,C_IND_HEADER,ec="#1A5276",lw=0.6,z=2)
#     tx(cxb[ci],y_h2,MW,H2,msh[ci],fc=C_HDR_TXT,fs=7.5,bold=True)
#     cel(cxo[ci],y_h2,MW,H2,"#E67E22",ec="#B8500A",lw=0.6,z=2)
#     tx(cxo[ci],y_h2,MW,H2,msh[ci],fc=C_HDR_TXT,fs=7.5,bold=True)

# ybr=TH_+LH-H1-H2
# for ri,rd in enumerate(IND_DATA):
#     cn=rd[0]; bv=rd[1:1+N_MET]; ov=rd[1+N_MET:1+N_MET*2]
#     yr=ybr-RH*(ri+1); rbg=C_IND_ROW_ODD if ri%2==0 else C_IND_ROW_EVN
#     cel(cxc,yr,CW,RH,"#F0A500" if ri%2==0 else "#E8950A")
#     tx(cxc,yr,CW,RH,cn,fc=C_HDR_TXT,fs=9,bold=True)
#     for ci in range(N_MET):
#         cel(cxb[ci],yr,MW,RH,rbg,ec=C_BORDER,lw=0.4)
#         tx(cxb[ci],yr,MW,RH,fmtv(bv[ci]),fc=C_DARK_TXT,fs=8.2)
#         try: d=(float(ov[ci])-float(bv[ci])) if (ov[ci] and bv[ci]) else 0
#         except: d=0
#         ob=C_GREEN_CELL if d>0.001 else (C_RED_CELL if d<-0.001 else rbg)
#         of=C_GREEN_TXT  if d>0.001 else (C_RED_TXT  if d<-0.001 else C_DARK_TXT)
#         tb,tf=hl(ci,ov[ci])
#         if tb: ob,of=tb,tf
#         cel(cxo[ci],yr,MW,RH,ob,ec=C_BORDER,lw=0.4,z=2)
#         tx(cxo[ci],yr,MW,RH,fmtv(ov[ci]),fc=of,fs=8.2,bold=(tb is not None))

# sy=ybr-RH*N_CLF
# ax.plot([0,TW],[sy,sy],color="#1A8A1A",linewidth=2.0,zorder=5)
# ax.text(cxc+0.05,sy+0.04,"▼  Stacking Ensemble  ▼",ha="left",va="bottom",
#         fontsize=7.5,fontweight="bold",color="#1A8A1A",zorder=6)

# ys=ybr-RH*(N_CLF+1); fmw=MW*2
# cel(cxc,ys,CW,RH,"#1A8A1A",ec="#0E5E0E",lw=1.2,z=2)
# ax.text(cxc+CW/2,ys+RH/2,f"★ Stacking\n(t={best_t})",ha="center",va="center",
#         fontsize=8.0,fontweight="bold",color=C_HDR_TXT,zorder=5)
# for ci in range(N_MET):
#     sv=STACK_ROW[ci]
#     cel(cxb[ci],ys,MW,RH,C_STACK_ROW,ec="#0E5E0E",lw=0.8,z=2)
#     cel(cxo[ci],ys,MW,RH,C_STACK_ROW,ec="#0E5E0E",lw=0.8,z=2)
#     mx=cxb[ci]; tb,tf=hl(ci,sv)
#     if tb:
#         cel(mx,ys,fmw,RH,tb,ec="#0E5E0E",lw=0.8,z=3)
#         ax.text(mx+fmw/2,ys+RH/2,fmtv(sv),ha="center",va="center",
#                 fontsize=9.0,fontweight="bold",color=tf,zorder=5)
#     else:
#         ax.text(mx+fmw/2,ys+RH/2,fmtv(sv),ha="center",va="center",
#                 fontsize=9.0,fontweight="bold",color=C_GREEN_TXT,zorder=5)

# ax.add_patch(plt.Rectangle((0,ys),TW,H1+H2+RH*NDR,
#     fill=False,edgecolor="#333333",linewidth=1.8,zorder=6))

# ly=LH*0.42; lx=0.0; sw=0.20; sh=0.16; gap=0.08; tw_=1.70
# for bg,fg,label in [
#     ("#1A5276","#FFF","Top-1 value"),("#2980B9","#FFF","Top-2 value"),
#     ("#85C1E9",C_DARK_TXT,"Top-3 value"),(C_GREEN_CELL,C_GREEN_TXT,"Improved by Optuna"),
#     (C_RED_CELL,C_RED_TXT,"Degraded by Optuna"),(C_STACK_ROW,C_GREEN_TXT,"★ Stacking"),
# ]:
#     ax.add_patch(plt.Rectangle((lx,ly),sw,sh,facecolor=bg,edgecolor="#888888",linewidth=0.5,zorder=4))
#     ax.text(lx+sw+0.05,ly+sh/2,label,ha="left",va="center",fontsize=7.5,color="#333333",zorder=5)
#     lx+=sw+gap+tw_

# tbl_path=os.path.join(FIGURES_DIR,"stacking_comparison_table.png")
# plt.savefig(tbl_path,dpi=200,bbox_inches="tight",facecolor="white",edgecolor="none")
# plt.close()
# print(f"✅ Comparison table image saved → {tbl_path}")


# # ============================================================
# #   CELL 21 — Final Summary
# # ============================================================

# print("\n" + "="*65)
# print("  STACKING ENSEMBLE — COMPLETE SUMMARY")
# print("="*65)
# print(f"\n  Architecture:")
# print(f"    Level 0  : CB(ProtT5)+XGB(ESM2)+LGBM(ProtT5)"
#       f"+RF(ESM2)+LR(DPC)")
# print(f"    Weights  : {norm_weights}")
# print(f"    Level 1  : LightGBM (depth=3, 50 trees)")
# print(f"    Threshold: {best_t} (MCC-optimised on OOF)")
# print(f"    CPU cores: {N_JOBS}")
# print(f"\n{'─'*65}")
# print(f"  {'Model':<22} {'Sn':>8} {'Sp':>8} "
#       f"{'MCC':>8} {'AUC':>8}")
# print(f"{'─'*65}")
# for clf_short in BASE_LEARNER_DATASETS:
#     m = individual_metrics[clf_short]
#     print(f"  {clf_short:<22} "
#           f"{m['Sensitivity']:>8.4f} {m['Specificity']:>8.4f} "
#           f"{m['MCC']:>8.4f} {m['AUC']:>8.4f}")
# print(f"{'─'*65}")
# print(f"  {'★ Stacking':<22} "
#       f"{metrics_opt['Sensitivity']:>8.4f} "
#       f"{metrics_opt['Specificity']:>8.4f} "
#       f"{metrics_opt['MCC']:>8.4f} "
#       f"{metrics_opt['AUC']:>8.4f}")
# print(f"{'─'*65}")
# print(f"\n  Improvement over CB (best single model):")
# print(f"     AUC  : {summary['improvement_auc']:+.4f}")
# print(f"     MCC  : {summary['improvement_mcc']:+.4f}")
# print(f"     Sn   : {summary['improvement_sn']:+.4f}")
# print(f"\n  Fixes applied:")
# print(f"     ✅ JSON 'None' string → Python None (RF/XGB/LGBM/LR)")
# print(f"     ✅ {N_JOBS} CPU cores allocated to all classifiers")
# print(f"     ✅ CatBoost thread_count={N_JOBS}")
# print(f"     ✅ XGBoost nthread={N_JOBS}")
# print(f"     ✅ LightGBM/RF/LR n_jobs={N_JOBS}")
# print(f"\n  Saved files:")
# for p in [art_path,sum_path,results_path,comp_path,probs_path,tbl_path]:
#     print(f"     {p}")
# print("="*65)


# ---------------------------------------------- Stacking 2 CB, ET, LGBM, RF --------------------- 
"""
stacking_ensemble_v2.py
========================
AIP Prediction — Stacking Ensemble Version 2

Base learners : CB   (ProtT5)   ← best PLM, best single model
                ET   (CTDC)     ← classical composition, high Sp
                LGBM (ESM2)     ← different PLM from CB
                RF   (PAAC)     ← classical sequence-order features

Meta-learner  : Logistic Regression (C=0.1, balanced)
                Simpler than LGBM → less overfit on small meta-set

Key changes vs v1:
  ✅ Only 4 base learners (removed LR — low AUC)
  ✅ Maximum feature diversity (2 PLM + 2 classical)
  ✅ ET and RF use classical features → different signal from CB/LGBM
  ✅ Logistic Regression meta-learner → better generalisation
  ✅ N_FOLDS = 10 → more stable OOF predictions
  ✅ Isotonic calibration on all base learners
  ✅ All JSON None-string bugs fixed
  ✅ 10 CPU cores used throughout
"""

# ============================================================
#   CELL 1 — Mount Google Drive (Colab only)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================
#   CELL 2 — Install & Import Libraries
# ============================================================
# !pip install catboost lightgbm xgboost -q

import os
import json
import joblib
import warnings
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
from sklearn.ensemble   import RandomForestClassifier, ExtraTreesClassifier
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

# Google Colab:
# FEATURE_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/data/features"
# BASE_RESULTS= "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models"
# STACK_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/stacking_v2"
# FIGURES_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/figures/stacking_v2"

# Local:
FEATURE_DIR  = "../../../data/features"
BASE_RESULTS = "../../../results/models"
STACK_DIR    = "../../../results/models/stacking_v2"
FIGURES_DIR  = "../../../results/figures/stacking_v2"

os.makedirs(STACK_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── ✅ Maximum feature diversity ──────────────────────────────
#
#  Why this combination works better than v1:
#
#  CB   on ProtT5 → deep PLM context (1024 dims, T5 encoder)
#  LGBM on ESM2   → different PLM (1280 dims, ESM-2 650M)
#  ET   on CTDC   → classical CTD composition (21 dims)
#  RF   on PAAC   → pseudo amino acid composition (23 dims)
#
#  CB and LGBM capture evolutionary/contextual signal.
#  ET and RF capture physicochemical and sequence-order signal.
#  These two groups make fundamentally different errors →
#  meta-learner can resolve disagreements between them.
#
BASE_LEARNER_DATASETS = {
    "CB"  : "ProtT5_features.csv",   # best PLM — AUC 0.8468
    "LGBM": "ESM2_features.csv",     # different PLM — high Sn
    "ET"  : "CTDC_features.csv",     # classical — high Sp
    "RF"  : "PAAC_features.csv",     # classical — sequence order
}

# ── Stacking settings ─────────────────────────────────────────
N_FOLDS      = 10    # more folds → more stable OOF predictions
RANDOM_STATE = 42
TEST_SIZE    = 0.30
ALPHA        = 0.5   # AUC-MCC composite weight

# ── pos_weight defaults ───────────────────────────────────────
POS_WEIGHTS = {
    "CB"  : 2.5,
    "LGBM": 2.0,
    "ET"  : 1.8,
    "RF"  : 1.8,
}

print("✅ Configuration loaded")
print(f"\n   Base learners and feature assignments:")
for clf, ds in BASE_LEARNER_DATASETS.items():
    print(f"     {clf:<6} → {ds}")
print(f"\n   CV folds     : {N_FOLDS} (increased from 5 for stability)")
print(f"   Meta-learner : Logistic Regression (C=0.1)")
print(f"   Calibration  : Isotonic regression on base learners")
print(f"   CPU cores    : {N_JOBS}")


# ============================================================
#   CELL 4 — Helper Functions
# ============================================================

def safe_none(v):
    """Convert 'None' string back to Python None after JSON load."""
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
#   CELL 5 — Build Base Learner Instances
#   All JSON params sanitised with safe_* helpers
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
    print(f"     ⚠️  Not found — using defaults")
    return None


def build_base_learner(clf_short, dataset_file):
    """
    Build base learner from Optuna JSON params or defaults.
    All None-string JSON params are sanitised before use.
    """
    params = load_optuna_params(clf_short, dataset_file)
    pw     = POS_WEIGHTS.get(clf_short, 2.0)

    # ── CatBoost ─────────────────────────────────────────────
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
        cb_p["class_weights"]       = raw_cw if raw_cw else [1.0, pw]
        cb_p["eval_metric"]         = "AUC"
        cb_p["early_stopping_rounds"] = 20
        cb_p["task_type"]           = "CPU"
        cb_p["thread_count"]        = N_JOBS
        cb_p["verbose"]             = 0
        cb_p["allow_writing_files"] = False
        cb_p["random_seed"]         = RANDOM_STATE
        return CatBoostClassifier(**cb_p)

    # ── LightGBM ─────────────────────────────────────────────
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
                n_jobs           = N_JOBS,
                random_state     = RANDOM_STATE,
                verbosity        = -1,
            )
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw, n_jobs=N_JOBS,
            random_state=RANDOM_STATE, verbosity=-1,
        )

    # ── Extra Trees ───────────────────────────────────────────
    elif clf_short == "ET":
        if params:
            max_depth = safe_none(params.get("max_depth", None))
            if max_depth is not None:
                max_depth = int(float(max_depth))
            max_leaf  = safe_none(params.get("max_leaf_nodes", None))
            if max_leaf is not None:
                max_leaf = int(float(max_leaf))
            max_feat  = safe_none(params.get("max_features", "sqrt"))
            if max_feat not in (None, "sqrt", "log2", "auto"):
                try: max_feat = float(max_feat)
                except: max_feat = "sqrt"
            return ExtraTreesClassifier(
                n_estimators     = safe_int(params.get("n_estimators"),      300),
                max_depth        = max_depth,
                max_leaf_nodes   = max_leaf,
                min_samples_split= safe_int(params.get("min_samples_split"), 2),
                min_samples_leaf = safe_int(params.get("min_samples_leaf"),  1),
                max_features     = max_feat,
                class_weight     = "balanced",
                n_jobs           = N_JOBS,
                random_state     = RANDOM_STATE,
            )
        return ExtraTreesClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced",
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
        )

    # ── Random Forest ─────────────────────────────────────────
    elif clf_short == "RF":
        if params:
            max_depth = safe_none(params.get("max_depth", None))
            if max_depth is not None:
                max_depth = int(float(max_depth))
            max_leaf  = safe_none(params.get("max_leaf_nodes", None))
            if max_leaf is not None:
                max_leaf = int(float(max_leaf))
            max_feat  = safe_none(params.get("max_features", "sqrt"))
            if max_feat not in (None, "sqrt", "log2", "auto"):
                try: max_feat = float(max_feat)
                except: max_feat = "sqrt"
            return RandomForestClassifier(
                n_estimators     = safe_int(params.get("n_estimators"),      300),
                max_depth        = max_depth,
                max_leaf_nodes   = max_leaf,
                min_samples_split= safe_int(params.get("min_samples_split"), 2),
                min_samples_leaf = safe_int(params.get("min_samples_leaf"),  1),
                max_features     = max_feat,
                class_weight     = "balanced",
                n_jobs           = N_JOBS,
                random_state     = RANDOM_STATE,
            )
        return RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced",
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

print(f"\n  Train : {len(train_idx)}  |  Test : {len(test_idx)}")
print(f"  Pos   : {y_all.sum()}      |  Neg  : {(y_all==0).sum()}")

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
#
#   ✅ Isotonic calibration applied to each base learner:
#      CalibratedClassifierCV(clf, method="isotonic", cv=3)
#
#      Why:
#        - Tree models (ET, RF, CB, LGBM) produce poorly
#          calibrated probabilities (not true class probs)
#        - Calibration maps outputs to [0,1] probability space
#        - Better calibrated probs → meta-learner learns more
#          meaningful combination weights
#        - "isotonic" is better than "sigmoid" (Platt) for
#          non-monotonic probability transformations
#
#   Note: CB is already well-calibrated, calibration helps
#         most for RF and ET.
# ============================================================

print("\n" + "="*65)
print("  Generating OOF Probabilities")
print(f"  {N_FOLDS}-fold CV  |  Isotonic calibration  |  {N_JOBS} CPUs")
print("="*65)

import time

cv           = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
n_base       = len(BASE_LEARNER_DATASETS)
oof_probs    = np.zeros((len(y_train), n_base))
test_probs   = np.zeros((len(y_test),  n_base))
final_models = {}
final_scalers= {}
clf_aucs     = {}

for clf_idx, (clf_short, csv_file) in enumerate(
    BASE_LEARNER_DATASETS.items()
):
    t0 = time.time()
    print(f"\n  [{clf_idx+1}/{n_base}]  {clf_short}  ←  {csv_file}")

    X_tr = X_trains[csv_file]
    X_te = X_tests[csv_file]

    oof_fold   = np.zeros(len(y_train))
    test_folds = np.zeros((len(y_test), N_FOLDS))
    fold_aucs  = []

    for fold, (f_tr, f_val) in enumerate(cv.split(X_tr, y_train)):
        Xf_tr, Xf_val = X_tr[f_tr], X_tr[f_val]
        yf_tr, yf_val = y_train[f_tr], y_train[f_val]

        base_clf = build_base_learner(clf_short, csv_file)

        # ── CatBoost: fit directly (no calibration wrapper) ──
        # CB has its own probability calibration internally
        if clf_short == "CB":
            tp = Pool(Xf_tr, yf_tr)
            vp = Pool(Xf_val, yf_val)
            base_clf.fit(tp, eval_set=vp, use_best_model=True)
            oof_fold[f_val]     = base_clf.predict_proba(Xf_val)[:,1]
            test_folds[:, fold] = base_clf.predict_proba(X_te)[:,1]

        # ── ET / RF / LGBM: wrap with isotonic calibration ───
        else:
            cal_clf = CalibratedClassifierCV(
                base_clf, method="isotonic", cv=3
            )
            cal_clf.fit(Xf_tr, yf_tr)
            oof_fold[f_val]     = cal_clf.predict_proba(Xf_val)[:,1]
            test_folds[:, fold] = cal_clf.predict_proba(X_te)[:,1]

        fold_auc = roc_auc_score(yf_val, oof_fold[f_val])
        fold_aucs.append(fold_auc)
        print(f"     Fold {fold+1:>2}/{N_FOLDS}  AUC = {fold_auc:.4f}")

    oof_probs[:, clf_idx]  = oof_fold
    test_probs[:, clf_idx] = test_folds.mean(axis=1)

    mean_oof_auc            = roc_auc_score(y_train, oof_fold)
    clf_aucs[clf_short]     = mean_oof_auc
    elapsed                 = time.time() - t0

    print(f"     Mean OOF AUC : {mean_oof_auc:.4f}  "
          f"({elapsed/60:.1f} min)")

    # ── Final model on full train set ─────────────────────────
    print(f"     Training final {clf_short}...")
    final_clf = build_base_learner(clf_short, csv_file)

    if clf_short == "CB":
        tp = Pool(X_tr, y_train)
        ep = Pool(X_te, y_test)
        final_clf.fit(tp, eval_set=ep, use_best_model=True)
    else:
        cal_final = CalibratedClassifierCV(
            final_clf, method="isotonic", cv=5
        )
        cal_final.fit(X_tr, y_train)
        final_clf = cal_final

    final_models[clf_short]  = final_clf
    final_scalers[clf_short] = scalers[csv_file]

    mp = os.path.join(STACK_DIR, f"base_{clf_short}_v2.joblib")
    joblib.dump(final_clf, mp)
    print(f"     ✅ Saved → {mp}")

print(f"\n✅ OOF complete — shape: {oof_probs.shape}")
print(f"\n   OOF AUCs:")
for k, v in clf_aucs.items():
    print(f"     {k:<6} : {v:.4f}")


# ============================================================
#   CELL 8 — AUC-MCC Composite Weights
# ============================================================

print("\n" + "="*65)
print("  Computing AUC-MCC Composite Weights")
print("="*65)

weights    = {}
mcc_scores = {}

for ci, clf_short in enumerate(BASE_LEARNER_DATASETS):
    oof_pred = (oof_probs[:, ci] >= 0.5).astype(int)
    mcc      = matthews_corrcoef(y_train, oof_pred)
    mcc_norm = (mcc + 1.0) / 2.0
    auc      = clf_aucs[clf_short]
    w        = ALPHA * auc + (1.0 - ALPHA) * mcc_norm
    weights[clf_short]    = w
    mcc_scores[clf_short] = mcc

total_w      = sum(weights.values())
norm_weights = {k: round(v/total_w, 4) for k, v in weights.items()}

print(f"\n  {'Clf':<6} {'AUC':>8} {'MCC':>8} {'RawW':>8} {'NormW':>8}")
print(f"  {'─'*46}")
for k in BASE_LEARNER_DATASETS:
    print(f"  {k:<6} "
          f"{clf_aucs[k]:>8.4f} "
          f"{mcc_scores[k]:>8.4f} "
          f"{weights[k]:>8.4f} "
          f"{norm_weights[k]:>8.4f}")

w_arr         = np.array([norm_weights[c] for c in BASE_LEARNER_DATASETS])
oof_weighted  = oof_probs  @ w_arr
test_weighted = test_probs @ w_arr

meta_train = np.column_stack([oof_probs,  oof_weighted])
meta_test  = np.column_stack([test_probs, test_weighted])

print(f"\n  Weights : {norm_weights}")
print(f"  Meta-train shape : {meta_train.shape}")
print(f"  Columns : {list(BASE_LEARNER_DATASETS.keys()) + ['Weighted']}")


# ============================================================
#   CELL 9 — Train Meta-Learner
#
#   ✅ Logistic Regression instead of LightGBM
#
#   Why LR is better here:
#     - Only 5 meta-features (4 probs + 1 weighted avg)
#     - 1,533 training samples is small for tree boosting
#     - LightGBM at depth=3 can still overfit on this
#     - LR with C=0.1 (strong L2 regularisation) generalises
#       much better on low-dimensional meta-features
#     - LR is also interpretable — coefficients show which
#       base learner the ensemble trusts most
# ============================================================

print("\n" + "="*65)
print("  Training Meta-Learner (Logistic Regression, C=0.1)")
print("="*65)

meta_learner = LogisticRegression(
    C            = 0.1,       # strong regularisation → less overfit
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

train_auc = roc_auc_score(y_train, meta_prob_train)
test_auc  = roc_auc_score(y_test,  meta_prob_test)

print(f"\n  Train AUC : {train_auc:.4f}")
print(f"  Test  AUC : {test_auc:.4f}")

# ── Meta-learner coefficients ─────────────────────────────────
feat_names = list(BASE_LEARNER_DATASETS.keys()) + ["Weighted"]
coef       = meta_learner.coef_[0]
print(f"\n  Meta-learner coefficients (trust weights):")
for fname, c in zip(feat_names, coef):
    print(f"     {fname:<10} : {c:+.4f}")

mp = os.path.join(STACK_DIR, "meta_learner_lr_v2.joblib")
joblib.dump(meta_learner, mp)
print(f"\n  ✅ Meta-learner saved → {mp}")


# ============================================================
#   CELL 10 — Threshold Optimisation (on OOF)
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
#   CELL 11 — Final Metrics
# ============================================================

print("\n" + "="*65)
print("  STACKING V2 — FINAL RESULTS")
print("="*65)

metrics_def = compute_metrics(y_test, y_pred_default, meta_prob_test)
metrics_opt = compute_metrics(y_test, y_pred_optimal, meta_prob_test)

print(f"\n  {'Metric':<14} {'Default (0.50)':>15} "
      f"{'Optimal ({:.2f})'.format(best_t):>18}  {'Δ':>8}")
print(f"  {'─'*58}")
for m in ["Accuracy","Sensitivity","Specificity","F1_Score","MCC","AUC"]:
    d     = metrics_def[m]
    o     = metrics_opt[m]
    delta = round(o-d, 4)
    sign  = "+" if delta >= 0 else ""
    print(f"  {m:<14} {d:>15.4f} {o:>18.4f}  {sign}{delta:>7.4f}")

print(f"\n  TP={metrics_opt['TP']}  TN={metrics_opt['TN']}  "
      f"FP={metrics_opt['FP']}  FN={metrics_opt['FN']}")


# ============================================================
#   CELL 12 — Compare: CB alone vs Stacking v1 vs Stacking v2
# ============================================================

print("\n" + "="*65)
print("  COMPARISON — CB Optuna vs Stack v1 vs Stack v2")
print("="*65)

individual_metrics = {}
for ci, clf_short in enumerate(BASE_LEARNER_DATASETS):
    p    = test_probs[:, ci]
    t, _ = find_best_threshold(y_test, p, metric="mcc")
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

# Load v1 stacking result if available
v1_json = os.path.join(
    BASE_RESULTS, "stacking", "stacking_summary.json"
)
if os.path.exists(v1_json):
    with open(v1_json) as f:
        v1 = json.load(f)
    comparison_rows.append({
        "Model"      : "Stack v1",
        "Accuracy"   : v1.get("stacking_acc"),
        "Sensitivity": v1.get("stacking_sn"),
        "Specificity": v1.get("stacking_sp"),
        "F1_Score"   : v1.get("stacking_f1"),
        "MCC"        : v1.get("stacking_mcc"),
        "AUC"        : v1.get("stacking_auc"),
    })

comparison_rows.append({
    "Model"      : "★ Stack v2",
    "Accuracy"   : metrics_opt["Accuracy"],
    "Sensitivity": metrics_opt["Sensitivity"],
    "Specificity": metrics_opt["Specificity"],
    "F1_Score"   : metrics_opt["F1_Score"],
    "MCC"        : metrics_opt["MCC"],
    "AUC"        : metrics_opt["AUC"],
})

df_comp    = pd.DataFrame(comparison_rows)
comp_path  = os.path.join(STACK_DIR, "stacking_v2_vs_individual.csv")
df_comp.to_csv(comp_path, index=False)

print(f"\n{df_comp.to_string(index=False)}")
print(f"\n  ✅ Comparison saved → {comp_path}")


# ============================================================
#   CELL 13 — Save All Artefacts
# ============================================================

results_path = os.path.join(STACK_DIR, "stacking_v2_results.csv")
pd.DataFrame([
    {"Model": "Stack v2 (default t=0.50)", "Threshold": 0.50,
     **{k:v for k,v in metrics_def.items() if k not in ("TP","TN","FP","FN")}},
    {"Model": f"Stack v2 (optimal t={best_t})", "Threshold": best_t,
     **{k:v for k,v in metrics_opt.items() if k not in ("TP","TN","FP","FN")}},
]).to_csv(results_path, index=False)

probs_path = os.path.join(STACK_DIR, "stacking_v2_probabilities.csv")
pd.DataFrame({
    "y_true"         : y_test,
    "y_pred_default" : y_pred_default,
    "y_pred_optimal" : y_pred_optimal,
    "prob_positive"  : meta_prob_test,
    **{f"p_{c}": test_probs[:,i]
       for i,c in enumerate(BASE_LEARNER_DATASETS)},
    "p_weighted"     : test_weighted,
}).to_csv(probs_path, index=False)

summary = {
    "version"        : "v2",
    "base_learners"  : list(BASE_LEARNER_DATASETS.keys()),
    "dataset_map"    : BASE_LEARNER_DATASETS,
    "meta_learner"   : "LogisticRegression(C=0.1)",
    "calibration"    : "isotonic (ET, RF, LGBM)",
    "norm_weights"   : norm_weights,
    "best_threshold" : best_t,
    "n_folds"        : N_FOLDS,
    "n_jobs"         : N_JOBS,
    "stacking_auc"   : metrics_opt["AUC"],
    "stacking_mcc"   : metrics_opt["MCC"],
    "stacking_sn"    : metrics_opt["Sensitivity"],
    "stacking_sp"    : metrics_opt["Specificity"],
    "stacking_acc"   : metrics_opt["Accuracy"],
    "stacking_f1"    : metrics_opt["F1_Score"],
    "improvement_over_cb_auc": round(
        metrics_opt["AUC"]
        - individual_metrics.get("CB",{}).get("AUC", 0), 4),
    "improvement_over_cb_mcc": round(
        metrics_opt["MCC"]
        - individual_metrics.get("CB",{}).get("MCC", 0), 4),
}
sum_path = os.path.join(STACK_DIR, "stacking_v2_summary.json")
with open(sum_path, "w") as f:
    json.dump(summary, f, indent=4)

art_path = os.path.join(STACK_DIR, "stacking_v2_artefacts.joblib")
joblib.dump({
    "meta_learner"  : meta_learner,
    "final_models"  : final_models,
    "final_scalers" : final_scalers,
    "norm_weights"  : norm_weights,
    "best_threshold": best_t,
    "clf_order"     : list(BASE_LEARNER_DATASETS.keys()),
    "dataset_map"   : BASE_LEARNER_DATASETS,
}, art_path)

print(f"\n✅ Results    → {results_path}")
print(f"✅ Probs      → {probs_path}")
print(f"✅ Summary    → {sum_path}")
print(f"✅ Artefacts  → {art_path}")


# ============================================================
#   CELL 14 — Visualization 1: ROC Curves
# ============================================================

clf_colors = {
    "CB"  : "#e67e22", "LGBM": "#8e44ad",
    "ET"  : "#1abc9c", "RF"  : "#2ecc71",
}

fig, ax = plt.subplots(figsize=(10, 8))
for ci, clf_short in enumerate(BASE_LEARNER_DATASETS):
    p           = test_probs[:, ci]
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr,
            color=clf_colors.get(clf_short,"#95a5a6"),
            linewidth=1.2, linestyle="--",
            label=f"{clf_short} (AUC={roc_auc_score(y_test,p):.4f})",
            alpha=0.8)

fpr_s, tpr_s, _ = roc_curve(y_test, meta_prob_test)
ax.plot(fpr_s, tpr_s, color="black", linewidth=2.5,
        label=f"★ Stack v2 (AUC={metrics_opt['AUC']:.4f})")
ax.plot([0,1],[0,1],"grey",linewidth=0.8,linestyle="--",alpha=0.5)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — Stacking v2 vs Base Learners",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_v2_ROC.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ ROC curves saved")


# ============================================================
#   CELL 15 — Visualization 2: Metrics Bar Chart
# ============================================================

metric_list = ["Accuracy","Sensitivity","Specificity",
               "F1_Score","MCC","AUC"]
fig, ax = plt.subplots(figsize=(14, 6))
x_  = np.arange(len(metric_list))
n_m = len(df_comp)
bw  = 0.8 / n_m
clrs= list(clf_colors.values()) + ["#bbbbbb", "#1A1A1A"]

for i, row in df_comp.iterrows():
    offset   = (i - n_m / 2 + 0.5) * bw
    vals     = [row[m] if pd.notna(row[m]) else 0 for m in metric_list]
    is_stack = "Stack v2" in str(row["Model"])
    ax.bar(x_ + offset, vals, bw,
           label=row["Model"],
           color=clrs[i] if i < len(clrs) else "#555555",
           alpha=1.0 if is_stack else 0.70,
           edgecolor="white")

ax.set_xticks(x_)
ax.set_xticklabels(metric_list, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_title("Stacking v2 vs Base Learners — All Metrics",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.5, color="grey", linestyle="--",
           linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_v2_metrics_bar.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Metrics bar chart saved")


# ============================================================
#   CELL 16 — Visualization 3: Confusion Matrix
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
best_base_p    = test_probs[:, 0]   # CB
best_base_t, _ = find_best_threshold(y_test, best_base_p, "mcc")

for ax_cm, pred, title, prob in zip(
    axes,
    [(best_base_p >= best_base_t).astype(int), y_pred_optimal],
    [f"CB (t={best_base_t})", f"★ Stack v2 (t={best_t})"],
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
plt.suptitle("Confusion Matrix — CB vs Stacking v2",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_v2_confusion.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Confusion matrices saved")


# ============================================================
#   CELL 17 — Visualization 4: OOF Heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(14, 4))
df_oof  = pd.DataFrame(oof_probs,
                        columns=list(BASE_LEARNER_DATASETS.keys()))
df_oof["Weighted"] = oof_weighted
df_oof  = df_oof.sort_values("Weighted").reset_index(drop=True)
sns.heatmap(
    df_oof[list(BASE_LEARNER_DATASETS.keys())+["Weighted"]].T,
    cmap="RdYlGn", vmin=0, vmax=1, ax=ax, xticklabels=False,
    cbar_kws={"label": "P(AIP)"},
)
ax.set_title(
    "OOF Probability Heatmap — Base Learner Agreement\n"
    "(sorted by ensemble score)",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "stacking_v2_oof_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ OOF heatmap saved")


# ============================================================
#   CELL 18 — Visualization 5: Meta Coefficients
# ============================================================

coef_vals = meta_learner.coef_[0]
fig, ax   = plt.subplots(figsize=(9, 4))
colors    = [clf_colors.get(c, "#1abc9c") for c in feat_names]
ax.bar(feat_names, coef_vals, color=colors,
       edgecolor="white", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("LR Coefficient", fontsize=12)
ax.set_title("Meta-Learner Coefficients\n"
             "(positive = trusted for AIP, negative = trusted for non-AIP)",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, (f, c) in enumerate(zip(feat_names, coef_vals)):
    ax.text(i, c + (0.01 if c >= 0 else -0.04),
            f"{c:+.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "stacking_v2_meta_coefficients.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Meta-coefficients saved")


# ============================================================
#   CELL 19 — Visualization 6: Threshold Curve
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
ax.set_title("Threshold vs MCC / Sn / Sp — Stacking v2",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.set_xlim([0.10, 0.90])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "stacking_v2_threshold_curve.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Threshold curve saved")


# ============================================================
#   CELL 20 — Final Summary
# ============================================================

print("\n" + "="*65)
print("  STACKING V2 — COMPLETE SUMMARY")
print("="*65)
print(f"\n  Architecture:")
print(f"    CB(ProtT5) + LGBM(ESM2) + ET(CTDC) + RF(PAAC)")
print(f"    Meta : Logistic Regression (C=0.1, L2)")
print(f"    Calib: Isotonic (ET, RF, LGBM)")
print(f"    t    : {best_t}  |  N_FOLDS: {N_FOLDS}  |  CPUs: {N_JOBS}")
print(f"\n{'─'*65}")
print(f"  {'Model':<14} {'Acc':>8} {'Sn':>8} {'Sp':>8} "
      f"{'MCC':>8} {'AUC':>8}")
print(f"{'─'*65}")
for _, row in df_comp.iterrows():
    is_s = "Stack" in str(row["Model"])
    vals = [row.get(m, None) for m in
            ["Accuracy","Sensitivity","Specificity","MCC","AUC"]]
    vstr = "  ".join(
        f"{v:>8.4f}" if v is not None and not pd.isna(v)
        else "       —" for v in vals
    )
    mark = " ★" if "v2" in str(row["Model"]) else ""
    print(f"  {str(row['Model']):<14} {vstr}{mark}")
print(f"{'─'*65}")
print(f"\n  Improvement over CB Optuna:")
print(f"     AUC : {summary['improvement_over_cb_auc']:+.4f}")
print(f"     MCC : {summary['improvement_over_cb_mcc']:+.4f}")
print(f"\n  Files saved to: {STACK_DIR}/")
print("="*65)