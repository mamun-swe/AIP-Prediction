# ============================================================
#   CELL 1 — Mount Google Drive
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================
#   CELL 2 — Install & Import Libraries
# ============================================================
# !pip install optuna xgboost -q  # Install Optuna & XGBoost for colab

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay
)

print("✅ Libraries loaded")
print(f"   Optuna version  : {optuna.__version__}")


# ============================================================
#   CELL 3 — Configuration
# ============================================================

# DRIVE DATA PATH (ONLY for Google Colab, ignored in local runs)
# FEATURE_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/data/features"
# RESULTS_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/xgb_optuna"
# FIGURES_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/figures/models/xgb_optuna"
# MODELS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/xgb_optuna"
# PARAMS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/xgb_optuna/best_params"

# LOCAL DATA PATH (ONLY for local runs, ignored in Google Colab)
FEATURE_DIR = "../../../../data/features/v2"
RESULTS_DIR = "../../../../results/models/v2/xgb_optuna"
FIGURES_DIR = "../../../../results/figures/models/v2/xgb_optuna"
MODELS_DIR = "../../../../results/models/v2/xgb_optuna"
PARAMS_DIR = "../../../../results/models/v2/xgb_optuna/best_params"

os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)
os.makedirs(PARAMS_DIR,   exist_ok=True)

# ── All 11 feature datasets ──────────────────────────────────
DATASETS = {
    "AAC"     : "AAC_features.csv",
    "DPC"     : "DPC_features.csv",
    "CTDC"    : "CTDC_features.csv",
    "CTDT"    : "CTDT_features.csv",
    "GAAC"    : "GAAC_features.csv",
    "PAAC"    : "PAAC_features.csv",
    "ProtBERT": "ProtBERT_features.csv",
    "ProtT5"  : "ProtT5_features.csv",
    "ESM2"    : "ESM2_features.csv",
    "BioBERT" : "BioBERT_features.csv",
    "ESMC"    : "ESMC_features.csv",
}

# ── Optuna settings ──────────────────────────────────────────
N_TRIALS     = 100
N_CV_FOLDS   = 5
OPTUNA_SEED  = 42

# ── Train/Test split ─────────────────────────────────────────
TEST_SIZE    = 0.30
RANDOM_STATE = 42

# ── Search space description ─────────────────────────────────
#
#  Parameters tuned by Optuna:
#    n_estimators     : number of boosting rounds [50, 500]
#    max_depth        : tree depth per round [2, 10]
#    learning_rate    : step shrinkage [0.005, 0.5] log-uniform
#    subsample        : row sampling per tree [0.5, 1.0]
#    colsample_bytree : column sampling per tree [0.3, 1.0]
#    colsample_bylevel: column sampling per level [0.3, 1.0]
#    min_child_weight : minimum sum of instance weight in child [1, 20]
#    gamma            : min loss reduction for a split [0.0, 5.0]
#    reg_alpha        : L1 regularisation [0.0, 10.0] log-uniform
#    reg_lambda       : L2 regularisation [0.1, 10.0] log-uniform
#    scale_pos_weight : class imbalance weight [1.0, 3.0]
#    grow_policy      : tree growth strategy (depthwise / lossguide)
#
SEARCH_SPACE = {
    "n_estimators"    : "int [50, 500]",
    "max_depth"       : "int [2, 10]",
    "learning_rate"   : "float [0.005, 0.5] log-uniform",
    "subsample"       : "float [0.5, 1.0]",
    "colsample_bytree": "float [0.3, 1.0]",
    "colsample_bylevel": "float [0.3, 1.0]",
    "min_child_weight": "int [1, 20]",
    "gamma"           : "float [0.0, 5.0]",
    "reg_alpha"       : "float [1e-8, 10.0] log-uniform",
    "reg_lambda"      : "float [0.1, 10.0] log-uniform",
    "scale_pos_weight": "float [1.0, 3.0]",
    "grow_policy"     : ["depthwise", "lossguide"],
}

print(f"✅ Config loaded")
print(f"   Datasets    : {len(DATASETS)}")
print(f"   N_TRIALS    : {N_TRIALS}")
print(f"   CV folds    : {N_CV_FOLDS}")
print(f"   Train/Test  : {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
print(f"\n   Search space:")
for k, v in SEARCH_SPACE.items():
    print(f"     {k:<22} : {v}")


# ============================================================
#   CELL 4 — Metric Helper Functions
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute all 6 performance metrics.

    Metrics:
      Accuracy    = (TP + TN) / (TP + TN + FP + FN)
      Sensitivity = TP / (TP + FN)   [Recall for positive class]
      Specificity = TN / (TN + FP)   [Recall for negative class]
      F1 Score    = 2 × (Precision × Recall) / (Precision + Recall)
      MCC         = Matthews Correlation Coefficient
      AUC         = Area Under ROC Curve
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    mcc         = matthews_corrcoef(y_true, y_pred)
    auc         = roc_auc_score(y_true, y_prob)

    return {
        "Accuracy"   : round(accuracy,    4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F1_Score"   : round(f1,          4),
        "MCC"        : round(mcc,         4),
        "AUC"        : round(auc,         4),
        "TP": int(tp), "TN": int(tn),
        "FP": int(fp), "FN": int(fn),
    }


def load_dataset(csv_path):
    """
    Load a feature CSV, auto-detect feature columns and label.
    Expects: seq_id column, feature columns, label column (last).
    """
    df = pd.read_csv(csv_path)

    drop_cols = [c for c in ["seq_id", "sequence", "length"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    return X, y, df.columns[-1], df.shape


# ============================================================
#   CELL 5 — Optuna Objective Function
# ============================================================

def make_objective(X_train, y_train, n_folds, seed):
    """
    Returns an Optuna objective function for XGBoost.

    Objective: maximise mean AUC across stratified k-fold CV
    on the training set.

    XGBoost-specific design:
      - learning_rate and regularisation params use log-uniform
        sampling → better exploration of wide ranges
      - colsample_bylevel tuned independently of colsample_bytree
        for fine-grained feature subsampling control
      - grow_policy: depthwise = standard level-wise growth;
        lossguide = leaf-wise (like LightGBM), can capture
        more complex patterns but may overfit
      - use_label_encoder=False suppresses deprecation warning
      - eval_metric="logloss" set to avoid extra warnings
    """
    def objective(trial):

        params = {
            "n_estimators"     : trial.suggest_int(
                "n_estimators", 50, 500
            ),
            "max_depth"        : trial.suggest_int(
                "max_depth", 2, 10
            ),
            "learning_rate"    : trial.suggest_float(
                "learning_rate", 0.005, 0.5, log=True
            ),
            "subsample"        : trial.suggest_float(
                "subsample", 0.5, 1.0
            ),
            "colsample_bytree" : trial.suggest_float(
                "colsample_bytree", 0.3, 1.0
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.3, 1.0
            ),
            "min_child_weight" : trial.suggest_int(
                "min_child_weight", 1, 20
            ),
            "gamma"            : trial.suggest_float(
                "gamma", 0.0, 5.0
            ),
            "reg_alpha"        : trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda"       : trial.suggest_float(
                "reg_lambda", 0.1, 10.0, log=True
            ),
            "scale_pos_weight" : trial.suggest_float(
                "scale_pos_weight", 1.0, 3.0
            ),
            "grow_policy"      : trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "use_label_encoder": False,
            "eval_metric"      : "logloss",
            "tree_method"      : "hist",
            "n_jobs"           : -1,
            "random_state"     : seed,
        }

        clf = XGBClassifier(**params)
        cv  = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        auc_scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return auc_scores.mean()

    return objective


def extract_best_params(trial_params, random_state):
    """
    Reconstruct the final XGBoost parameter dict from
    Optuna trial params. Adds fixed non-tuned parameters.
    """
    params = trial_params.copy()
    params.update({
        "use_label_encoder": False,
        "eval_metric"      : "logloss",
        "tree_method"      : "hist",
        "n_jobs"           : -1,
        "random_state"     : random_state,
    })
    return params


# ============================================================
#   CELL 6 — Main Training Loop with Optuna
# ============================================================

all_results   = []
best_model    = None
best_scaler   = None
best_name     = ""
best_auc      = -1.0
all_probs     = {}
all_studies   = {}

print("=" * 65)
print("  XGBoost + Optuna — Training on 11 Datasets")
print(f"  Trials per dataset : {N_TRIALS}")
print(f"  CV folds           : {N_CV_FOLDS} (StratifiedKFold)")
print(f"  Sampler            : TPE (Tree-structured Parzen Estimator)")
print("=" * 65)

for ds_name, csv_file in DATASETS.items():

    csv_path = os.path.join(FEATURE_DIR, csv_file)

    if not os.path.exists(csv_path):
        print(f"\n⏭  [{ds_name}] File not found — skipping: {csv_file}")
        continue

    print(f"\n{'─'*60}")
    print(f"  Dataset : {ds_name}  ({csv_file})")

    # ── Load & split ─────────────────────────────────────────
    X, y, label_col, shape = load_dataset(csv_path)
    print(f"  Shape   : {shape}  |  Features: {X.shape[1]}  |  "
          f"Pos: {y.sum()}  Neg: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    print(f"  Train   : {len(X_train)} samples  |  "
          f"Test: {len(X_test)} samples")

    # ── Scale features ───────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Run Optuna ───────────────────────────────────────────
    print(f"  Running Optuna ({N_TRIALS} trials, {N_CV_FOLDS}-fold CV)...")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = TPESampler(seed=OPTUNA_SEED),
        study_name = f"XGB_{ds_name}",
    )
    study.optimize(
        make_objective(X_train, y_train, N_CV_FOLDS, RANDOM_STATE),
        n_trials          = N_TRIALS,
        show_progress_bar = True,
    )

    all_studies[ds_name] = study

    # ── Extract best parameters ───────────────────────────────
    best_params = extract_best_params(
        study.best_trial.params, RANDOM_STATE
    )
    cv_auc = study.best_trial.value

    print(f"\n  ── Best Parameters (trial #{study.best_trial.number}) ──")
    skip = {"use_label_encoder", "eval_metric",
            "tree_method", "n_jobs", "random_state"}
    for k, v in best_params.items():
        if k not in skip:
            print(f"     {k:<22} : {v}")
    print(f"     {'CV AUC':<22} : {cv_auc:.4f}")

    # ── Train final model with best params ────────────────────
    clf = XGBClassifier(**best_params)
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    # ── Predict ──────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # ── Metrics ──────────────────────────────────────────────
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["Dataset"]    = ds_name
    metrics["Features"]   = X.shape[1]
    metrics["CV_AUC"]     = round(cv_auc, 4)
    metrics["Best_Trial"] = study.best_trial.number
    metrics["Train_N"]    = len(X_train)
    metrics["Test_N"]     = len(X_test)

    for k, v in best_params.items():
        if k not in skip:
            metrics[f"param_{k}"] = v
    all_results.append(metrics)

    print(f"\n  ── Test Set Results ─────────────────────────────")
    print(f"  Accuracy    : {metrics['Accuracy']:.4f}")
    print(f"  Sensitivity : {metrics['Sensitivity']:.4f}")
    print(f"  Specificity : {metrics['Specificity']:.4f}")
    print(f"  F1 Score    : {metrics['F1_Score']:.4f}")
    print(f"  MCC         : {metrics['MCC']:.4f}")
    print(f"  AUC (test)  : {metrics['AUC']:.4f}")
    print(f"  AUC (CV)    : {metrics['CV_AUC']:.4f}")
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  "
          f"FP={metrics['FP']}  FN={metrics['FN']}")

    # ── Save probabilities ───────────────────────────────────
    df_probs = pd.DataFrame({
        "y_true"       : y_test,
        "y_pred"       : y_pred,
        "prob_positive": y_prob,
        "prob_negative": 1 - y_prob,
    })
    prob_path = os.path.join(
        RESULTS_DIR, f"{ds_name}_XGB_Optuna_probabilities.csv"
    )
    df_probs.to_csv(prob_path, index=False)

    # ── Save best params as JSON ─────────────────────────────
    params_to_save = {
        k: v for k, v in best_params.items() if k not in skip
    }
    params_to_save.update({
        "cv_auc"     : round(cv_auc, 4),
        "test_auc"   : metrics["AUC"],
        "accuracy"   : metrics["Accuracy"],
        "sensitivity": metrics["Sensitivity"],
        "specificity": metrics["Specificity"],
        "f1_score"   : metrics["F1_Score"],
        "mcc"        : metrics["MCC"],
        "best_trial" : study.best_trial.number,
        "n_trials"   : N_TRIALS,
        "dataset"    : ds_name,
    })
    param_path = os.path.join(
        PARAMS_DIR, f"{ds_name}_XGB_best_params.json"
    )
    with open(param_path, "w") as f:
        json.dump(params_to_save, f, indent=4)

    # ── Also save native XGBoost JSON model ──────────────────
    xgb_json_path = os.path.join(
        MODELS_DIR, f"XGB_Optuna_{ds_name}_model.json"
    )
    clf.save_model(xgb_json_path)

    all_probs[ds_name] = {
        "y_test"     : y_test,
        "y_pred"     : y_pred,
        "y_prob"     : y_prob,
        "clf"        : clf,
        "scaler"     : scaler,
        "best_params": best_params,
        "cv_auc"     : cv_auc,
    }

    # ── Track best model ─────────────────────────────────────
    if metrics["AUC"] > best_auc:
        best_auc    = metrics["AUC"]
        best_name   = ds_name
        best_model  = clf
        best_scaler = scaler

    print(f"\n  ✅ Params saved → {param_path}")
    print(f"  ✅ Probs  saved → {prob_path}")
    print(f"  ✅ Model  saved → {xgb_json_path}")

print(f"\n{'='*65}")
print(f"  ✅ Optuna tuning complete for {len(all_results)} datasets")
print(f"  🏆 Best model: {best_name}  (AUC = {best_auc:.4f})")
print(f"{'='*65}")


# ============================================================
#   CELL 7 — Results Summary Table
# ============================================================

metric_cols = ["Dataset", "CV_AUC", "Accuracy", "Sensitivity",
               "Specificity", "F1_Score", "MCC", "AUC",
               "Best_Trial", "Features"]

df_results = pd.DataFrame(all_results)[metric_cols].sort_values(
    "AUC", ascending=False
).reset_index(drop=True)

df_results.index += 1

print("\n── XGB + Optuna Performance Summary (sorted by Test AUC) ──")
print(df_results.to_string())

summary_path = os.path.join(
    RESULTS_DIR, "XGB_Optuna_all_results_summary.csv"
)
df_results.to_csv(summary_path, index=True, index_label="Rank")
print(f"\n✅ Summary saved → {summary_path}")

df_full   = pd.DataFrame(all_results)
full_path = os.path.join(
    RESULTS_DIR, "XGB_Optuna_full_results_with_params.csv"
)
df_full.to_csv(full_path, index=False)
print(f"✅ Full results (with params) saved → {full_path}")


# ============================================================
#   CELL 8 — Save Best Model
# ============================================================

best_model_path  = os.path.join(
    MODELS_DIR, f"XGB_Optuna_best_model_{best_name}.joblib"
)
best_scaler_path = os.path.join(
    MODELS_DIR, f"XGB_Optuna_best_scaler_{best_name}.joblib"
)
best_xgb_json    = os.path.join(
    MODELS_DIR, f"XGB_Optuna_best_model_{best_name}.json"
)

joblib.dump(best_model,  best_model_path)
joblib.dump(best_scaler, best_scaler_path)
best_model.save_model(best_xgb_json)

best_params_summary = {
    k: v for k, v in all_probs[best_name]["best_params"].items()
    if k not in {"use_label_encoder", "eval_metric",
                 "tree_method", "n_jobs", "random_state"}
}
best_params_summary.update({
    "dataset" : best_name,
    "test_auc": best_auc,
    "cv_auc"  : round(all_probs[best_name]["cv_auc"], 4),
})
best_overall_path = os.path.join(
    MODELS_DIR, "XGB_Optuna_best_overall_params.json"
)
with open(best_overall_path, "w") as f:
    json.dump(best_params_summary, f, indent=4)

print(f"✅ Best model saved")
print(f"   Dataset            : {best_name}")
print(f"   Test AUC           : {best_auc:.4f}")
print(f"   CV  AUC            : {all_probs[best_name]['cv_auc']:.4f}")
print(f"   Model  (.joblib)   : {best_model_path}")
print(f"   Model  (.json)     : {best_xgb_json}")
print(f"   Scaler             : {best_scaler_path}")
print(f"   Best params (JSON) : {best_overall_path}")
print(f"\n   Best hyperparameters:")
skip_print = {"use_label_encoder", "eval_metric",
              "tree_method", "n_jobs", "random_state"}
for k, v in all_probs[best_name]["best_params"].items():
    if k not in skip_print:
        print(f"     {k:<22} : {v}")


# ============================================================
#   CELL 9 — Visualization 1: Metrics Heatmap
# ============================================================

heat_cols = ["Accuracy", "Sensitivity", "Specificity",
             "F1_Score", "MCC", "AUC", "CV_AUC"]
heat_data = df_results.set_index("Dataset")[heat_cols]

fig, ax = plt.subplots(figsize=(15, max(5, len(heat_data) * 0.7)))
sns.heatmap(
    heat_data,
    annot=True, fmt=".4f", cmap="YlGn",
    linewidths=0.5, linecolor="grey",
    vmin=0, vmax=1, ax=ax,
    cbar_kws={"label": "Score"}
)
ax.set_title(
    f"XGBoost + Optuna ({N_TRIALS} trials) — "
    f"Performance Metrics Across All 11 Datasets",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Dataset", fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15,
                   ha="right", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "XGB_Optuna_metrics_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Metrics heatmap saved")


# ============================================================
#   CELL 10 — Visualization 2: Grouped Bar Chart
# ============================================================

plot_cols = ["Accuracy", "Sensitivity", "Specificity",
             "F1_Score", "MCC", "AUC"]

fig, ax = plt.subplots(figsize=(16, 6))
x      = np.arange(len(df_results))
width  = 0.13
colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]

for i, (col, color) in enumerate(zip(plot_cols, colors)):
    offset = (i - len(plot_cols) / 2 + 0.5) * width
    ax.bar(x + offset, df_results[col].values,
           width, label=col, color=color,
           alpha=0.85, edgecolor="white")

ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("XGBoost + Optuna — All Metrics per Dataset",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(df_results["Dataset"],
                   rotation=20, ha="right", fontsize=10)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "XGB_Optuna_grouped_bar_chart.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Grouped bar chart saved")


# ============================================================
#   CELL 11 — Visualization 3: ROC Curves
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))
cmap   = plt.cm.get_cmap("tab10", len(all_probs))
colors = [cmap(i) for i in range(len(all_probs))]

for (ds_name, data), color in zip(all_probs.items(), colors):
    fpr, tpr, _ = roc_curve(data["y_test"], data["y_prob"])
    auc_val     = roc_auc_score(data["y_test"], data["y_prob"])
    lw = 2.5 if ds_name == best_name else 1.2
    ls = "-"  if ds_name == best_name else "--"
    ax.plot(fpr, tpr, color=color, linewidth=lw, linestyle=ls,
            label=f"{ds_name} (AUC={auc_val:.4f})"
                  + (" ★" if ds_name == best_name else ""))

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("XGBoost + Optuna — ROC Curves for All Datasets",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "XGB_Optuna_ROC_curves.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ ROC curves saved")


# ============================================================
#   CELL 12 — Visualization 4: Confusion Matrix (Best Model)
# ============================================================

best_data = all_probs[best_name]
cm        = confusion_matrix(best_data["y_test"], best_data["y_pred"])

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-AIP (0)", "AIP (1)"]
)
disp.plot(cmap="Blues", ax=ax, colorbar=False)
ax.set_title(
    f"Confusion Matrix — Best: {best_name}\n"
    f"(Test AUC={best_auc:.4f}  "
    f"CV AUC={all_probs[best_name]['cv_auc']:.4f})",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         f"XGB_Optuna_confusion_matrix_{best_name}.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Confusion matrix saved ({best_name})")


# ============================================================
#   CELL 13 — Visualization 5: AUC Ranking
#             Shows Test AUC and CV AUC side by side
# ============================================================

fig, ax = plt.subplots(figsize=(13, 5))
sorted_df = df_results.sort_values("AUC", ascending=True)
x         = np.arange(len(sorted_df))
width     = 0.35

ax.barh(x + width / 2, sorted_df["AUC"].values, width,
        label="Test AUC",
        color=["#2ecc71" if n == best_name else "#e67e22"
               for n in sorted_df["Dataset"]],
        edgecolor="white", alpha=0.85)
ax.barh(x - width / 2, sorted_df["CV_AUC"].values, width,
        label="CV AUC",
        color=["#27ae60" if n == best_name else "#d35400"
               for n in sorted_df["Dataset"]],
        edgecolor="white", alpha=0.65)

ax.set_yticks(x)
ax.set_yticklabels(sorted_df["Dataset"], fontsize=10)
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("AUC Score", fontsize=12)
ax.set_title(
    f"XGBoost + Optuna ({N_TRIALS} trials) — "
    f"Test AUC vs CV AUC Ranking",
    fontsize=13, fontweight="bold"
)
ax.set_xlim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "XGB_Optuna_AUC_ranking.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ AUC ranking chart saved")


# ============================================================
#   CELL 14 — Visualization 6: Optimization History
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharey=False)
axes = axes.flatten()

for idx, (ds_name, study) in enumerate(all_studies.items()):
    ax = axes[idx]

    trial_nums = [t.number for t in study.trials
                  if t.value is not None]
    trial_vals = [t.value  for t in study.trials
                  if t.value is not None]

    running_best = []
    cur_best     = -np.inf
    for v in trial_vals:
        cur_best = max(cur_best, v)
        running_best.append(cur_best)

    ax.scatter(trial_nums, trial_vals,
               color="#95a5a6", s=12, alpha=0.5,
               label="Trial AUC", zorder=2)
    ax.plot(trial_nums, running_best,
            "-", color="#e74c3c", linewidth=2,
            label=f"Best: {max(trial_vals):.4f}", zorder=3)
    ax.axhline(max(trial_vals), color="#e74c3c",
               linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title(ds_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Trial Number", fontsize=9)
    ax.set_ylabel("CV AUC", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, N_TRIALS)

for idx in range(len(all_studies), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(
    f"Optuna Optimization History — XGBoost "
    f"({N_TRIALS} trials per dataset)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "XGB_Optuna_optimization_history.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Optimization history plot saved")


# ============================================================
#   CELL 15 — Visualization 7: Parameter Importance (Fanova)
# ============================================================

try:
    from optuna.importance import get_param_importances

    best_study  = all_studies[best_name]
    importances = get_param_importances(best_study)

    param_names  = list(importances.keys())
    param_values = list(importances.values())

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(param_names[::-1], param_values[::-1],
                   color="#e67e22", edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, param_values[::-1]):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Importance (Fanova)", fontsize=12)
    ax.set_title(
        f"Optuna Parameter Importance — {best_name}\n"
        f"(based on {N_TRIALS} trials)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlim(0, max(param_values) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR,
                     f"XGB_Optuna_param_importance_{best_name}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"✅ Parameter importance plot saved ({best_name})")

except Exception as e:
    print(f"ℹ️  Parameter importance skipped: {e}")


# ============================================================
#   CELL 16 — Visualization 8: XGBoost Feature Importance
#             Uses the gain-based importance from the best model
# ============================================================

importances = best_model.get_booster().get_score(
    importance_type="gain"
)

if importances:
    imp_df = pd.DataFrame(
        list(importances.items()), columns=["Feature", "Gain"]
    ).sort_values("Gain", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(imp_df)), imp_df["Gain"].values,
           color="#e67e22", edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(imp_df)))
    ax.set_xticklabels(imp_df["Feature"].values,
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Gain (Average Split Gain)", fontsize=12)
    ax.set_title(
        f"XGBoost — Top 30 Feature Importances (Gain)\n"
        f"Best Dataset: {best_name}  "
        f"(n_estimators="
        f"{all_probs[best_name]['best_params']['n_estimators']})",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR,
                     f"XGB_Optuna_feature_importance_{best_name}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"✅ Feature importance plot saved ({best_name})")
else:
    print("ℹ️  No feature importance data available")


# ============================================================
#   CELL 17 — Best Parameters Table (All Datasets)
# ============================================================

print("\n── Best Hyperparameters per Dataset ──────────────────────")
skip_cols = {"use_label_encoder", "eval_metric",
             "tree_method", "n_jobs", "random_state"}
param_rows = []
for ds_name, data in all_probs.items():
    row = {"Dataset": ds_name}
    row.update({
        k: v for k, v in data["best_params"].items()
        if k not in skip_cols
    })
    row["CV_AUC"]   = round(data["cv_auc"], 4)
    row["Test_AUC"] = round(
        roc_auc_score(data["y_test"], data["y_prob"]), 4
    )
    param_rows.append(row)

df_params = pd.DataFrame(param_rows)
print(df_params.to_string(index=False))

params_table_path = os.path.join(
    PARAMS_DIR, "XGB_Optuna_all_best_params.csv"
)
df_params.to_csv(params_table_path, index=False)
print(f"\n✅ Best params table saved → {params_table_path}")


# ============================================================
#   CELL 18 — Final Summary
# ============================================================

best_row    = df_results[df_results["Dataset"] == best_name].iloc[0]
best_params = all_probs[best_name]["best_params"]

print("=" * 65)
print("  XGBOOST + OPTUNA — FINAL SUMMARY")
print("=" * 65)
print(f"\n  Results saved to       : {RESULTS_DIR}")
print(f"  Figures saved to       : {FIGURES_DIR}")
print(f"  Model saved to         : {MODELS_DIR}")
print(f"  Best params saved to   : {PARAMS_DIR}")
print(f"\n{'─'*65}")
print(f"  {'Dataset':<12} {'CV_AUC':>8} {'Acc':>8} {'Sn':>8} "
      f"{'Sp':>8} {'F1':>8} {'MCC':>8} {'AUC':>8}")
print(f"{'─'*65}")
for _, row in df_results.iterrows():
    marker = " ★" if row["Dataset"] == best_name else ""
    print(f"  {row['Dataset']:<12} {row['CV_AUC']:>8.4f} "
          f"{row['Accuracy']:>8.4f} {row['Sensitivity']:>8.4f} "
          f"{row['Specificity']:>8.4f} {row['F1_Score']:>8.4f} "
          f"{row['MCC']:>8.4f} {row['AUC']:>8.4f}{marker}")
print(f"{'─'*65}")
print(f"\n  🏆 Best Dataset   : {best_name}")
print(f"     Test AUC       : {best_auc:.4f}")
print(f"     CV  AUC        : {all_probs[best_name]['cv_auc']:.4f}")
print(f"     Accuracy       : {best_row['Accuracy']:.4f}")
print(f"     Sensitivity    : {best_row['Sensitivity']:.4f}")
print(f"     Specificity    : {best_row['Specificity']:.4f}")
print(f"     F1 Score       : {best_row['F1_Score']:.4f}")
print(f"     MCC            : {best_row['MCC']:.4f}")
print(f"\n  Best hyperparameters ({best_name}):")
for k, v in best_params.items():
    if k not in skip_print:
        print(f"     {k:<22} : {v}")
print(f"\n  Optuna settings:")
print(f"     Sampler        : TPE (Tree-structured Parzen Estimator)")
print(f"     Trials         : {N_TRIALS}")
print(f"     CV folds       : {N_CV_FOLDS} (StratifiedKFold)")
print(f"     Objective      : Maximise mean CV AUC")
print(f"\n  Saved files:")
print(f"     Per-dataset JSON params  : {PARAMS_DIR}/")
print(f"     All-params CSV           : {params_table_path}")
print(f"     Full results CSV         : {full_path}")
print(f"     Per-dataset probs CSV    : {RESULTS_DIR}/")
print(f"     Best model (.joblib)     : {best_model_path}")
print(f"     Best model (.json)       : {best_xgb_json}")
print(f"     Per-dataset .json models : {MODELS_DIR}/")
print("=" * 65)