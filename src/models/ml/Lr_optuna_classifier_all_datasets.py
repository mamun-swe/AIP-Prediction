# ============================================================
#   CELL 1 — Mount Google Drive
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================
#   CELL 2 — Install & Import Libraries
# ============================================================
# !pip install optuna -q  # Install Optuna for colab

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

from sklearn.linear_model import LogisticRegression
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
print(f"   Optuna version : {optuna.__version__}")


# ============================================================
#   CELL 3 — Configuration
# ============================================================

# DRIVE DATA PATH (ONLY for Google Colab, ignored in local runs)
# FEATURE_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/data/features"
# RESULTS_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/lr_optuna"
# FIGURES_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/figures/models/lr_optuna"
# MODELS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/lr_optuna"
# PARAMS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/lr_optuna/best_params"

# LOCAL DATA PATH (ONLY for local runs, ignored in Google Colab)
FEATURE_DIR  = "../../../data/features"
RESULTS_DIR  = "../../../results/models/lr_optuna"
FIGURES_DIR  = "../../../results/figures/models/lr_optuna"
MODELS_DIR   = "../../../results/models/lr_optuna"
PARAMS_DIR   = "../../../results/models/lr_optuna/best_params"

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

# ── Two bugs fixed in this version ──────────────────────────
#
#  BUG 1: CategoricalDistribution dynamic value space error
#    Cause  : "solver" had different choices per penalty
#    Fix    : Always sample solver from ALL_SOLVERS (constant),
#             then remap incompatible choices via FALLBACK_SOLVER
#
#  BUG 2: InvalidParameterError — penalty='none' invalid
#    Cause  : The STRING "none" is not a valid sklearn penalty.
#             sklearn only accepts: 'l1', 'l2', 'elasticnet', None
#             where None is the Python object, not the string.
#    Fix    : Remove "none" from penalty choices completely.
#             Use only ['l1', 'l2', 'elasticnet'] — these cover
#             the full regularisation space including no-penalty
#             behaviour (very large C achieves near-zero reg).
#
# ── Solver compatibility map ─────────────────────────────────
ALL_SOLVERS = ["lbfgs", "liblinear", "saga", "newton-cg"]

VALID_SOLVERS = {
    "l1"        : {"liblinear", "saga"},
    "l2"        : {"lbfgs", "liblinear", "saga", "newton-cg"},
    "elasticnet": {"saga"},
}

FALLBACK_SOLVER = {
    "l1"        : "saga",
    "l2"        : "lbfgs",
    "elasticnet": "saga",
}

# ── Search space description ─────────────────────────────────
SEARCH_SPACE = {
    "penalty"      : ["l1", "l2", "elasticnet"],
    "C"            : "float [1e-4, 1e4] log-uniform",
    "solver"       : f"sampled from {ALL_SOLVERS} → remapped per penalty",
    "l1_ratio"     : "float [0.0, 1.0]  (elasticnet only)",
    "max_iter"     : "int [200, 5000]",
    "class_weight" : ["balanced", None],
    "fit_intercept": [True, False],
    "tol"          : "float [1e-6, 1e-2] log-uniform",
}

print(f"✅ Config loaded")
print(f"   Datasets    : {len(DATASETS)}")
print(f"   N_TRIALS    : {N_TRIALS}")
print(f"   CV folds    : {N_CV_FOLDS}")
print(f"   Train/Test  : {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
print(f"\n   Search space:")
for k, v in SEARCH_SPACE.items():
    print(f"     {k:<16} : {v}")


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
#   CELL 5 — Optuna Objective Function  (FULLY FIXED)
# ============================================================

def make_objective(X_train, y_train, n_folds, seed):
    """
    Returns an Optuna objective function for Logistic Regression.

    Objective: maximise mean AUC across stratified k-fold CV.

    ✅ FIX 1 — CategoricalDistribution dynamic value space:
       solver always sampled from the full ALL_SOLVERS list
       (constant 4 choices every trial), then remapped if
       incompatible with the chosen penalty.

    ✅ FIX 2 — InvalidParameterError penalty='none':
       Sklearn LogisticRegression only accepts:
           'l1', 'l2', 'elasticnet', or Python None
       The STRING "none" is NOT valid.
       Solution: remove 'none' from penalty choices entirely.
       Use C with a very large value (e.g. 1e4) to approximate
       no regularisation when needed.
    """
    def objective(trial):

        # ── penalty: 3 valid sklearn values only ─────────────
        penalty = trial.suggest_categorical(
            "penalty", ["l1", "l2", "elasticnet"]
        )

        # ── solver: ALWAYS from full list (BUG 1 fix) ────────
        solver_raw = trial.suggest_categorical(
            "solver", ALL_SOLVERS
        )
        solver = solver_raw \
            if solver_raw in VALID_SOLVERS[penalty] \
            else FALLBACK_SOLVER[penalty]

        # ── l1_ratio: elasticnet only ─────────────────────────
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) \
            if penalty == "elasticnet" else None

        # ── C: log-uniform across wide range ─────────────────
        C = trial.suggest_float("C", 1e-4, 1e4, log=True)

        params = {
            "penalty"      : penalty,
            "C"            : C,
            "solver"       : solver,
            "l1_ratio"     : l1_ratio,
            "max_iter"     : trial.suggest_int(
                "max_iter", 200, 5000
            ),
            "class_weight" : trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "fit_intercept": trial.suggest_categorical(
                "fit_intercept", [True, False]
            ),
            "tol"          : trial.suggest_float(
                "tol", 1e-6, 1e-2, log=True
            ),
            "n_jobs"       : -1,
            "random_state" : seed,
        }

        clf = LogisticRegression(**params)
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
    Reconstruct the final LR parameter dict from Optuna trial
    params. Applies the same solver remapping as the objective.
    """
    params  = trial_params.copy()
    penalty = params.get("penalty", "l2")

    # Remap solver if incompatible with penalty
    solver_raw    = params.get("solver", "lbfgs")
    params["solver"] = solver_raw \
        if solver_raw in VALID_SOLVERS.get(penalty, set()) \
        else FALLBACK_SOLVER.get(penalty, "lbfgs")

    # elasticnet must use saga
    if penalty == "elasticnet":
        params["solver"] = "saga"

    # l1_ratio only for elasticnet
    if penalty != "elasticnet":
        params["l1_ratio"] = None

    params["n_jobs"]       = -1
    params["random_state"] = random_state
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
print("  Logistic Regression + Optuna — Training on 11 Datasets")
print(f"  Trials per dataset : {N_TRIALS}")
print(f"  CV folds           : {N_CV_FOLDS} (StratifiedKFold)")
print(f"  Sampler            : TPE (Tree-structured Parzen Estimator)")
print(f"  Penalty choices    : ['l1', 'l2', 'elasticnet']  (no 'none')")
print(f"  Solver choices     : {ALL_SOLVERS}  → remapped per penalty")
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
        study_name = f"LR_{ds_name}",
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
    skip_print = {"n_jobs", "random_state"}
    for k, v in best_params.items():
        if k not in skip_print:
            print(f"     {k:<16} : {v}")
    print(f"     {'CV AUC':<16} : {cv_auc:.4f}")

    # ── Train final model ─────────────────────────────────────
    clf = LogisticRegression(**best_params)
    clf.fit(X_train, y_train)

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
        if k not in skip_print:
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
        RESULTS_DIR, f"{ds_name}_LR_Optuna_probabilities.csv"
    )
    df_probs.to_csv(prob_path, index=False)

    # ── Save best params as JSON ─────────────────────────────
    params_to_save = {
        k: (str(v) if v is None else v)
        for k, v in best_params.items()
        if k not in skip_print
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
        PARAMS_DIR, f"{ds_name}_LR_best_params.json"
    )
    with open(param_path, "w") as f:
        json.dump(params_to_save, f, indent=4)

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

print("\n── LR + Optuna Performance Summary (sorted by Test AUC) ──")
print(df_results.to_string())

summary_path = os.path.join(
    RESULTS_DIR, "LR_Optuna_all_results_summary.csv"
)
df_results.to_csv(summary_path, index=True, index_label="Rank")
print(f"\n✅ Summary saved → {summary_path}")

df_full   = pd.DataFrame(all_results)
full_path = os.path.join(
    RESULTS_DIR, "LR_Optuna_full_results_with_params.csv"
)
df_full.to_csv(full_path, index=False)
print(f"✅ Full results (with params) saved → {full_path}")


# ============================================================
#   CELL 8 — Save Best Model
# ============================================================

best_model_path  = os.path.join(
    MODELS_DIR, f"LR_Optuna_best_model_{best_name}.joblib"
)
best_scaler_path = os.path.join(
    MODELS_DIR, f"LR_Optuna_best_scaler_{best_name}.joblib"
)

joblib.dump(best_model,  best_model_path)
joblib.dump(best_scaler, best_scaler_path)

best_params_summary = {
    k: (str(v) if v is None else v)
    for k, v in all_probs[best_name]["best_params"].items()
    if k not in {"n_jobs", "random_state"}
}
best_params_summary.update({
    "dataset" : best_name,
    "test_auc": best_auc,
    "cv_auc"  : round(all_probs[best_name]["cv_auc"], 4),
})
best_overall_path = os.path.join(
    MODELS_DIR, "LR_Optuna_best_overall_params.json"
)
with open(best_overall_path, "w") as f:
    json.dump(best_params_summary, f, indent=4)

print(f"✅ Best model saved")
print(f"   Dataset          : {best_name}")
print(f"   Test AUC         : {best_auc:.4f}")
print(f"   CV  AUC          : {all_probs[best_name]['cv_auc']:.4f}")
print(f"   Model            : {best_model_path}")
print(f"   Scaler           : {best_scaler_path}")
print(f"   Best params JSON : {best_overall_path}")
print(f"\n   Best hyperparameters:")
for k, v in all_probs[best_name]["best_params"].items():
    if k not in {"n_jobs", "random_state"}:
        print(f"     {k:<16} : {v}")


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
    f"Logistic Regression + Optuna ({N_TRIALS} trials) — "
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
                         "LR_Optuna_metrics_heatmap.png"),
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
ax.set_title("Logistic Regression + Optuna — All Metrics per Dataset",
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
                         "LR_Optuna_grouped_bar_chart.png"),
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
    bp = data["best_params"]
    ax.plot(
        fpr, tpr, color=color, linewidth=lw, linestyle=ls,
        label=f"{ds_name} [{bp['penalty']}] "
              f"C={bp['C']:.2e} (AUC={auc_val:.4f})"
              + (" ★" if ds_name == best_name else "")
    )

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Logistic Regression + Optuna — ROC Curves",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "LR_Optuna_ROC_curves.png"),
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
bp = best_data["best_params"]
ax.set_title(
    f"Confusion Matrix — Best: {best_name}\n"
    f"penalty={bp['penalty']}  C={bp['C']:.2e}  "
    f"solver={bp['solver']}\n"
    f"(Test AUC={best_auc:.4f}  "
    f"CV AUC={best_data['cv_auc']:.4f})",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         f"LR_Optuna_confusion_matrix_{best_name}.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Confusion matrix saved ({best_name})")


# ============================================================
#   CELL 13 — Visualization 5: AUC Ranking
# ============================================================

fig, ax = plt.subplots(figsize=(13, 5))
sorted_df = df_results.sort_values("AUC", ascending=True)
x         = np.arange(len(sorted_df))
width     = 0.35

ax.barh(x + width / 2, sorted_df["AUC"].values, width,
        label="Test AUC",
        color=["#2ecc71" if n == best_name else "#c0392b"
               for n in sorted_df["Dataset"]],
        edgecolor="white", alpha=0.85)
ax.barh(x - width / 2, sorted_df["CV_AUC"].values, width,
        label="CV AUC",
        color=["#27ae60" if n == best_name else "#922b21"
               for n in sorted_df["Dataset"]],
        edgecolor="white", alpha=0.65)

for i, ds_name in enumerate(sorted_df["Dataset"]):
    bp  = all_probs[ds_name]["best_params"]
    auc = sorted_df.loc[sorted_df["Dataset"] == ds_name,
                        "AUC"].values[0]
    ax.text(auc + 0.005, i + width / 2,
            f"{auc:.4f}  [{bp['penalty']}] "
            f"C={bp['C']:.2e}",
            va="center", fontsize=9)

ax.set_yticks(x)
ax.set_yticklabels(sorted_df["Dataset"], fontsize=10)
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("AUC Score", fontsize=12)
ax.set_title(
    f"Logistic Regression + Optuna ({N_TRIALS} trials) — "
    f"Test AUC vs CV AUC Ranking",
    fontsize=13, fontweight="bold"
)
ax.set_xlim(0, 1.22)
ax.legend(fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "LR_Optuna_AUC_ranking.png"),
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

    bp = all_probs.get(ds_name, {}).get("best_params", {})
    if bp:
        ax.set_xlabel(
            f"[{bp.get('penalty','?')}]  "
            f"C={bp.get('C', 0):.2e}  "
            f"{bp.get('solver','?')}",
            fontsize=8
        )

    ax.set_title(ds_name, fontsize=11, fontweight="bold")
    ax.set_ylabel("CV AUC", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, N_TRIALS)

for idx in range(len(all_studies), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(
    f"Optuna Optimization History — Logistic Regression "
    f"({N_TRIALS} trials per dataset)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "LR_Optuna_optimization_history.png"),
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

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(param_names[::-1], param_values[::-1],
                   color="#c0392b", edgecolor="white", alpha=0.85)
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
                     f"LR_Optuna_param_importance_{best_name}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"✅ Parameter importance plot saved ({best_name})")

except Exception as e:
    print(f"ℹ️  Parameter importance skipped: {e}")


# ============================================================
#   CELL 16 — Visualization 8: Signed Coefficients (Best Model)
# ============================================================

coef     = best_model.coef_[0]
n_show   = min(30, len(coef))
n_side   = n_show // 2

top_pos_idx = np.argsort(coef)[::-1][:n_side]
top_neg_idx = np.argsort(coef)[:n_side]
top_idx     = np.concatenate([top_pos_idx, top_neg_idx])
top_coef    = coef[top_idx]
top_labels  = [f"F{i}" for i in top_idx]
coef_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_coef]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(top_coef)), top_coef,
       color=coef_colors, edgecolor="white", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(top_coef)))
ax.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Feature Index", fontsize=12)
ax.set_ylabel("Coefficient Value", fontsize=12)
bp_best = all_probs[best_name]["best_params"]
ax.set_title(
    f"Logistic Regression — Top {n_show} Signed Feature Coefficients\n"
    f"Best Dataset: {best_name}  "
    f"penalty={bp_best['penalty']}  "
    f"C={bp_best['C']:.2e}  "
    f"solver={bp_best['solver']}",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71",
          label="Positive coeff → pushes toward AIP (1)"),
    Patch(facecolor="#e74c3c",
          label="Negative coeff → pushes toward Non-AIP (0)"),
]
ax.legend(handles=legend_elements, fontsize=10)
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR,
                 f"LR_Optuna_coefficients_{best_name}.png"),
    dpi=150, bbox_inches="tight"
)
plt.show()
print(f"✅ Signed coefficient plot saved ({best_name})")


# ============================================================
#   CELL 17 — Visualization 9: Penalty & Solver Distribution
# ============================================================

penalty_counts = {}
solver_counts  = {}
for ds_name, data in all_probs.items():
    pen = data["best_params"].get("penalty", "?")
    sol = data["best_params"].get("solver",  "?")
    penalty_counts[pen] = penalty_counts.get(pen, 0) + 1
    solver_counts[sol]  = solver_counts.get(sol,  0) + 1

pen_colors = {
    "l1": "#3498db", "l2": "#2ecc71", "elasticnet": "#e67e22"
}
sol_colors = {
    "lbfgs": "#1abc9c", "liblinear": "#e74c3c",
    "saga": "#f39c12",  "newton-cg": "#8e44ad"
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].pie(
    list(penalty_counts.values()),
    labels    = list(penalty_counts.keys()),
    colors    = [pen_colors.get(k, "#95a5a6") for k in penalty_counts],
    autopct   = "%1.0f%%",
    startangle= 140,
    wedgeprops= dict(edgecolor="white", linewidth=1.5),
)
axes[0].set_title("Penalty selected\nacross 11 datasets",
                  fontsize=12, fontweight="bold")

axes[1].pie(
    list(solver_counts.values()),
    labels    = list(solver_counts.keys()),
    colors    = [sol_colors.get(k, "#95a5a6") for k in solver_counts],
    autopct   = "%1.0f%%",
    startangle= 140,
    wedgeprops= dict(edgecolor="white", linewidth=1.5),
)
axes[1].set_title("Solver selected\nacross 11 datasets",
                  fontsize=12, fontweight="bold")

plt.suptitle(
    "Logistic Regression + Optuna — Penalty & Solver Distribution",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "LR_Optuna_penalty_solver_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Penalty & solver distribution plot saved")


# ============================================================
#   CELL 18 — Best Parameters Table (All Datasets)
# ============================================================

print("\n── Best Hyperparameters per Dataset ──────────────────────")
param_rows = []
for ds_name, data in all_probs.items():
    row = {"Dataset": ds_name}
    row.update({
        k: v for k, v in data["best_params"].items()
        if k not in {"n_jobs", "random_state"}
    })
    row["CV_AUC"]   = round(data["cv_auc"], 4)
    row["Test_AUC"] = round(
        roc_auc_score(data["y_test"], data["y_prob"]), 4
    )
    param_rows.append(row)

df_params = pd.DataFrame(param_rows)
print(df_params.to_string(index=False))

params_table_path = os.path.join(
    PARAMS_DIR, "LR_Optuna_all_best_params.csv"
)
df_params.to_csv(params_table_path, index=False)
print(f"\n✅ Best params table saved → {params_table_path}")


# ============================================================
#   CELL 19 — Final Summary
# ============================================================

best_row    = df_results[df_results["Dataset"] == best_name].iloc[0]
best_params = all_probs[best_name]["best_params"]

print("=" * 65)
print("  LOGISTIC REGRESSION + OPTUNA — FINAL SUMMARY")
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
    if k not in {"n_jobs", "random_state"}:
        print(f"     {k:<16} : {v}")
print(f"\n  Penalty distribution:")
for pen, cnt in sorted(penalty_counts.items(), key=lambda x: -x[1]):
    print(f"     {pen:<14} : {cnt} dataset(s)")
print(f"\n  Solver distribution:")
for sol, cnt in sorted(solver_counts.items(), key=lambda x: -x[1]):
    print(f"     {sol:<14} : {cnt} dataset(s)")
print(f"\n  Bugs fixed:")
print(f"     ✅ Bug 1: solver always sampled from {ALL_SOLVERS}")
print(f"              incompatible choices remapped via FALLBACK_SOLVER")
print(f"     ✅ Bug 2: penalty='none' (string) removed — only")
print(f"              'l1', 'l2', 'elasticnet' used (sklearn valid values)")
print(f"\n  Optuna settings:")
print(f"     Sampler : TPE  |  Trials : {N_TRIALS}  "
      f"|  CV folds : {N_CV_FOLDS}  |  Objective : max CV AUC")
print(f"\n  Saved files:")
print(f"     Per-dataset JSON params  : {PARAMS_DIR}/")
print(f"     All-params CSV           : {params_table_path}")
print(f"     Full results CSV         : {full_path}")
print(f"     Per-dataset probs CSV    : {RESULTS_DIR}/")
print(f"     Best model               : {best_model_path}")
print("=" * 65)