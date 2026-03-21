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

from sklearn.tree import DecisionTreeClassifier
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
# RESULTS_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/dt_optuna"
# FIGURES_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/figures/dt_optuna"
# MODELS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/models/dt_optuna"
# PARAMS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/dt_optuna/best_params"

# LOCAL DATA PATH (ONLY for local runs, ignored in Google Colab)
FEATURE_DIR  = "../../../data/features"
RESULTS_DIR  = "../../../results/models/dt_optuna"
FIGURES_DIR  = "../../../results/figures/models/dt_optuna"
MODELS_DIR   = "../../../results/models/dt_optuna"
PARAMS_DIR   = "../../../results/models/dt_optuna/best_params"


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
N_TRIALS     = 100     # number of Optuna trials per dataset
N_CV_FOLDS   = 5       # stratified k-fold inside objective
OPTUNA_SEED  = 42

# ── Train/Test split ─────────────────────────────────────────
TEST_SIZE    = 0.30
RANDOM_STATE = 42

# ── Search space description (for reference) ─────────────────
SEARCH_SPACE = {
    "criterion"        : ["gini", "entropy", "log_loss"],
    "max_depth"        : "int [3, 30] or None",
    "min_samples_split": "int [2, 20]",
    "min_samples_leaf" : "int [1, 20]",
    "max_features"     : ["sqrt", "log2", None],
    "splitter"         : ["best", "random"],
    "class_weight"     : ["balanced", None],
    "ccp_alpha"        : "float [0.0, 0.05]",
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
    Returns an Optuna objective function closed over the training
    data. Each trial samples a different hyperparameter combination
    and evaluates it via stratified k-fold CV on the training set.

    Objective: maximise mean AUC across k folds.

    Separating CV from the held-out test set ensures that Optuna
    selects parameters that generalise, not ones that overfit to
    a single split.
    """
    def objective(trial):

        # ── Sample max_depth ─────────────────────────────────
        use_none_depth = trial.suggest_categorical(
            "max_depth_none", [True, False]
        )
        if use_none_depth:
            max_depth = None
        else:
            max_depth = trial.suggest_int("max_depth", 3, 30)

        params = {
            "criterion"        : trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "splitter"         : trial.suggest_categorical(
                "splitter", ["best", "random"]
            ),
            "max_depth"        : max_depth,
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            "min_samples_leaf" : trial.suggest_int(
                "min_samples_leaf", 1, 20
            ),
            "max_features"     : trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "class_weight"     : trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "ccp_alpha"        : trial.suggest_float(
                "ccp_alpha", 0.0, 0.05
            ),
            "random_state"     : seed,
        }

        clf = DecisionTreeClassifier(**params)
        cv  = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        auc_scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return auc_scores.mean()

    return objective


def extract_best_params(best_trial_params, random_state):
    """
    Reconstruct the final parameter dict from Optuna trial params.
    Handles the max_depth_none conditional encoding.
    """
    params = best_trial_params.copy()

    use_none = params.pop("max_depth_none", True)
    if not use_none:
        params["max_depth"] = params.get("max_depth", None)
    else:
        params.pop("max_depth", None)
        params["max_depth"] = None

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
print("  Decision Tree + Optuna — Training on 11 Datasets")
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
        study_name = f"DT_{ds_name}",
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
    for k, v in best_params.items():
        if k != "random_state":
            print(f"     {k:<22} : {v}")
    print(f"     {'CV AUC':<22} : {cv_auc:.4f}")

    # ── Train final model with best parameters ────────────────
    clf = DecisionTreeClassifier(**best_params)
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

    # Store best params as individual result columns
    for k, v in best_params.items():
        if k != "random_state":
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
        RESULTS_DIR, f"{ds_name}_DT_Optuna_probabilities.csv"
    )
    df_probs.to_csv(prob_path, index=False)

    # ── Save best params as JSON ─────────────────────────────
    params_to_save = {
        k: (str(v) if v is None else v)
        for k, v in best_params.items()
    }
    params_to_save.update({
        "cv_auc"    : round(cv_auc, 4),
        "test_auc"  : metrics["AUC"],
        "accuracy"  : metrics["Accuracy"],
        "sensitivity": metrics["Sensitivity"],
        "specificity": metrics["Specificity"],
        "f1_score"  : metrics["F1_Score"],
        "mcc"       : metrics["MCC"],
        "best_trial": study.best_trial.number,
        "n_trials"  : N_TRIALS,
        "dataset"   : ds_name,
    })
    param_path = os.path.join(
        PARAMS_DIR, f"{ds_name}_DT_best_params.json"
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

print("\n── DT + Optuna Performance Summary (sorted by Test AUC) ──")
print(df_results.to_string())

summary_path = os.path.join(
    RESULTS_DIR, "DT_Optuna_all_results_summary.csv"
)
df_results.to_csv(summary_path, index=True, index_label="Rank")
print(f"\n✅ Summary saved → {summary_path}")

# Save full table including all best param columns
df_full = pd.DataFrame(all_results)
full_path = os.path.join(
    RESULTS_DIR, "DT_Optuna_full_results_with_params.csv"
)
df_full.to_csv(full_path, index=False)
print(f"✅ Full results (with params) saved → {full_path}")


# ============================================================
#   CELL 8 — Save Best Model
# ============================================================

best_model_path  = os.path.join(
    MODELS_DIR, f"DT_Optuna_best_model_{best_name}.joblib"
)
best_scaler_path = os.path.join(
    MODELS_DIR, f"DT_Optuna_best_scaler_{best_name}.joblib"
)

joblib.dump(best_model,  best_model_path)
joblib.dump(best_scaler, best_scaler_path)

# Save best overall param summary
best_params_summary = {
    k: (str(v) if v is None else v)
    for k, v in all_probs[best_name]["best_params"].items()
}
best_params_summary.update({
    "dataset"  : best_name,
    "test_auc" : best_auc,
    "cv_auc"   : round(all_probs[best_name]["cv_auc"], 4),
})
best_overall_path = os.path.join(
    MODELS_DIR, "DT_Optuna_best_overall_params.json"
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
    if k != "random_state":
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
    f"Decision Tree + Optuna ({N_TRIALS} trials) — "
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
                         "DT_Optuna_metrics_heatmap.png"),
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
ax.set_title("Decision Tree + Optuna — All Metrics per Dataset",
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
                         "DT_Optuna_grouped_bar_chart.png"),
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

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5,
        label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Decision Tree + Optuna — ROC Curves for All Datasets",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "DT_Optuna_ROC_curves.png"),
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
                         f"DT_Optuna_confusion_matrix_{best_name}.png"),
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
        color=["#2ecc71" if n == best_name else "#3498db"
               for n in sorted_df["Dataset"]],
        edgecolor="white", alpha=0.85)
ax.barh(x - width / 2, sorted_df["CV_AUC"].values, width,
        label="CV AUC",
        color=["#27ae60" if n == best_name else "#2980b9"
               for n in sorted_df["Dataset"]],
        edgecolor="white", alpha=0.65)

ax.set_yticks(x)
ax.set_yticklabels(sorted_df["Dataset"], fontsize=10)
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("AUC Score", fontsize=12)
ax.set_title(
    f"Decision Tree + Optuna ({N_TRIALS} trials) — "
    f"Test AUC vs CV AUC Ranking",
    fontsize=13, fontweight="bold"
)
ax.set_xlim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "DT_Optuna_AUC_ranking.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ AUC ranking chart saved")


# ============================================================
#   CELL 14 — Visualization 6: Optimization History
#             AUC improvement curve over 100 trials per dataset
# ============================================================

fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharey=False)
axes = axes.flatten()

for idx, (ds_name, study) in enumerate(all_studies.items()):
    ax = axes[idx]

    trial_nums  = [t.number for t in study.trials
                   if t.value is not None]
    trial_vals  = [t.value  for t in study.trials
                   if t.value is not None]

    # Running best AUC
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
    f"Optuna Optimization History — Decision Tree "
    f"({N_TRIALS} trials per dataset)",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "DT_Optuna_optimization_history.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Optimization history plot saved")


# ============================================================
#   CELL 15 — Visualization 7: Parameter Importance
#             Which hyperparameters mattered most for AUC?
# ============================================================

try:
    from optuna.importance import get_param_importances

    best_study  = all_studies[best_name]
    importances = get_param_importances(best_study)

    param_names  = list(importances.keys())
    param_values = list(importances.values())
    clean_names  = [
        n.replace("max_depth_none", "max_depth (None?)")
        for n in param_names
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(clean_names[::-1], param_values[::-1],
                   color="#3498db", edgecolor="white", alpha=0.85)
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
                     f"DT_Optuna_param_importance_{best_name}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"✅ Parameter importance plot saved ({best_name})")

except Exception as e:
    print(f"ℹ️  Parameter importance skipped: {e}")


# ============================================================
#   CELL 16 — Best Parameters Table (All Datasets)
# ============================================================

print("\n── Best Hyperparameters per Dataset ──────────────────────")
param_rows = []
for ds_name, data in all_probs.items():
    row = {"Dataset": ds_name}
    row.update({k: v for k, v in data["best_params"].items()
                if k != "random_state"})
    row["CV_AUC"]   = round(data["cv_auc"], 4)
    row["Test_AUC"] = round(
        roc_auc_score(data["y_test"], data["y_prob"]), 4
    )
    param_rows.append(row)

df_params = pd.DataFrame(param_rows)
print(df_params.to_string(index=False))

params_table_path = os.path.join(
    PARAMS_DIR, "DT_Optuna_all_best_params.csv"
)
df_params.to_csv(params_table_path, index=False)
print(f"\n✅ Best params table saved → {params_table_path}")


# ============================================================
#   CELL 17 — Final Summary
# ============================================================

best_row    = df_results[df_results["Dataset"] == best_name].iloc[0]
best_params = all_probs[best_name]["best_params"]

print("=" * 65)
print("  DECISION TREE + OPTUNA — FINAL SUMMARY")
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
    if k != "random_state":
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
print(f"     Best model               : {best_model_path}")
print("=" * 65)