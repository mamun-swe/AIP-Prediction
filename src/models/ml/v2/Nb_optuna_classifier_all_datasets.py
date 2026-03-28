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

from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import MinMaxScaler
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
# RESULTS_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/nb_optuna"
# FIGURES_DIR  = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/figures/models/nb_optuna"
# MODELS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/nb_optuna"
# PARAMS_DIR   = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results/models/nb_optuna/best_params"

# LOCAL DATA PATH (ONLY for local runs, ignored in Google Colab)
FEATURE_DIR = "../../../../data/features/v2"
RESULTS_DIR = "../../../../results/models/v2/nb_optuna"
FIGURES_DIR = "../../../../results/figures/models/v2/nb_optuna"
MODELS_DIR = "../../../../results/models/v2/nb_optuna"
PARAMS_DIR = "../../../../results/models/v2/nb_optuna/best_params"

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

# ── Scaler: MinMaxScaler for ALL NB variants ─────────────────
#
#  Why MinMaxScaler (not StandardScaler)?
#    GaussianNB   : works with any real values — MinMaxScaler OK
#    ComplementNB : requires NON-NEGATIVE features
#    BernoulliNB  : requires NON-NEGATIVE features
#    MinMaxScaler → [0, 1] always non-negative ✅
#    StandardScaler can produce negative values → breaks CNB/BNB
#
# ── NB variant colour map (used across multiple plots) ───────
NB_COLORS = {
    "gaussian"  : "#3498db",
    "complement": "#e67e22",
    "bernoulli" : "#9b59b6",
}
NB_LINE_STYLES = {
    "gaussian"  : "-",
    "complement": "--",
    "bernoulli" : "-.",
}

# ── Search space description ─────────────────────────────────
#
#  Three NB variants explored:
#
#  GaussianNB   — assumes Gaussian (normal) distribution per feature
#    var_smoothing : [1e-12, 1e-1] prevents zero-variance dominance
#
#  ComplementNB — uses complement class statistics, better for imbalance
#    alpha         : Laplace smoothing [1e-3, 10.0]
#    norm          : normalise weights [True / False]
#
#  BernoulliNB  — binary feature model, uses binarised input
#    alpha         : Laplace smoothing [1e-3, 10.0]
#    binarize      : threshold for binarising features [0.0, 1.0]
#
#  All variants:
#    class_prior_pos : [0.3, 0.7] → priors=[1-p, p]
#                      only applied to GaussianNB (supports priors)
#                      CNB/BNB handle imbalance via counts/alpha
#
#  ⚠️  Optuna CONSTANT DISTRIBUTION RULE:
#      All parameters always sampled from the same fixed range.
#      Only relevant ones passed to the chosen variant inside
#      build_nb_classifier() — unused ones are silently ignored.
#
SEARCH_SPACE = {
    "nb_type"        : ["gaussian", "complement", "bernoulli"],
    "var_smoothing"  : "float [1e-12, 1e-1] log-uniform  (gaussian only)",
    "alpha"          : "float [1e-3, 10.0]  log-uniform  (complement / bernoulli)",
    "norm"           : "[True, False]  (complement only)",
    "binarize"       : "float [0.0, 1.0]  (bernoulli only)",
    "class_prior_pos": "float [0.3, 0.7]  → priors=[1-p, p]  (gaussian only)",
}

print(f"✅ Config loaded")
print(f"   Datasets    : {len(DATASETS)}")
print(f"   N_TRIALS    : {N_TRIALS}")
print(f"   CV folds    : {N_CV_FOLDS}")
print(f"   Train/Test  : {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
print(f"   Scaler      : MinMaxScaler [0,1]")
print(f"\n   Search space:")
for k, v in SEARCH_SPACE.items():
    print(f"     {k:<18} : {v}")


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


def build_nb_classifier(nb_type, var_smoothing, alpha,
                         norm, binarize, class_prior_pos):
    """
    Instantiate the correct NB classifier from sampled params.
    Only passes parameters relevant to the chosen variant.

    class_prior_pos → priors = [1 - p, p]
      Dataset ratio: pos:neg = 876:1314 ≈ 0.40:0.60
      Optuna explores [0.3, 0.7] to find the best prior balance.
      priors only accepted by GaussianNB, not CNB/BNB.
    """
    priors = [1.0 - class_prior_pos, class_prior_pos]

    if nb_type == "gaussian":
        return GaussianNB(
            var_smoothing = var_smoothing,
            priors        = priors,
        )
    elif nb_type == "complement":
        return ComplementNB(
            alpha = alpha,
            norm  = norm,
        )
    else:  # bernoulli
        return BernoulliNB(
            alpha    = alpha,
            binarize = binarize,
        )


# ============================================================
#   CELL 5 — Optuna Objective Function
# ============================================================

def make_objective(X_train, y_train, n_folds, seed):
    """
    Returns an Optuna objective function for Naive Bayes.

    Objective: maximise mean AUC across stratified k-fold CV.

    NB-specific design — Optuna constant distribution rule:
      All parameters sampled from fixed ranges every trial.
      Only relevant params passed to chosen variant via
      build_nb_classifier(). Unused params are silently ignored.

      This avoids the CategoricalDistribution dynamic value
      space error encountered in the LR script.
    """
    def objective(trial):

        nb_type = trial.suggest_categorical(
            "nb_type", ["gaussian", "complement", "bernoulli"]
        )

        # ── All always sampled from fixed ranges ──────────────
        var_smoothing   = trial.suggest_float(
            "var_smoothing", 1e-12, 1e-1, log=True
        )
        alpha           = trial.suggest_float(
            "alpha", 1e-3, 10.0, log=True
        )
        norm            = trial.suggest_categorical(
            "norm", [True, False]
        )
        binarize        = trial.suggest_float(
            "binarize", 0.0, 1.0
        )
        class_prior_pos = trial.suggest_float(
            "class_prior_pos", 0.3, 0.7
        )

        clf = build_nb_classifier(
            nb_type, var_smoothing, alpha,
            norm, binarize, class_prior_pos
        )

        cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        auc_scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return auc_scores.mean()

    return objective


def extract_best_params(trial_params):
    """
    From the full Optuna trial params dict, extract only
    the params that are relevant to the chosen NB variant.
    Returns a clean, variant-specific dict.
    """
    p       = trial_params.copy()
    nb_type = p.get("nb_type", "gaussian")

    if nb_type == "gaussian":
        return {
            "nb_type"        : nb_type,
            "var_smoothing"  : p["var_smoothing"],
            "class_prior_pos": p["class_prior_pos"],
        }
    elif nb_type == "complement":
        return {
            "nb_type"        : nb_type,
            "alpha"          : p["alpha"],
            "norm"           : p["norm"],
            "class_prior_pos": p["class_prior_pos"],
        }
    else:  # bernoulli
        return {
            "nb_type"        : nb_type,
            "alpha"          : p["alpha"],
            "binarize"       : p["binarize"],
            "class_prior_pos": p["class_prior_pos"],
        }


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
print("  Naive Bayes + Optuna — Training on 11 Datasets")
print(f"  Trials per dataset : {N_TRIALS}")
print(f"  CV folds           : {N_CV_FOLDS} (StratifiedKFold)")
print(f"  Sampler            : TPE (Tree-structured Parzen Estimator)")
print(f"  Variants           : GaussianNB / ComplementNB / BernoulliNB")
print(f"  Scaler             : MinMaxScaler → [0, 1]")
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

    # ── Scale to [0, 1] ──────────────────────────────────────
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Run Optuna ───────────────────────────────────────────
    print(f"  Running Optuna ({N_TRIALS} trials, {N_CV_FOLDS}-fold CV)...")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = TPESampler(seed=OPTUNA_SEED),
        study_name = f"NB_{ds_name}",
    )
    study.optimize(
        make_objective(X_train, y_train, N_CV_FOLDS, RANDOM_STATE),
        n_trials          = N_TRIALS,
        show_progress_bar = True,
    )

    all_studies[ds_name] = study

    # ── Extract best parameters ───────────────────────────────
    best_params = extract_best_params(study.best_trial.params)
    cv_auc      = study.best_trial.value
    nb_type     = best_params["nb_type"]

    print(f"\n  ── Best Parameters (trial #{study.best_trial.number}) ──")
    for k, v in best_params.items():
        print(f"     {k:<18} : {v}")
    print(f"     {'CV AUC':<18} : {cv_auc:.4f}")

    # ── Build and train final model ───────────────────────────
    clf = build_nb_classifier(
        nb_type         = best_params["nb_type"],
        var_smoothing   = best_params.get("var_smoothing", 1e-9),
        alpha           = best_params.get("alpha", 1.0),
        norm            = best_params.get("norm", False),
        binarize        = best_params.get("binarize", 0.0),
        class_prior_pos = best_params.get("class_prior_pos", 0.40),
    )
    clf.fit(X_train, y_train)

    # ── Predict ──────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # ── Metrics ──────────────────────────────────────────────
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["Dataset"]    = ds_name
    metrics["Features"]   = X.shape[1]
    metrics["NB_Type"]    = nb_type
    metrics["CV_AUC"]     = round(cv_auc, 4)
    metrics["Best_Trial"] = study.best_trial.number
    metrics["Train_N"]    = len(X_train)
    metrics["Test_N"]     = len(X_test)

    for k, v in best_params.items():
        metrics[f"param_{k}"] = v
    all_results.append(metrics)

    print(f"\n  ── Test Set Results ─────────────────────────────")
    print(f"  NB variant  : {nb_type}")
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
        RESULTS_DIR, f"{ds_name}_NB_Optuna_probabilities.csv"
    )
    df_probs.to_csv(prob_path, index=False)

    # ── Save best params as JSON ─────────────────────────────
    params_to_save = dict(best_params)
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
        PARAMS_DIR, f"{ds_name}_NB_best_params.json"
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
        "nb_type"    : nb_type,
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
print(f"  🏆 Best model: {best_name}  "
      f"(AUC = {best_auc:.4f}  "
      f"variant = {all_probs[best_name]['nb_type']})")
print(f"{'='*65}")


# ============================================================
#   CELL 7 — Results Summary Table
# ============================================================

metric_cols = ["Dataset", "NB_Type", "CV_AUC", "Accuracy",
               "Sensitivity", "Specificity", "F1_Score",
               "MCC", "AUC", "Best_Trial", "Features"]

df_results = pd.DataFrame(all_results)[metric_cols].sort_values(
    "AUC", ascending=False
).reset_index(drop=True)

df_results.index += 1

print("\n── NB + Optuna Performance Summary (sorted by Test AUC) ──")
print(df_results.to_string())

summary_path = os.path.join(
    RESULTS_DIR, "NB_Optuna_all_results_summary.csv"
)
df_results.to_csv(summary_path, index=True, index_label="Rank")
print(f"\n✅ Summary saved → {summary_path}")

df_full   = pd.DataFrame(all_results)
full_path = os.path.join(
    RESULTS_DIR, "NB_Optuna_full_results_with_params.csv"
)
df_full.to_csv(full_path, index=False)
print(f"✅ Full results (with params) saved → {full_path}")


# ============================================================
#   CELL 8 — Save Best Model
# ============================================================

best_model_path  = os.path.join(
    MODELS_DIR, f"NB_Optuna_best_model_{best_name}.joblib"
)
best_scaler_path = os.path.join(
    MODELS_DIR, f"NB_Optuna_best_scaler_{best_name}.joblib"
)

joblib.dump(best_model,  best_model_path)
joblib.dump(best_scaler, best_scaler_path)

best_params_summary = dict(all_probs[best_name]["best_params"])
best_params_summary.update({
    "dataset" : best_name,
    "test_auc": best_auc,
    "cv_auc"  : round(all_probs[best_name]["cv_auc"], 4),
})
best_overall_path = os.path.join(
    MODELS_DIR, "NB_Optuna_best_overall_params.json"
)
with open(best_overall_path, "w") as f:
    json.dump(best_params_summary, f, indent=4)

print(f"✅ Best model saved")
print(f"   Dataset          : {best_name}")
print(f"   NB variant       : {all_probs[best_name]['nb_type']}")
print(f"   Test AUC         : {best_auc:.4f}")
print(f"   CV  AUC          : {all_probs[best_name]['cv_auc']:.4f}")
print(f"   Model            : {best_model_path}")
print(f"   Scaler           : {best_scaler_path}")
print(f"   Best params JSON : {best_overall_path}")
print(f"\n   Best hyperparameters:")
for k, v in all_probs[best_name]["best_params"].items():
    print(f"     {k:<18} : {v}")


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
    f"Naive Bayes + Optuna ({N_TRIALS} trials) — "
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
                         "NB_Optuna_metrics_heatmap.png"),
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

# Annotate NB variant at base of each group
for xi, row in zip(x, df_results.itertuples()):
    ax.text(xi, 0.02, row.NB_Type[:3].upper(),
            ha="center", va="bottom", fontsize=7,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1",
                      facecolor=NB_COLORS.get(row.NB_Type, "#2c3e50"),
                      alpha=0.85))

ax.set_xlabel("Dataset", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Naive Bayes + Optuna — All Metrics per Dataset\n"
             "(GAU=Gaussian  COM=Complement  BER=Bernoulli)",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(df_results["Dataset"],
                   rotation=20, ha="right", fontsize=10)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "NB_Optuna_grouped_bar_chart.png"),
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
    lw          = 2.5 if ds_name == best_name else 1.2
    nbt         = data["nb_type"]
    ax.plot(
        fpr, tpr, color=color, linewidth=lw,
        linestyle=NB_LINE_STYLES.get(nbt, "-"),
        label=f"{ds_name} [{nbt[:3].upper()}] "
              f"(AUC={auc_val:.4f})"
              + (" ★" if ds_name == best_name else "")
    )

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Naive Bayes + Optuna — ROC Curves for All Datasets\n"
             "(GAU=Gaussian  COM=Complement  BER=Bernoulli)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "NB_Optuna_ROC_curves.png"),
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
bp  = best_data["best_params"]
nbt = best_data["nb_type"]
param_str = (
    f"var_smooth={bp.get('var_smoothing','-'):.2e}"
    if nbt == "gaussian"
    else f"alpha={bp.get('alpha','-'):.4f}"
)
ax.set_title(
    f"Confusion Matrix — Best: {best_name}\n"
    f"variant={nbt}  {param_str}\n"
    f"(Test AUC={best_auc:.4f}  "
    f"CV AUC={best_data['cv_auc']:.4f})",
    fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         f"NB_Optuna_confusion_matrix_{best_name}.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Confusion matrix saved ({best_name})")


# ============================================================
#   CELL 13 — Visualization 5: AUC Ranking
#             Bars coloured by NB variant
# ============================================================

from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(13, 5))
sorted_df = df_results.sort_values("AUC", ascending=True)
x         = np.arange(len(sorted_df))
width     = 0.35

test_colors = [
    "#2ecc71" if n == best_name
    else NB_COLORS.get(all_probs[n]["nb_type"], "#3498db")
    for n in sorted_df["Dataset"]
]

ax.barh(x + width / 2, sorted_df["AUC"].values, width,
        label="Test AUC", color=test_colors,
        edgecolor="white", alpha=0.85)
ax.barh(x - width / 2, sorted_df["CV_AUC"].values, width,
        label="CV AUC",
        color=["#27ae60" if n == best_name else "#7f8c8d"
               for n in sorted_df["Dataset"]],
        edgecolor="white", alpha=0.65)

for i, ds_name in enumerate(sorted_df["Dataset"]):
    auc = sorted_df.loc[sorted_df["Dataset"] == ds_name,
                        "AUC"].values[0]
    nbt = all_probs[ds_name]["nb_type"]
    ax.text(auc + 0.005, i + width / 2,
            f"{auc:.4f}  [{nbt[:3].upper()}]",
            va="center", fontsize=9)

ax.set_yticks(x)
ax.set_yticklabels(sorted_df["Dataset"], fontsize=10)
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("AUC Score", fontsize=12)
ax.set_title(
    f"Naive Bayes + Optuna ({N_TRIALS} trials) — "
    f"Test AUC vs CV AUC Ranking",
    fontsize=13, fontweight="bold"
)
ax.set_xlim(0, 1.18)
ax.grid(axis="x", alpha=0.3)

variant_patches = [
    Patch(color="#3498db", label="Gaussian"),
    Patch(color="#e67e22", label="Complement"),
    Patch(color="#9b59b6", label="Bernoulli"),
    Patch(color="#2ecc71", label="Best dataset"),
]
ax.legend(handles=variant_patches, fontsize=10, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "NB_Optuna_AUC_ranking.png"),
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

    nbt        = all_probs.get(ds_name, {}).get("nb_type", "?")
    title_col  = NB_COLORS.get(nbt, "#2c3e50")

    ax.scatter(trial_nums, trial_vals,
               color="#95a5a6", s=12, alpha=0.5,
               label="Trial AUC", zorder=2)
    ax.plot(trial_nums, running_best,
            "-", color="#e74c3c", linewidth=2,
            label=f"Best: {max(trial_vals):.4f}", zorder=3)
    ax.axhline(max(trial_vals), color="#e74c3c",
               linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(f"Best: {nbt}", fontsize=9)
    ax.set_title(ds_name, fontsize=11,
                 fontweight="bold", color=title_col)
    ax.set_ylabel("CV AUC", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, N_TRIALS)

for idx in range(len(all_studies), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(
    f"Optuna Optimization History — Naive Bayes "
    f"({N_TRIALS} trials per dataset)\n"
    f"Title colour: blue=Gaussian  orange=Complement  purple=Bernoulli",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "NB_Optuna_optimization_history.png"),
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
                   color="#16a085", edgecolor="white", alpha=0.85)
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
                     f"NB_Optuna_param_importance_{best_name}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print(f"✅ Parameter importance plot saved ({best_name})")

except Exception as e:
    print(f"ℹ️  Parameter importance skipped: {e}")


# ============================================================
#   CELL 16 — Visualization 8: Variant Distribution
#             + Class-Conditional Feature Analysis
#             (unique to Naive Bayes — no other classifier has this)
# ============================================================

variant_counts = {}
for data in all_probs.values():
    nbt = data["nb_type"]
    variant_counts[nbt] = variant_counts.get(nbt, 0) + 1

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ── Left: NB variant distribution pie ────────────────────────
ax_pie = axes[0]
pie_colors = [NB_COLORS.get(k, "#95a5a6") for k in variant_counts]
ax_pie.pie(
    list(variant_counts.values()),
    labels     = list(variant_counts.keys()),
    colors     = pie_colors,
    autopct    = "%1.0f%%",
    startangle = 140,
    wedgeprops = dict(edgecolor="white", linewidth=2),
)
ax_pie.set_title("NB Variant selected\nacross 11 datasets",
                 fontsize=12, fontweight="bold")

# ── Right: Class-conditional feature analysis (best model) ───
ax_dist = axes[1]
best_clf = all_probs[best_name]["clf"]
nbt      = all_probs[best_name]["nb_type"]

if nbt == "gaussian":
    means_neg = best_clf.theta_[0]
    means_pos = best_clf.theta_[1]
    n_feat    = min(20, len(means_neg))
    top_idx   = np.argsort(
        np.abs(means_pos - means_neg)
    )[::-1][:n_feat]
    xi = np.arange(n_feat)
    ax_dist.bar(xi - 0.2, means_neg[top_idx], 0.4,
                label="Non-AIP (0)", color="#e74c3c", alpha=0.75)
    ax_dist.bar(xi + 0.2, means_pos[top_idx], 0.4,
                label="AIP (1)", color="#2ecc71", alpha=0.75)
    ax_dist.set_xticks(xi)
    ax_dist.set_xticklabels([f"F{i}" for i in top_idx],
                             rotation=45, ha="right", fontsize=8)
    ax_dist.set_ylabel("Class-conditional Mean (MinMax scaled)",
                       fontsize=10)
    ax_dist.set_title(
        f"GaussianNB — Top {n_feat} Most Discriminative Features\n"
        f"({best_name})",
        fontsize=11, fontweight="bold"
    )
    ax_dist.legend(fontsize=10)

else:
    log_probs = best_clf.feature_log_prob_
    log_ratio = log_probs[1] - log_probs[0] \
                if log_probs.shape[0] >= 2 else log_probs[0]

    n_feat   = min(20, len(log_ratio))
    top_idx  = np.argsort(np.abs(log_ratio))[::-1][:n_feat]
    top_lr   = log_ratio[top_idx]
    bar_cols = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_lr]

    ax_dist.bar(range(n_feat), top_lr, color=bar_cols,
                edgecolor="white", alpha=0.85)
    ax_dist.axhline(0, color="black", linewidth=0.8)
    ax_dist.set_xticks(range(n_feat))
    ax_dist.set_xticklabels([f"F{i}" for i in top_idx],
                             rotation=45, ha="right", fontsize=8)
    ax_dist.set_ylabel("Log P(feat|AIP) − Log P(feat|NonAIP)",
                       fontsize=10)
    ax_dist.set_title(
        f"{nbt.capitalize()}NB — Top {n_feat} Log-Probability Ratios\n"
        f"({best_name})",
        fontsize=11, fontweight="bold"
    )
    ax_dist.legend(handles=[
        Patch(color="#2ecc71", label="More AIP-like"),
        Patch(color="#e74c3c", label="More Non-AIP-like"),
    ], fontsize=10)

ax_dist.grid(axis="y", alpha=0.3)
plt.suptitle("Naive Bayes + Optuna — Variant Distribution & "
             "Class-Conditional Feature Analysis",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,
                         "NB_Optuna_variant_class_analysis.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Variant distribution & class-conditional plot saved")


# ============================================================
#   CELL 17 — Best Parameters Table (All Datasets)
# ============================================================

print("\n── Best Hyperparameters per Dataset ──────────────────────")
param_rows = []
for ds_name, data in all_probs.items():
    row = {"Dataset": ds_name}
    row.update(data["best_params"])
    row["CV_AUC"]   = round(data["cv_auc"], 4)
    row["Test_AUC"] = round(
        roc_auc_score(data["y_test"], data["y_prob"]), 4
    )
    param_rows.append(row)

df_params = pd.DataFrame(param_rows)
print(df_params.to_string(index=False))

params_table_path = os.path.join(
    PARAMS_DIR, "NB_Optuna_all_best_params.csv"
)
df_params.to_csv(params_table_path, index=False)
print(f"\n✅ Best params table saved → {params_table_path}")


# ============================================================
#   CELL 18 — Final Summary
# ============================================================

best_row    = df_results[df_results["Dataset"] == best_name].iloc[0]
best_params = all_probs[best_name]["best_params"]

print("=" * 65)
print("  NAIVE BAYES + OPTUNA — FINAL SUMMARY")
print("=" * 65)
print(f"\n  Results saved to       : {RESULTS_DIR}")
print(f"  Figures saved to       : {FIGURES_DIR}")
print(f"  Model saved to         : {MODELS_DIR}")
print(f"  Best params saved to   : {PARAMS_DIR}")
print(f"\n{'─'*65}")
print(f"  {'Dataset':<12} {'Variant':<12} {'CV_AUC':>8} {'Acc':>8} "
      f"{'Sn':>8} {'Sp':>8} {'F1':>8} {'MCC':>8} {'AUC':>8}")
print(f"{'─'*65}")
for _, row in df_results.iterrows():
    marker = " ★" if row["Dataset"] == best_name else ""
    print(f"  {row['Dataset']:<12} {row['NB_Type']:<12} "
          f"{row['CV_AUC']:>8.4f} "
          f"{row['Accuracy']:>8.4f} {row['Sensitivity']:>8.4f} "
          f"{row['Specificity']:>8.4f} {row['F1_Score']:>8.4f} "
          f"{row['MCC']:>8.4f} {row['AUC']:>8.4f}{marker}")
print(f"{'─'*65}")
print(f"\n  🏆 Best Dataset   : {best_name}")
print(f"     NB variant     : {all_probs[best_name]['nb_type']}")
print(f"     Test AUC       : {best_auc:.4f}")
print(f"     CV  AUC        : {all_probs[best_name]['cv_auc']:.4f}")
print(f"     Accuracy       : {best_row['Accuracy']:.4f}")
print(f"     Sensitivity    : {best_row['Sensitivity']:.4f}")
print(f"     Specificity    : {best_row['Specificity']:.4f}")
print(f"     F1 Score       : {best_row['F1_Score']:.4f}")
print(f"     MCC            : {best_row['MCC']:.4f}")
print(f"\n  Best hyperparameters ({best_name}):")
for k, v in best_params.items():
    print(f"     {k:<18} : {v}")
print(f"\n  NB variant distribution:")
for nbt, cnt in sorted(variant_counts.items(), key=lambda x: -x[1]):
    print(f"     {nbt:<12} : {cnt} dataset(s)")
print(f"\n  Optuna settings:")
print(f"     Sampler        : TPE (Tree-structured Parzen Estimator)")
print(f"     Trials         : {N_TRIALS}")
print(f"     CV folds       : {N_CV_FOLDS} (StratifiedKFold)")
print(f"     Objective      : Maximise mean CV AUC")
print(f"     Scaler         : MinMaxScaler [0,1]")
print(f"\n  Saved files:")
print(f"     Per-dataset JSON params  : {PARAMS_DIR}/")
print(f"     All-params CSV           : {params_table_path}")
print(f"     Full results CSV         : {full_path}")
print(f"     Per-dataset probs CSV    : {RESULTS_DIR}/")
print(f"     Best model               : {best_model_path}")
print("=" * 65)