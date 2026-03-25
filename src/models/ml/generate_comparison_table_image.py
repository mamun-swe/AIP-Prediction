"""
generate_comparison_table_image.py
===================================
Generates a clean comparison table image exactly like the
sample — Default vs Optuna, one row per classifier,
6 metrics per group.

Usage:
    python generate_comparison_table_image.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")


# ============================================================
#   CONFIGURATION
# ============================================================

# ── Update to your results folder ───────────────────────────
# Google Colab:
# BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results"

# Local (update this path):
BASE_DIR    = "../../../results"

OUTPUT_DIR  = os.path.join(BASE_DIR, "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = ["Accuracy", "Sensitivity", "Specificity",
           "F1 Score", "MCC", "ACC"]

METRIC_KEYS = ["Accuracy", "Sensitivity", "Specificity",
               "F1_Score", "MCC", "Accuracy"]
# Note: ACC and Accuracy are the same metric —
# last column uses Accuracy values as ACC

ALL_DATASETS = [
    "AAC", "DPC", "CTDC", "CTDT", "GAAC", "PAAC",
    "ProtBERT", "ProtT5", "ESM2", "BioBERT", "ESMC"
]

CLASSIFIERS = {
    "DT"  : {
        "baseline_folder": "models/decision_tree",
        "optuna_folder"  : "models/dt_optuna",
        "baseline_file"  : "DT_all_results_summary.csv",
        "optuna_file"    : "DT_Optuna_all_results_summary.csv",
    },
    "RF"  : {
        "baseline_folder": "models/random_forest",
        "optuna_folder"  : "models/rf_optuna",
        "baseline_file"  : "RF_all_results_summary.csv",
        "optuna_file"    : "RF_Optuna_all_results_summary.csv",
    },
    "XGB" : {
        "baseline_folder": "models/xgboost",
        "optuna_folder"  : "models/xgb_optuna",
        "baseline_file"  : "XGB_all_results_summary.csv",
        "optuna_file"    : "XGB_Optuna_all_results_summary.csv",
    },
    "KNN" : {
        "baseline_folder": "models/knn",
        "optuna_folder"  : "models/knn_optuna",
        "baseline_file"  : "KNN_all_results_summary.csv",
        "optuna_file"    : "KNN_Optuna_all_results_summary.csv",
    },
    "SVM" : {
        "baseline_folder": "models/svm",
        "optuna_folder"  : "models/svm_optuna",
        "baseline_file"  : "SVM_all_results_summary.csv",
        "optuna_file"    : "SVM_Optuna_all_results_summary.csv",
    },
    "ET"  : {
        "baseline_folder": "models/extra_trees",
        "optuna_folder"  : "models/et_optuna",
        "baseline_file"  : "ET_all_results_summary.csv",
        "optuna_file"    : "ET_Optuna_all_results_summary.csv",
    },
    "LGBM": {
        "baseline_folder": "models/lightgbm",
        "optuna_folder"  : "models/lgbm_optuna",
        "baseline_file"  : "LGBM_all_results_summary.csv",
        "optuna_file"    : "LGBM_Optuna_all_results_summary.csv",
    },
    "LR"  : {
        "baseline_folder": "models/logistic_regression",
        "optuna_folder"  : "models/lr_optuna",
        "baseline_file"  : "LR_all_results_summary.csv",
        "optuna_file"    : "LR_Optuna_all_results_summary.csv",
    },
    "CB"  : {
        "baseline_folder": "models/catboost",
        "optuna_folder"  : "models/cb_optuna",
        "baseline_file"  : "CB_all_results_summary.csv",
        "optuna_file"    : "CB_Optuna_all_results_summary.csv",
    },
    "NB"  : {
        "baseline_folder": "models/naive_bayes",
        "optuna_folder"  : "models/nb_optuna",
        "baseline_file"  : "NB_all_results_summary.csv",
        "optuna_file"    : "NB_Optuna_all_results_summary.csv",
    },
}


# ============================================================
#   HELPER — Load best result per classifier
# ============================================================

def load_best(path, metric_keys):
    """Load CSV and return best row by AUC."""
    if not os.path.exists(path):
        return {k: None for k in metric_keys}

    df = pd.read_csv(path)
    for col in ["Rank", "Unnamed: 0", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "AUC" not in df.columns or "Dataset" not in df.columns:
        return {k: None for k in metric_keys}

    df = df.sort_values("AUC", ascending=False)
    best = df.iloc[0]

    result = {}
    for k in metric_keys:
        col = k
        if col in df.columns:
            try:
                result[k] = round(float(best[col]), 4)
            except Exception:
                result[k] = None
        else:
            result[k] = None

    return result


# ============================================================
#   BUILD TABLE DATA
# ============================================================

print("Loading results...")

actual_metric_keys = ["Accuracy", "Sensitivity", "Specificity",
                      "F1_Score", "MCC", "AUC"]

table_data = []

for clf_short, cfg in CLASSIFIERS.items():
    base_path   = os.path.join(
        BASE_DIR, cfg["baseline_folder"], cfg["baseline_file"]
    )
    optuna_path = os.path.join(
        BASE_DIR, cfg["optuna_folder"], cfg["optuna_file"]
    )

    b_found = os.path.exists(base_path)
    o_found = os.path.exists(optuna_path)

    if not b_found:
        print(f"  ⚠️  Baseline not found: {base_path}")
    if not o_found:
        print(f"  ⚠️  Optuna   not found: {optuna_path}")

    base_vals  = load_best(base_path,   actual_metric_keys)
    optuna_vals= load_best(optuna_path, actual_metric_keys)

    row = {"CLF": clf_short}
    for k in actual_metric_keys:
        row[f"B_{k}"] = base_vals[k]
        row[f"O_{k}"] = optuna_vals[k]
    table_data.append(row)

df = pd.DataFrame(table_data)
print(f"✅ Loaded {len(df)} classifiers")


# ============================================================
#   DRAW TABLE IMAGE
# ============================================================

# ── Colours matching the sample image ────────────────────────
HEADER_BG    = "#F0A500"   # orange/yellow header
SUBHEADER_BG = "#F0A500"   # same for sub-header row
CLF_COL_BG   = "#F0A500"   # classifier column background
ROW_ODD      = "#FFFFFF"   # white rows
ROW_EVEN     = "#F7F7F7"   # very light grey alternate rows
DIVIDER_COL  = "#3A7FCA"   # blue divider between Default/Optuna
TEXT_DARK    = "#1A1A1A"
TEXT_WHITE   = "#FFFFFF"
BORDER_COL   = "#CCCCCC"

# ── Table dimensions ─────────────────────────────────────────
N_ROWS    = len(df)          # 10 classifiers
N_METRICS = 6                # per group
N_COLS    = 1 + N_METRICS * 2  # clf + 6 default + 6 optuna = 13

COL_W_CLF  = 0.80   # inches — classifier name column
COL_W_MET  = 0.72   # inches — metric columns
ROW_H      = 0.35   # inches — data rows
HEADER_H   = 0.40   # inches — header rows

TABLE_W = COL_W_CLF + COL_W_MET * N_METRICS * 2
TABLE_H = HEADER_H * 2 + ROW_H * N_ROWS

fig_w = TABLE_W + 0.6
fig_h = TABLE_H + 0.6

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, TABLE_W)
ax.set_ylim(0, TABLE_H)
ax.axis("off")

# ── Compute column x positions ────────────────────────────────
x_clf   = 0.0
x_cols  = [x_clf + COL_W_CLF + i * COL_W_MET
           for i in range(N_METRICS * 2)]
# x_cols[0..5]  = Default metrics
# x_cols[6..11] = Optuna metrics

def cell_rect(x, y, w, h, fc, ec=BORDER_COL, lw=0.6, zorder=1):
    ax.add_patch(plt.Rectangle(
        (x, y), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder
    ))

def cell_text(x, y, w, h, txt, fc="black",
              fontsize=8, bold=False, ha="center"):
    ax.text(
        x + w / 2, y + h / 2, txt,
        ha=ha, va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=fc, zorder=2,
        clip_on=True
    )

# ── Row 1: "Default" and "Optuna" spanning headers ───────────
y_h1 = TABLE_H - HEADER_H

# empty top-left cell
cell_rect(x_clf, y_h1, COL_W_CLF, HEADER_H, HEADER_BG)

# Default spanning header
default_w = COL_W_MET * N_METRICS
cell_rect(x_clf + COL_W_CLF, y_h1,
          default_w, HEADER_H, HEADER_BG,
          ec="#DDDDDD", lw=1.0)
cell_text(x_clf + COL_W_CLF, y_h1,
          default_w, HEADER_H,
          "Default", fc=TEXT_WHITE,
          fontsize=11, bold=True)

# Optuna spanning header
optuna_w = COL_W_MET * N_METRICS
cell_rect(x_clf + COL_W_CLF + default_w, y_h1,
          optuna_w, HEADER_H, DIVIDER_COL,
          ec="#2060A0", lw=1.0)
cell_text(x_clf + COL_W_CLF + default_w, y_h1,
          optuna_w, HEADER_H,
          "Optuna", fc=TEXT_WHITE,
          fontsize=11, bold=True)

# ── Row 2: metric sub-headers ─────────────────────────────────
y_h2 = TABLE_H - HEADER_H * 2

# empty clf cell
cell_rect(x_clf, y_h2, COL_W_CLF, HEADER_H, HEADER_BG)

metric_display = ["Accuracy", "Sensitivity", "Specificity",
                  "F1\nScore", "MCC", "AUC"]

for i in range(N_METRICS):
    # Default sub-header
    cell_rect(x_cols[i], y_h2,
              COL_W_MET, HEADER_H, HEADER_BG,
              ec="#DDDDDD", lw=0.6)
    cell_text(x_cols[i], y_h2,
              COL_W_MET, HEADER_H,
              metric_display[i], fc=TEXT_WHITE,
              fontsize=7, bold=True)

    # Optuna sub-header
    cell_rect(x_cols[i + N_METRICS], y_h2,
              COL_W_MET, HEADER_H, DIVIDER_COL,
              ec="#2060A0", lw=0.6)
    cell_text(x_cols[i + N_METRICS], y_h2,
              COL_W_MET, HEADER_H,
              metric_display[i], fc=TEXT_WHITE,
              fontsize=7, bold=True)

# ── Data rows ─────────────────────────────────────────────────
metric_order = ["Accuracy", "Sensitivity", "Specificity",
                "F1_Score", "MCC", "AUC"]

for ri, (_, row) in enumerate(df.iterrows()):
    y_row  = TABLE_H - HEADER_H * 2 - ROW_H * (ri + 1)
    row_bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN

    # Classifier name cell
    cell_rect(x_clf, y_row, COL_W_CLF, ROW_H,
              CLF_COL_BG if ri % 2 == 0 else "#E8950A")
    cell_text(x_clf, y_row, COL_W_CLF, ROW_H,
              row["CLF"], fc=TEXT_WHITE,
              fontsize=9, bold=True)

    # Default metric cells
    for i, mk in enumerate(metric_order):
        val = row[f"B_{mk}"]
        txt = f"{val:.4f}" if val is not None and not pd.isna(val) else "—"
        cell_rect(x_cols[i], y_row,
                  COL_W_MET, ROW_H, row_bg)
        cell_text(x_cols[i], y_row,
                  COL_W_MET, ROW_H, txt,
                  fc=TEXT_DARK, fontsize=8)

    # Optuna metric cells
    for i, mk in enumerate(metric_order):
        b_val = row[f"B_{mk}"]
        o_val = row[f"O_{mk}"]
        txt   = (f"{o_val:.4f}"
                 if o_val is not None and not pd.isna(o_val)
                 else "—")

        # Highlight green if improved, red if degraded
        cell_bg = row_bg
        txt_col = TEXT_DARK
        if (b_val is not None and o_val is not None
                and not pd.isna(b_val) and not pd.isna(o_val)):
            delta = o_val - b_val
            if delta > 0.001:
                cell_bg = "#D5F5E3"   # light green
                txt_col = "#1A6B34"
            elif delta < -0.001:
                cell_bg = "#FADBD8"   # light red
                txt_col = "#922B21"

        cell_rect(x_cols[i + N_METRICS], y_row,
                  COL_W_MET, ROW_H, cell_bg,
                  ec=BORDER_COL, lw=0.6)
        cell_text(x_cols[i + N_METRICS], y_row,
                  COL_W_MET, ROW_H, txt,
                  fc=txt_col, fontsize=8)

# ── Outer border ─────────────────────────────────────────────
ax.add_patch(plt.Rectangle(
    (0, 0), TABLE_W, TABLE_H,
    fill=False, edgecolor="#333333", linewidth=1.5, zorder=3
))

# ── Vertical divider between Default and Optuna ───────────────
div_x = x_clf + COL_W_CLF + COL_W_MET * N_METRICS
ax.plot([div_x, div_x], [0, TABLE_H],
        color=DIVIDER_COL, linewidth=2.0, zorder=3)

# ── Legend ───────────────────────────────────────────────────
legend_y = -0.18
ax.add_patch(plt.Rectangle(
    (0.0, legend_y), 0.18, 0.12,
    facecolor="#D5F5E3", edgecolor="#AAAAAA",
    linewidth=0.6, transform=ax.transData
))
ax.text(0.20, legend_y + 0.06,
        "Improved after Optuna tuning",
        va="center", fontsize=7.5, color="#1A6B34",
        transform=ax.transData)

ax.add_patch(plt.Rectangle(
    (2.0, legend_y), 0.18, 0.12,
    facecolor="#FADBD8", edgecolor="#AAAAAA",
    linewidth=0.6, transform=ax.transData
))
ax.text(2.20, legend_y + 0.06,
        "Degraded after Optuna tuning",
        va="center", fontsize=7.5, color="#922B21",
        transform=ax.transData)

# ── Title ─────────────────────────────────────────────────────
ax.text(
    TABLE_W / 2, TABLE_H + 0.10,
    "Performance Comparison: Default vs Optuna-Tuned Classifiers",
    ha="center", va="bottom",
    fontsize=12, fontweight="bold", color="#1A1A1A",
    transform=ax.transData
)
ax.text(
    TABLE_W / 2, TABLE_H + 0.02,
    "(Best dataset per classifier — AIP Prediction)",
    ha="center", va="bottom",
    fontsize=8, color="#555555",
    transform=ax.transData
)

plt.subplots_adjust(left=0.02, right=0.98,
                    top=0.90, bottom=0.10)

# ── Save ──────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "comparison_table.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white")
plt.show()
print(f"\n✅ Table image saved → {out_path}")
print(f"   Green cells = Optuna improved over Default")
print(f"   Red cells   = Optuna degraded vs Default")
print(f"   White/grey  = no meaningful change")