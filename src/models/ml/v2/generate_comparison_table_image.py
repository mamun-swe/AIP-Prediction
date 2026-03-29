"""
generate_comparison_table_image.py
===================================
Generates a clean comparison table image — Default vs Optuna,
one row per classifier, 6 metrics per group.

Usage:
    python generate_comparison_table_image.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ============================================================
#   CONFIGURATION
# ============================================================

# Google Colab:
# BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/AIP Prediction/results"

# Local:
BASE_DIR   = "../../../../results/comparison"

OUTPUT_DIR = os.path.join(BASE_DIR, "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

actual_metric_keys = ["Accuracy", "Sensitivity", "Specificity",
                      "F1_Score", "MCC", "AUC"]


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

    df   = df.sort_values("AUC", ascending=False)
    best = df.iloc[0]

    result = {}
    for k in metric_keys:
        if k in df.columns:
            try:
                result[k] = round(float(best[k]), 4)
            except Exception:
                result[k] = None
        else:
            result[k] = None
    return result


# ============================================================
#   BUILD TABLE DATA
# ============================================================

print("Loading results...")

table_data = []

for clf_short, cfg in CLASSIFIERS.items():
    base_path   = os.path.join(
        BASE_DIR, cfg["baseline_folder"], cfg["baseline_file"]
    )
    optuna_path = os.path.join(
        BASE_DIR, cfg["optuna_folder"], cfg["optuna_file"]
    )

    if not os.path.exists(base_path):
        print(f"  ⚠️  Baseline not found: {base_path}")
    if not os.path.exists(optuna_path):
        print(f"  ⚠️  Optuna   not found: {optuna_path}")

    base_vals   = load_best(base_path,   actual_metric_keys)
    optuna_vals = load_best(optuna_path, actual_metric_keys)

    row = {"CLF": clf_short}
    for k in actual_metric_keys:
        row[f"B_{k}"] = base_vals[k]
        row[f"O_{k}"] = optuna_vals[k]
    table_data.append(row)

df = pd.DataFrame(table_data)
print(f"✅ Loaded {len(df)} classifiers")


# ============================================================
#   COLOUR SCHEME
# ============================================================

HEADER_BG   = "#F0A500"
DIVIDER_COL = "#3A7FCA"
CLF_COL_BG  = "#F0A500"
ROW_ODD     = "#FFFFFF"
ROW_EVEN    = "#F7F7F7"
TEXT_DARK   = "#1A1A1A"
TEXT_WHITE  = "#FFFFFF"
BORDER_COL  = "#CCCCCC"


# ============================================================
#   DRAW TABLE IMAGE
# ============================================================

N_ROWS    = len(df)
N_METRICS = 6

COL_W_CLF = 0.80
COL_W_MET = 0.72
ROW_H     = 0.35
HEADER_H  = 0.40

TABLE_W = COL_W_CLF + COL_W_MET * N_METRICS * 2
TABLE_H = HEADER_H * 2 + ROW_H * N_ROWS

fig_w = TABLE_W + 0.6
fig_h = TABLE_H + 0.7

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, TABLE_W)
ax.set_ylim(0, TABLE_H)
ax.axis("off")

x_clf  = 0.0
x_cols = [x_clf + COL_W_CLF + i * COL_W_MET
          for i in range(N_METRICS * 2)]


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
        color=fc, zorder=2, clip_on=True
    )


# ── Row 1: Spanning group headers ─────────────────────────────
y_h1      = TABLE_H - HEADER_H
default_w = COL_W_MET * N_METRICS
optuna_w  = COL_W_MET * N_METRICS

cell_rect(x_clf, y_h1, COL_W_CLF, HEADER_H, HEADER_BG)

cell_rect(x_clf + COL_W_CLF, y_h1,
          default_w, HEADER_H, HEADER_BG,
          ec="#DDDDDD", lw=1.0)
cell_text(x_clf + COL_W_CLF, y_h1,
          default_w, HEADER_H,
          "Default", fc=TEXT_WHITE, fontsize=11, bold=True)

cell_rect(x_clf + COL_W_CLF + default_w, y_h1,
          optuna_w, HEADER_H, DIVIDER_COL,
          ec="#2060A0", lw=1.0)
cell_text(x_clf + COL_W_CLF + default_w, y_h1,
          optuna_w, HEADER_H,
          "Optuna", fc=TEXT_WHITE, fontsize=11, bold=True)


# ── Row 2: Metric sub-headers ─────────────────────────────────
y_h2           = TABLE_H - HEADER_H * 2
metric_display = ["Accuracy", "Sensitivity", "Specificity",
                  "F1\nScore", "MCC", "AUC"]

cell_rect(x_clf, y_h2, COL_W_CLF, HEADER_H, HEADER_BG)

for i in range(N_METRICS):
    cell_rect(x_cols[i], y_h2,
              COL_W_MET, HEADER_H, HEADER_BG,
              ec="#DDDDDD", lw=0.6)
    cell_text(x_cols[i], y_h2,
              COL_W_MET, HEADER_H,
              metric_display[i], fc=TEXT_WHITE,
              fontsize=7, bold=True)

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

# Collect all Optuna values to find top-3 per metric column
all_optuna_vals = {i: [] for i in range(N_METRICS)}
for _, row in df.iterrows():
    for ci, mk in enumerate(metric_order):
        v = row[f"O_{mk}"]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            all_optuna_vals[ci].append(float(v))

top3_per_col = {
    ci: sorted(set(all_optuna_vals[ci]), reverse=True)[:3]
    for ci in range(N_METRICS)
}


def optuna_cell_color(ci, o_val, b_val):
    """
    Returns (bg_color, text_color) for an Optuna metric cell.

    Priority:
      1. Top-3 value across all methods → blue highlight
      2. Improved vs default            → green
      3. Degraded vs default            → red
      4. Default                        → plain row bg
    """
    if o_val is None or (isinstance(o_val, float) and np.isnan(o_val)):
        return ROW_ODD, TEXT_DARK

    fv   = float(o_val)
    top3 = top3_per_col.get(ci, [])

    # Top-3 highlight (takes priority over green/red)
    if len(top3) > 0 and fv >= top3[0]:
        return "#1A5276", TEXT_WHITE    # darkest blue — top 1
    if len(top3) > 1 and fv >= top3[1]:
        return "#2980B9", TEXT_WHITE    # medium blue — top 2
    if len(top3) > 2 and fv >= top3[2]:
        return "#85C1E9", TEXT_DARK     # light blue — top 3

    # Delta vs default
    if (b_val is not None
            and not (isinstance(b_val, float) and np.isnan(b_val))):
        delta = fv - float(b_val)
        if delta > 0.001:
            return "#D5F5E3", "#1A6B34"   # light green — improved
        elif delta < -0.001:
            return "#FADBD8", "#922B21"   # light red — degraded

    return ROW_ODD, TEXT_DARK


for ri, (_, row) in enumerate(df.iterrows()):
    y_row  = TABLE_H - HEADER_H * 2 - ROW_H * (ri + 1)
    row_bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
    clf_bg = CLF_COL_BG if ri % 2 == 0 else "#E8950A"

    # Classifier name cell
    cell_rect(x_clf, y_row, COL_W_CLF, ROW_H, clf_bg)
    cell_text(x_clf, y_row, COL_W_CLF, ROW_H,
              row["CLF"], fc=TEXT_WHITE, fontsize=9, bold=True)

    # Default metric cells
    for i, mk in enumerate(metric_order):
        val = row[f"B_{mk}"]
        txt = (f"{val:.4f}"
               if val is not None
               and not (isinstance(val, float) and np.isnan(val))
               else "—")
        cell_rect(x_cols[i], y_row, COL_W_MET, ROW_H, row_bg)
        cell_text(x_cols[i], y_row, COL_W_MET, ROW_H,
                  txt, fc=TEXT_DARK, fontsize=8)

    # Optuna metric cells
    for i, mk in enumerate(metric_order):
        b_val   = row[f"B_{mk}"]
        o_val   = row[f"O_{mk}"]
        txt     = (f"{o_val:.4f}"
                   if o_val is not None
                   and not (isinstance(o_val, float) and np.isnan(o_val))
                   else "—")
        cell_bg, txt_col = optuna_cell_color(i, o_val, b_val)
        cell_rect(x_cols[i + N_METRICS], y_row,
                  COL_W_MET, ROW_H, cell_bg,
                  ec=BORDER_COL, lw=0.6)
        cell_text(x_cols[i + N_METRICS], y_row,
                  COL_W_MET, ROW_H, txt,
                  fc=txt_col, fontsize=8)


# ── Outer border ──────────────────────────────────────────────
ax.add_patch(plt.Rectangle(
    (0, 0), TABLE_W, TABLE_H,
    fill=False, edgecolor="#333333",
    linewidth=1.5, zorder=3
))

# ── Vertical divider between Default and Optuna ───────────────
div_x = x_clf + COL_W_CLF + COL_W_MET * N_METRICS
ax.plot([div_x, div_x], [0, TABLE_H],
        color=DIVIDER_COL, linewidth=2.0, zorder=3)


# ── Title ─────────────────────────────────────────────────────
ax.text(
    TABLE_W / 2, TABLE_H + 0.12,
    "Performance Comparison: Default vs Optuna-Tuned Classifiers",
    ha="center", va="bottom",
    fontsize=12, fontweight="bold", color="#1A1A1A",
    transform=ax.transData
)
ax.text(
    TABLE_W / 2, TABLE_H + 0.03,
    "(Best dataset per classifier — AIP Prediction)",
    ha="center", va="bottom",
    fontsize=8, color="#555555",
    transform=ax.transData
)


# ── Legend ────────────────────────────────────────────────────
legend_y = -0.18
items = [
    ("#D5F5E3", "#1A6B34", "Improved after Optuna tuning"),
    ("#FADBD8", "#922B21", "Degraded after Optuna tuning"),
    ("#1A5276", TEXT_WHITE, "Top-1 value"),
    ("#2980B9", TEXT_WHITE, "Top-2 value"),
    ("#85C1E9", TEXT_DARK,  "Top-3 value"),
]
lx = 0.0
for bg, fc, label in items:
    ax.add_patch(plt.Rectangle(
        (lx, legend_y), 0.16, 0.12,
        facecolor=bg, edgecolor="#AAAAAA",
        linewidth=0.6, transform=ax.transData
    ))
    ax.text(lx + 0.18, legend_y + 0.06,
            label, va="center", fontsize=7.5,
            color=fc, transform=ax.transData)
    lx += 1.65

plt.subplots_adjust(left=0.02, right=0.98,
                    top=0.90, bottom=0.10)


# ── Save ──────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "ml_comparison_table_v2.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white")
plt.show()
print(f"\n✅ Table saved → {out_path}")
print(f"   10 classifiers only — no stacking row")
print(f"   Green cells  = Optuna improved  |  Red = degraded")
print(f"   Blue cells   = Top-1/2/3 value across all methods")