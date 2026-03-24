

# ============================================================
#   CELL 1 — Mount Google Drive
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================
#   CELL 2 — Install & Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

print("✅ Libraries loaded")


# ============================================================
#   CELL 3 — Configuration (Edit paths here)
# ============================================================

# LOCAL DATA PATH
FASTA_PATH  = "../../data/raw/AIP_ind.fasta"

# DRIVE DATA PATH
# FASTA_PATH  = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/data/raw/AIP_ind.fasta"

# Output will be saved here (ONLY for Google Colab, ignored in local runs)
# OUTPUT_DIR  = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/data/features"

# Output will be saved here (ONLY for local runs, ignored in Google Colab)
OUTPUT_DIR  = "../../data/features"

OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "GAAC_features.csv")

# Output will be saved here (ONLY for Google Colab, ignored in local runs)
# FIGURES_DIR = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/results/figures"

# Output will be saved here (ONLY for local runs, ignored in Google Colab)
FIGURES_DIR = "../../results/figures"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"✅ Features dir : {OUTPUT_DIR}")
print(f"✅ Figures dir  : {FIGURES_DIR}")


# ============================================================
#   CELL 4 — FASTA Parser
# ============================================================

def parse_fasta(filepath):
    """Parse FASTA file → DataFrame with seq_id, sequence, label"""
    entries    = []
    current_id = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:]
            elif line and current_id:
                label = 1 if current_id.startswith("pos") else 0
                entries.append({
                    "seq_id"  : current_id,
                    "sequence": line.upper(),
                    "label"   : label
                })

    df = pd.DataFrame(entries)
    print(f"✅ Parsed {len(df)} sequences")
    print(f"   Positive (AIP)    : {df['label'].sum()}")
    print(f"   Negative (non-AIP): {(df['label']==0).sum()}")
    return df

df_seq = parse_fasta(FASTA_PATH)
df_seq.head(3)


# ============================================================
#   CELL 5 — Define GAAC Groups
# ============================================================
#
#  GAAC (Grouped Amino Acid Composition) reduces the 20
#  standard amino acids into 5 physicochemical groups.
#
#  Instead of 20 individual frequencies (AAC), GAAC gives
#  just 5 features — one fraction per group.
#
#  Groups (Shen & Chou, 2001):
#    G1 — Aliphatic  : A, G, V, I, L, M
#    G2 — Aromatic   : F, Y, W
#    G3 — Positive   : K, R, H
#    G4 — Negative   : D, E
#    G5 — Uncharged  : S, T, C, P, N, Q
#
# ============================================================

GAAC_GROUPS = {
    "GAAC_Aliphatic" : set("AGVILM"),   # G1
    "GAAC_Aromatic"  : set("FYW"),      # G2
    "GAAC_Positive"  : set("KRH"),      # G3
    "GAAC_Negative"  : set("DE"),       # G4
    "GAAC_Uncharged" : set("STCPNQ"),   # G5
}

# Verify all 20 amino acids are covered exactly once
all_covered = set("".join("".join(v) for v in GAAC_GROUPS.values()))
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

print("── GAAC Group Definition ───────────────────────────")
for name, members in GAAC_GROUPS.items():
    print(f"  {name:20s}: {''.join(sorted(members))}")

print(f"\n  All 20 AAs covered : {'✅ Yes' if all_covered == AMINO_ACIDS else '⚠️  No'}")
print(f"  Total features     : {len(GAAC_GROUPS)}")


# ============================================================
#   CELL 6 — GAAC Feature Extraction
# ============================================================

def compute_gaac(sequence):
    """
    GAAC — Grouped Amino Acid Composition:
    Fraction of residues belonging to each of 5
    physicochemical groups.

    Formula:
        GAAC(group_g) = count(residues in group_g) / len(sequence)

    Example: sequence = "ACDEFGHIK" (length = 9)
        Aliphatic (AGVILM) → A, G  → count = 2  → 2/9 = 0.2222
        Aromatic  (FYW)    → F     → count = 1  → 1/9 = 0.1111
        Positive  (KRH)    → H, I, K → wait, I is Aliphatic
                           → H, K  → count = 2  → 2/9 = 0.2222
        Negative  (DE)     → D, E  → count = 2  → 2/9 = 0.2222
        Uncharged (STCPNQ) → C     → count = 1  → 1/9 = 0.1111
        (sum = 1.0 ✅)

    Returns: dict of 5 features
        e.g. {'GAAC_Aliphatic': 0.22, 'GAAC_Aromatic': 0.11, ...}
    """
    length = len(sequence)
    return {
        group_name: round(
            sum(1 for aa in sequence if aa in aa_set) / length, 6
        )
        for group_name, aa_set in GAAC_GROUPS.items()
    }


def extract_gaac(df):
    """Apply GAAC extraction to full dataframe."""
    print("Extracting GAAC features... (5 per sequence)")
    gaac_records = df["sequence"].apply(compute_gaac)
    gaac_df      = pd.DataFrame(list(gaac_records))
    result       = pd.concat([df.reset_index(drop=True), gaac_df], axis=1)
    # ── REMOVED ──────────────────────────────────────────────
    # result.insert(2, "length", result["sequence"].apply(len))
    # ─────────────────────────────────────────────────────────────────
    return result

df_gaac = extract_gaac(df_seq)

# ── CHANGED (Block): drop sequence & length, move label to end ──
df_gaac = df_gaac.drop(columns=["sequence"])                        # ← remove sequence column
cols   = [c for c in df_gaac.columns if c != "label"] + ["label"] # ← move label to end
df_gaac = df_gaac[cols]                                             # ← reorder columns
# ────────────────────────────────────────────────────────────────

gaac_cols = [c for c in df_gaac.columns if c.startswith("GAAC_")]

print(f"\n✅ GAAC extraction complete")
print(f"   Shape    : {df_gaac.shape}")
print(f"   Features : {gaac_cols}")
df_gaac.head(5)


# ============================================================
#   CELL 7 — Validate Output
# ============================================================

print("── Validation ──────────────────────────────────────")
print(f"Total GAAC features : {len(gaac_cols)}")
print(f"Missing values      : {df_gaac[gaac_cols].isnull().sum().sum()}")
print(f"Value range         : {df_gaac[gaac_cols].values.min():.4f} – "
      f"{df_gaac[gaac_cols].values.max():.4f}")

row_sums = df_gaac[gaac_cols].sum(axis=1).round(4)
print(f"All rows sum to 1.0 : {'✅ Yes' if row_sums.eq(1.0).all() else '⚠️  No'}")
print(f"  Min row sum : {row_sums.min()}")
print(f"  Max row sum : {row_sums.max()}")


# ============================================================
#   CELL 8 — Statistical Summary: Pos vs Neg
# ============================================================

stats = df_gaac.groupby("label")[gaac_cols].mean().T
stats.columns = ["Negative (0)", "Positive (1)"]
stats["Difference"] = (stats["Positive (1)"] - stats["Negative (0)"]).round(6)
stats = stats.sort_values("Difference", ascending=False)

print("── GAAC Mean Values: Positive vs Negative ──────────")
print(stats.to_string())


# ============================================================
#   CELL 9 — Visualization 1: Grouped Bar Chart
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

x         = np.arange(len(gaac_cols))
width     = 0.35
pos_means = df_gaac[df_gaac["label"] == 1][gaac_cols].mean().values
neg_means = df_gaac[df_gaac["label"] == 0][gaac_cols].mean().values

bars1 = ax.bar(x - width/2, pos_means, width,
               label="AIP (pos)", color="#2ecc71",
               alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + width/2, neg_means, width,
               label="Non-AIP (neg)", color="#e74c3c",
               alpha=0.85, edgecolor="white")

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8, color="#27ae60")

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8, color="#c0392b")

short_labels = [c.replace("GAAC_", "") for c in gaac_cols]
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=11)
ax.set_ylabel("Mean Fraction", fontsize=12)
ax.set_title("GAAC: Mean Group Composition — AIP vs Non-AIP",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(pos_means.max(), neg_means.max()) + 0.05)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "GAAC_mean_comparison.png"), dpi=150)
plt.show()
print("✅ Bar chart saved")


# ============================================================
#   CELL 10 — Visualization 2: Pie Charts (Pos vs Neg)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

short_labels = [c.replace("GAAC_", "") for c in gaac_cols]
colors       = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]

for ax, lbl, label_val in zip(axes, ["AIP (Positive)", "Non-AIP (Negative)"], [1, 0]):
    group_data = df_gaac[df_gaac["label"] == label_val][gaac_cols].mean().values
    wedges, texts, autotexts = ax.pie(
        group_data,
        labels=short_labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title(lbl, fontsize=12, fontweight="bold")

plt.suptitle("GAAC: Amino Acid Group Composition",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "GAAC_pie_charts.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Pie charts saved")


# ============================================================
#   CELL 11 — Visualization 3: Discriminative Feature Bar Plot
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

diff   = stats["Difference"]
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in diff]
short  = [c.replace("GAAC_", "") for c in diff.index]

bars = ax.bar(range(len(diff)), diff.values,
              color=colors, edgecolor="white", width=0.5)

# Add value labels
for bar, val in zip(bars, diff.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + (0.001 if val >= 0 else -0.003),
            f"{val:+.4f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(diff)))
ax.set_xticklabels(short, fontsize=11)
ax.set_xlabel("GAAC Group", fontsize=12)
ax.set_ylabel("Mean Difference (Pos − Neg)", fontsize=12)
ax.set_title("Discriminative Power of GAAC Features",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="Higher in AIP"),
                   Patch(facecolor="#e74c3c", label="Higher in non-AIP")]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "GAAC_discriminative_features.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Discriminative plot saved")


# ============================================================
#   CELL 12 — Visualization 4: Box Plot (Distribution per Group)
# ============================================================

fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)

for ax, col in zip(axes, gaac_cols):
    data_pos = df_gaac[df_gaac["label"] == 1][col].values
    data_neg = df_gaac[df_gaac["label"] == 0][col].values

    bp = ax.boxplot(
        [data_pos, data_neg],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2)
    )
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][1].set_facecolor("#e74c3c")
    bp["boxes"][1].set_alpha(0.75)

    ax.set_title(col.replace("GAAC_", ""), fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["AIP", "Non-AIP"], fontsize=9)
    ax.set_ylabel("Fraction", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("GAAC: Group Fraction Distribution (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "GAAC_boxplots.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Box plots saved")


# ============================================================
#   CELL 13 — GAAC vs AAC: Correlation with Label
# ============================================================

# Load AAC if available and compare label correlations
aac_path = os.path.join(OUTPUT_DIR, "AAC_features.csv")

if os.path.exists(aac_path):
    df_aac    = pd.read_csv(aac_path)
    aac_cols  = [c for c in df_aac.columns if c.startswith("AAC_")]

    aac_corr  = df_aac[aac_cols + ["label"]].corr()["label"].drop("label").abs()
    gaac_corr = df_gaac[gaac_cols + ["label"]].corr()["label"].drop("label").abs()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar(range(len(aac_corr)), aac_corr.values,
                color="#3498db", alpha=0.8, edgecolor="white")
    axes[0].set_title("AAC — |Correlation| with Label (20 features)",
                      fontweight="bold")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("|Pearson Correlation|")
    axes[0].set_xticks(range(len(aac_corr)))
    axes[0].set_xticklabels([c.replace("AAC_", "") for c in aac_corr.index],
                             fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(gaac_corr)), gaac_corr.values,
                color="#e67e22", alpha=0.8, edgecolor="white")
    axes[1].set_title("GAAC — |Correlation| with Label (5 features)",
                      fontweight="bold")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("|Pearson Correlation|")
    axes[1].set_xticks(range(len(gaac_corr)))
    axes[1].set_xticklabels([c.replace("GAAC_", "") for c in gaac_corr.index],
                             fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Label Correlation Comparison: AAC vs GAAC",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "GAAC_vs_AAC_correlation.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ AAC vs GAAC correlation plot saved")
else:
    print("ℹ️  AAC file not found — skipping comparison plot")
    print(f"   Run 01_AAC_feature_extraction.py first to enable this plot")


# ============================================================
#   CELL 14 — Save Final CSV
# ============================================================

df_gaac.to_csv(OUTPUT_CSV, index=False)

print("=" * 55)
print("✅ GAAC Feature Extraction COMPLETE")
print(f"   File     : {OUTPUT_CSV}")
print(f"   Shape    : {df_gaac.shape}")
print(f"   Features : 5 (Aliphatic, Aromatic, Positive,")
print(f"                   Negative, Uncharged)")
print(f"   Figures  : {FIGURES_DIR}")
print("=" * 55)