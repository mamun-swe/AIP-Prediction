

# ============================================================
#   CELL 2 — Install & Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import product
import os

print("✅ Libraries loaded")


# ============================================================
#   CELL 3 — Configuration (Edit paths here)
# ============================================================

FASTA_PATH  = "../../data/raw/AIP_ind.fasta"

# Features saved here
OUTPUT_DIR  = "../../data/features"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "DPC_features.csv")

# Images saved here
FIGURES_DIR = "../../results/figures"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"✅ Features dir : {OUTPUT_DIR}")
print(f"✅ Figures dir  : {FIGURES_DIR}")


# ============================================================
#   CELL 4 — FASTA Parser
# ============================================================

def parse_fasta(filepath):
    """
    Parse a FASTA file into a DataFrame.
    Labels inferred from header: pos → 1, neg → 0
    """
    entries = []
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
#   CELL 5 — DPC Feature Extraction
# ============================================================

# 20 standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# All 400 dipeptide combinations (AA × AA)
ALL_DIPEPTIDES = [a + b for a, b in product(AMINO_ACIDS, repeat=2)]

print(f"Total dipeptide features: {len(ALL_DIPEPTIDES)}")  # 400


def compute_dpc(sequence):
    """
    Dipeptide Composition (DPC):
    Frequency of each consecutive amino acid pair in a sequence.

    Formula:
        DPC(aa_i, aa_j) = count(aa_i aa_j) / (len(sequence) - 1)

    Example: sequence = "ACDE"
        dipeptides → AC, CD, DE
        DPC(AC) = 1/3 = 0.3333

    Returns: dict of 400 features  e.g. {'DPC_AA': 0.0, 'DPC_AC': 0.333, ...}
    """
    length = len(sequence)

    # Need at least 2 residues for a dipeptide
    if length < 2:
        return {f"DPC_{dp}": 0.0 for dp in ALL_DIPEPTIDES}

    # Extract all consecutive dipeptides
    dipeptides = [sequence[i:i+2] for i in range(length - 1)]
    total      = len(dipeptides)          # = length - 1
    counts     = Counter(dipeptides)

    return {f"DPC_{dp}": round(counts.get(dp, 0) / total, 6)
            for dp in ALL_DIPEPTIDES}


def extract_dpc(df):
    """Apply DPC extraction to full dataframe."""
    print("Extracting DPC features... (400 per sequence)")
    dpc_records = df["sequence"].apply(compute_dpc)
    dpc_df      = pd.DataFrame(list(dpc_records))
    result      = pd.concat([df.reset_index(drop=True), dpc_df], axis=1)
    result.insert(2, "length", result["sequence"].apply(len))
    return result

df_dpc = extract_dpc(df_seq)

print(f"\n✅ DPC extraction complete")
print(f"   Shape   : {df_dpc.shape}  (sequences × features)")
df_dpc.iloc[:3, :8]   # preview first few columns


# ============================================================
#   CELL 6 — Validate Output
# ============================================================

dpc_cols = [c for c in df_dpc.columns if c.startswith("DPC_")]

print("── Validation ──────────────────────────────────────")
print(f"Total DPC features   : {len(dpc_cols)}")
print(f"Missing values       : {df_dpc[dpc_cols].isnull().sum().sum()}")

# Each row should sum to 1.0 (allow tiny float tolerance)
row_sums = df_dpc[dpc_cols].sum(axis=1).round(4)
print(f"All rows sum to 1.0  : {row_sums.eq(1.0).all()}")
print(f"Value range          : {df_dpc[dpc_cols].values.min():.4f} – "
      f"{df_dpc[dpc_cols].values.max():.4f}")

# How many dipeptides never appear (all zeros)?
zero_cols = (df_dpc[dpc_cols] == 0).all(axis=0).sum()
print(f"Zero-only features   : {zero_cols} / {len(dpc_cols)}")
print("✅ Validation complete")


# ============================================================
#   CELL 7 — Statistical Summary: Top Discriminative DPCs
# ============================================================

stats = df_dpc.groupby("label")[dpc_cols].mean().T
stats.columns = ["Negative (0)", "Positive (1)"]
stats["Difference"] = (stats["Positive (1)"] - stats["Negative (0)"]).round(6)
stats = stats.sort_values("Difference", ascending=False)

print("── Top 15 DPC features higher in AIP (Positive) ──")
print(stats.head(15).to_string())

print("\n── Top 15 DPC features higher in non-AIP (Negative) ──")
print(stats.tail(15).to_string())


# ============================================================
#   CELL 8 — Visualization 1: Top 30 Discriminative DPC Bar Plot
# ============================================================

# Select top 15 pos + top 15 neg discriminative features
top_pos = stats.head(15)
top_neg = stats.tail(15)
top30   = pd.concat([top_pos, top_neg])

fig, ax = plt.subplots(figsize=(16, 6))

colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top30["Difference"]]
ax.bar(range(len(top30)), top30["Difference"].values, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)

ax.set_xticks(range(len(top30)))
ax.set_xticklabels([c.replace("DPC_", "") for c in top30.index],
                   rotation=90, fontsize=9)
ax.set_xlabel("Dipeptide Feature", fontsize=12)
ax.set_ylabel("Mean Difference (Pos − Neg)", fontsize=12)
ax.set_title("Top 30 Discriminative DPC Features (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="Higher in AIP"),
                   Patch(facecolor="#e74c3c", label="Higher in non-AIP")]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "DPC_discriminative_features.png"), dpi=150)
plt.show()
print("✅ Plot saved")


# ============================================================
#   CELL 9 — Visualization 2: DPC 20×20 Heatmap (Mean Difference)
# ============================================================

# Build 20×20 matrix of mean differences
diff_matrix = np.zeros((20, 20))
for i, aa1 in enumerate(AMINO_ACIDS):
    for j, aa2 in enumerate(AMINO_ACIDS):
        dp  = f"DPC_{aa1}{aa2}"
        val = stats.loc[dp, "Difference"] if dp in stats.index else 0.0
        diff_matrix[i][j] = val

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    diff_matrix,
    xticklabels=AMINO_ACIDS,
    yticklabels=AMINO_ACIDS,
    cmap="RdYlGn",
    center=0,
    annot=False,
    linewidths=0.3,
    linecolor="grey",
    ax=ax,
    cbar_kws={"label": "Mean Difference (Pos − Neg)"}
)
ax.set_title("DPC Feature Difference Heatmap (AIP vs Non-AIP)\n"
             "Green = higher in AIP  |  Red = higher in non-AIP",
             fontsize=13, fontweight="bold")
ax.set_xlabel("2nd Amino Acid", fontsize=11)
ax.set_ylabel("1st Amino Acid", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "DPC_difference_heatmap.png"), dpi=150)
plt.show()
print("✅ Heatmap saved")


# ============================================================
#   CELL 10 — Visualization 3: DPC Distribution (Pos vs Neg)
#             for Top 6 most discriminative dipeptides
# ============================================================

top6 = list(stats.head(3).index) + list(stats.tail(3).index)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, feat in enumerate(top6):
    ax  = axes[idx]
    pos_vals = df_dpc[df_dpc["label"] == 1][feat]
    neg_vals = df_dpc[df_dpc["label"] == 0][feat]

    ax.hist(pos_vals, bins=20, alpha=0.65, color="#2ecc71",
            label="AIP (pos)", edgecolor="white")
    ax.hist(neg_vals, bins=20, alpha=0.65, color="#e74c3c",
            label="Non-AIP (neg)", edgecolor="white")

    ax.set_title(feat.replace("DPC_", "Dipeptide: "), fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Distribution of Top 6 Discriminative DPC Features",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "DPC_top6_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Distribution plots saved")


# ============================================================
#   CELL 11 — Save Final CSV
# ============================================================

df_dpc.to_csv(OUTPUT_CSV, index=False)

print("=" * 55)
print("✅ DPC Feature Extraction COMPLETE")
print(f"   File     : {OUTPUT_CSV}")
print(f"   Shape    : {df_dpc.shape}")
print(f"   Columns  : seq_id, sequence, length, label, DPC_AA...DPC_YY")
print(f"   Figures  : {FIGURES_DIR}")
print("=" * 55)