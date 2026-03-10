
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

FASTA_PATH  = "../../data/raw/AIP_ind.fasta"

OUTPUT_DIR  = "../../data/features"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "CTDC_features.csv")

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
#   CELL 5 — Define CTD Property Groups
# ============================================================
#
#  CTD divides the 20 amino acids into 3 groups for each of
#  7 physicochemical properties (Dubchak et al., 1995).
#
#  CTDC (Composition) = fraction of residues in each group
#  → 7 properties × 3 groups = 21 features total
#
# ============================================================

CTD_PROPERTIES = {

    # 1. Hydrophobicity
    #    Ref: Kyte & Doolittle scale
    "Hydrophobicity": {
        "G1": set("RKEDQN"),      # Polar
        "G2": set("GASTPHY"),     # Neutral
        "G3": set("CLVIMFW"),     # Hydrophobic
    },

    # 2. Normalized Van der Waals Volume
    #    Ref: Zimmerman et al.
    "VanDerWaals": {
        "G1": set("GASTPDC"),     # Small
        "G2": set("NVEQIL"),      # Medium
        "G3": set("MHKFRYW"),     # Large
    },

    # 3. Polarity
    #    Ref: Grantham scale
    "Polarity": {
        "G1": set("LIFWCMVY"),    # Low polarity
        "G2": set("PATGS"),       # Medium polarity
        "G3": set("HQRKNED"),     # High polarity
    },

    # 4. Polarizability
    #    Ref: Charton & Charton
    "Polarizability": {
        "G1": set("GASDT"),       # Low
        "G2": set("CPNVEQIL"),    # Medium
        "G3": set("KMHFRYW"),     # High
    },

    # 5. Charge
    #    Based on net charge at pH 7
    "Charge": {
        "G1": set("KR"),          # Positive
        "G2": set("ANCQGHILMFPSTWYV"),  # Neutral
        "G3": set("DE"),          # Negative
    },

    # 6. Secondary Structure tendency
    #    Ref: Chou-Fasman parameters
    "SecondaryStructure": {
        "G1": set("EALMQKRH"),    # Helix
        "G2": set("VIYCWFT"),     # Strand / Sheet
        "G3": set("GNPSD"),       # Coil / Turn
    },

    # 7. Solvent Accessibility
    #    Ref: Janin scale
    "SolventAccessibility": {
        "G1": set("ALFCGIVW"),    # Buried
        "G2": set("RKQEND"),      # Intermediate
        "G3": set("MSPTHY"),      # Exposed
    },
}

# Print summary of groupings
print("── CTD Property Groups ─────────────────────────────")
for prop, groups in CTD_PROPERTIES.items():
    print(f"\n  {prop}")
    for gname, members in groups.items():
        print(f"    {gname}: {''.join(sorted(members))}")

print(f"\n✅ Total CTDC features: {len(CTD_PROPERTIES)} properties × 3 groups = "
      f"{len(CTD_PROPERTIES)*3}")


# ============================================================
#   CELL 6 — CTDC Feature Extraction
# ============================================================

def compute_ctdc(sequence):
    """
    CTDC — Composition of CTD:
    For each property, compute the fraction of residues
    that fall into each of the 3 defined groups.

    Formula:
        CTDC(property_p, group_g) =
            count(residues in group_g) / len(sequence)

    Returns: dict of 21 features
        e.g. {'CTDC_Hydrophobicity_G1': 0.15,
               'CTDC_Hydrophobicity_G2': 0.35,
               'CTDC_Hydrophobicity_G3': 0.50, ...}
    """
    length  = len(sequence)
    features = {}

    for prop_name, groups in CTD_PROPERTIES.items():
        for group_name, aa_set in groups.items():
            count = sum(1 for aa in sequence if aa in aa_set)
            key   = f"CTDC_{prop_name}_{group_name}"
            features[key] = round(count / length, 6)

    return features


def extract_ctdc(df):
    """Apply CTDC extraction to full dataframe."""
    print("Extracting CTDC features... (21 per sequence)")
    ctdc_records = df["sequence"].apply(compute_ctdc)
    ctdc_df      = pd.DataFrame(list(ctdc_records))
    result       = pd.concat([df.reset_index(drop=True), ctdc_df], axis=1)
    result.insert(2, "length", result["sequence"].apply(len))
    return result

df_ctdc = extract_ctdc(df_seq)

print(f"\n✅ CTDC extraction complete")
print(f"   Shape    : {df_ctdc.shape}")
ctdc_cols = [c for c in df_ctdc.columns if c.startswith("CTDC_")]
print(f"   Features : {ctdc_cols}")
df_ctdc.head(3)


# ============================================================
#   CELL 7 — Validate Output
# ============================================================

print("── Validation ──────────────────────────────────────")
print(f"Total CTDC features : {len(ctdc_cols)}")
print(f"Missing values      : {df_ctdc[ctdc_cols].isnull().sum().sum()}")
print(f"Value range         : {df_ctdc[ctdc_cols].values.min():.4f} – "
      f"{df_ctdc[ctdc_cols].values.max():.4f}")

# For each property, the 3 groups must sum to 1.0
print("\n── Group sum check (each property must sum to 1.0) ──")
for prop in CTD_PROPERTIES.keys():
    prop_cols = [f"CTDC_{prop}_G{i}" for i in range(1, 4)]
    row_sums  = df_ctdc[prop_cols].sum(axis=1).round(4)
    ok        = row_sums.eq(1.0).all()
    print(f"  {prop:25s}: {'✅ OK' if ok else '⚠️  FAIL'}")


# ============================================================
#   CELL 8 — Statistical Summary: Pos vs Neg
# ============================================================

stats = df_ctdc.groupby("label")[ctdc_cols].mean().T
stats.columns = ["Negative (0)", "Positive (1)"]
stats["Difference"] = (stats["Positive (1)"] - stats["Negative (0)"]).round(6)
stats = stats.sort_values("Difference", ascending=False)

print("── CTDC Mean Values: Positive vs Negative ──────────")
print(stats.to_string())


# ============================================================
#   CELL 9 — Visualization 1: Grouped Bar Chart per Property
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes      = axes.flatten()

for idx, prop in enumerate(CTD_PROPERTIES.keys()):
    ax        = axes[idx]
    prop_cols = [f"CTDC_{prop}_G{i}" for i in range(1, 4)]
    labels    = ["G1", "G2", "G3"]

    pos_means = df_ctdc[df_ctdc["label"] == 1][prop_cols].mean().values
    neg_means = df_ctdc[df_ctdc["label"] == 0][prop_cols].mean().values

    x     = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, pos_means, width, label="AIP (pos)",
           color="#2ecc71", alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, neg_means, width, label="Non-AIP (neg)",
           color="#e74c3c", alpha=0.85, edgecolor="white")

    ax.set_title(prop, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Fraction", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

# Hide the last unused subplot
axes[-1].set_visible(False)

plt.suptitle("CTDC: Mean Composition per Property Group (AIP vs Non-AIP)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "CTDC_grouped_bar_chart.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Grouped bar chart saved")


# ============================================================
#   CELL 10 — Visualization 2: Heatmap of All 21 CTDC Features
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

# Build matrix: rows = 21 features, cols = [Neg mean, Pos mean]
heatmap_data = stats[["Negative (0)", "Positive (1)"]].T
short_labels = [c.replace("CTDC_", "").replace("_G", "\nG")
                for c in stats.index]

sns.heatmap(
    heatmap_data,
    xticklabels=short_labels,
    yticklabels=["Non-AIP (neg)", "AIP (pos)"],
    cmap="YlOrRd",
    annot=True,
    fmt=".3f",
    linewidths=0.5,
    linecolor="grey",
    ax=ax,
    cbar_kws={"label": "Mean Fraction"}
)
ax.set_title("CTDC Feature Heatmap — Mean Values (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("CTDC Features", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "CTDC_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Heatmap saved")


# ============================================================
#   CELL 11 — Visualization 3: Discriminative Feature Bar Plot
# ============================================================

fig, ax = plt.subplots(figsize=(13, 5))

diff   = stats["Difference"]
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in diff]
short  = [c.replace("CTDC_", "").replace("_", "\n") for c in diff.index]

ax.bar(range(len(diff)), diff.values, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(diff)))
ax.set_xticklabels(short, fontsize=8)
ax.set_xlabel("CTDC Feature", fontsize=12)
ax.set_ylabel("Mean Difference (Pos − Neg)", fontsize=12)
ax.set_title("Discriminative Power of CTDC Features",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="Higher in AIP"),
                   Patch(facecolor="#e74c3c", label="Higher in non-AIP")]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "CTDC_discriminative_features.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Discriminative feature plot saved")


# ============================================================
#   CELL 12 — Save Final CSV
# ============================================================

df_ctdc.to_csv(OUTPUT_CSV, index=False)

print("=" * 55)
print("✅ CTDC Feature Extraction COMPLETE")
print(f"   File     : {OUTPUT_CSV}")
print(f"   Shape    : {df_ctdc.shape}")
print(f"   Features : 7 properties × 3 groups = 21 CTDC features")
print(f"   Figures  : {FIGURES_DIR}")
print("=" * 55)