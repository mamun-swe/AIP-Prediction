

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
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "CTDT_features.csv")

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
#            (Same groups used in CTDC — consistent reference)
# ============================================================

CTD_PROPERTIES = {

    "Hydrophobicity": {
        "G1": set("RKEDQN"),
        "G2": set("GASTPHY"),
        "G3": set("CLVIMFW"),
    },
    "VanDerWaals": {
        "G1": set("GASTPDC"),
        "G2": set("NVEQIL"),
        "G3": set("MHKFRYW"),
    },
    "Polarity": {
        "G1": set("LIFWCMVY"),
        "G2": set("PATGS"),
        "G3": set("HQRKNED"),
    },
    "Polarizability": {
        "G1": set("GASDT"),
        "G2": set("CPNVEQIL"),
        "G3": set("KMHFRYW"),
    },
    "Charge": {
        "G1": set("KR"),
        "G2": set("ANCQGHILMFPSTWYV"),
        "G3": set("DE"),
    },
    "SecondaryStructure": {
        "G1": set("EALMQKRH"),
        "G2": set("VIYCWFT"),
        "G3": set("GNPSD"),
    },
    "SolventAccessibility": {
        "G1": set("ALFCGIVW"),
        "G2": set("RKQEND"),
        "G3": set("MSPTHY"),
    },
}

# 3 transition pairs per property: G1↔G2, G1↔G3, G2↔G3
TRANSITION_PAIRS = [("G1", "G2"), ("G1", "G3"), ("G2", "G3")]

print("── CTD Properties loaded ───────────────────────────")
print(f"   Properties  : {len(CTD_PROPERTIES)}")
print(f"   Transitions : {len(TRANSITION_PAIRS)} pairs per property")
print(f"   Total CTDT  : {len(CTD_PROPERTIES) * len(TRANSITION_PAIRS)} features")


# ============================================================
#   CELL 6 — CTDT Feature Extraction
# ============================================================

def get_group(aa, groups):
    """
    Return the group label (G1/G2/G3) for a given amino acid
    within a property's group dictionary.
    Returns None if amino acid is not found.
    """
    for group_name, aa_set in groups.items():
        if aa in aa_set:
            return group_name
    return None


def compute_ctdt(sequence):
    """
    CTDT — Transition of CTD:
    For each property, count transitions between group pairs
    in consecutive amino acid positions.

    A transition occurs when residue[i] and residue[i+1]
    belong to DIFFERENT groups.

    Formula:
        CTDT(property_p, Gx↔Gy) =
            count(transitions between Gx and Gy)
            ─────────────────────────────────────
                    len(sequence) - 1

    Example:
        sequence = "RKDE"   (Hydrophobicity groups: G1 G1 G3 G1)
        transitions: G1→G1 (no), G1→G3 (yes G1↔G3), G3→G1 (yes G1↔G3)
        CTDT(Hydrophobicity, G1↔G3) = 2 / 3 = 0.6667

    Returns: dict of 21 features
        e.g. {'CTDT_Hydrophobicity_G1G2': 0.05,
               'CTDT_Hydrophobicity_G1G3': 0.10,
               'CTDT_Hydrophobicity_G2G3': 0.20, ...}
    """
    length = len(sequence)
    features = {}

    # Need at least 2 residues for any transition
    if length < 2:
        for prop_name in CTD_PROPERTIES:
            for g1, g2 in TRANSITION_PAIRS:
                features[f"CTDT_{prop_name}_{g1}{g2}"] = 0.0
        return features

    total_pairs = length - 1   # number of consecutive pairs

    for prop_name, groups in CTD_PROPERTIES.items():

        # Map each residue to its group label for this property
        group_seq = [get_group(aa, groups) for aa in sequence]

        # Count each transition pair
        transition_counts = {f"{g1}{g2}": 0 for g1, g2 in TRANSITION_PAIRS}

        for i in range(total_pairs):
            curr = group_seq[i]
            nxt  = group_seq[i + 1]

            if curr is None or nxt is None:
                continue

            # Transition is symmetric: G1→G2 == G2→G1
            if curr != nxt:
                pair = tuple(sorted([curr, nxt]))
                key  = f"{pair[0]}{pair[1]}"
                if key in transition_counts:
                    transition_counts[key] += 1

        # Normalize by total consecutive pairs
        for g1, g2 in TRANSITION_PAIRS:
            key   = f"CTDT_{prop_name}_{g1}{g2}"
            count = transition_counts[f"{g1}{g2}"]
            features[key] = round(count / total_pairs, 6)

    return features


def extract_ctdt(df):
    """Apply CTDT extraction to full dataframe."""
    print("Extracting CTDT features... (21 per sequence)")
    ctdt_records = df["sequence"].apply(compute_ctdt)
    ctdt_df      = pd.DataFrame(list(ctdt_records))
    result       = pd.concat([df.reset_index(drop=True), ctdt_df], axis=1)
    result.insert(2, "length", result["sequence"].apply(len))
    return result

df_ctdt = extract_ctdt(df_seq)

ctdt_cols = [c for c in df_ctdt.columns if c.startswith("CTDT_")]

print(f"\n✅ CTDT extraction complete")
print(f"   Shape    : {df_ctdt.shape}")
print(f"   Features : {len(ctdt_cols)}")
df_ctdt.head(3)


# ============================================================
#   CELL 7 — Validate Output
# ============================================================

print("── Validation ──────────────────────────────────────")
print(f"Total CTDT features : {len(ctdt_cols)}")
print(f"Missing values      : {df_ctdt[ctdt_cols].isnull().sum().sum()}")
print(f"Value range         : {df_ctdt[ctdt_cols].values.min():.4f} – "
      f"{df_ctdt[ctdt_cols].values.max():.4f}")

# Transition values must be between 0 and 1
all_valid = ((df_ctdt[ctdt_cols] >= 0) & (df_ctdt[ctdt_cols] <= 1)).all().all()
print(f"All values in [0,1] : {'✅ OK' if all_valid else '⚠️  FAIL'}")

# Check sum of transitions ≤ 1 per property (not strict equality)
print("\n── Transition sum check (must be ≤ 1.0 per property) ──")
for prop in CTD_PROPERTIES.keys():
    prop_cols = [f"CTDT_{prop}_{g1}{g2}" for g1, g2 in TRANSITION_PAIRS]
    row_sums  = df_ctdt[prop_cols].sum(axis=1).round(4)
    ok        = (row_sums <= 1.001).all()
    print(f"  {prop:25s}: {'✅ OK' if ok else '⚠️  FAIL'} "
          f"(mean sum = {row_sums.mean():.4f})")


# ============================================================
#   CELL 8 — Statistical Summary: Pos vs Neg
# ============================================================

stats = df_ctdt.groupby("label")[ctdt_cols].mean().T
stats.columns = ["Negative (0)", "Positive (1)"]
stats["Difference"] = (stats["Positive (1)"] - stats["Negative (0)"]).round(6)
stats = stats.sort_values("Difference", ascending=False)

print("── CTDT Mean Transition Values: Positive vs Negative ──")
print(stats.to_string())


# ============================================================
#   CELL 9 — Visualization 1: Grouped Bar Chart per Property
# ============================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

transition_labels = ["G1↔G2", "G1↔G3", "G2↔G3"]

for idx, prop in enumerate(CTD_PROPERTIES.keys()):
    ax        = axes[idx]
    prop_cols = [f"CTDT_{prop}_{g1}{g2}" for g1, g2 in TRANSITION_PAIRS]

    pos_means = df_ctdt[df_ctdt["label"] == 1][prop_cols].mean().values
    neg_means = df_ctdt[df_ctdt["label"] == 0][prop_cols].mean().values

    x     = np.arange(3)
    width = 0.35

    ax.bar(x - width/2, pos_means, width, label="AIP (pos)",
           color="#2ecc71", alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, neg_means, width, label="Non-AIP (neg)",
           color="#e74c3c", alpha=0.85, edgecolor="white")

    ax.set_title(prop, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(transition_labels, fontsize=8)
    ax.set_ylabel("Mean Transition Rate", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

axes[-1].set_visible(False)

plt.suptitle("CTDT: Mean Transition Rate per Property (AIP vs Non-AIP)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "CTDT_grouped_bar_chart.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Grouped bar chart saved")


# ============================================================
#   CELL 10 — Visualization 2: Heatmap of All 21 CTDT Features
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

heatmap_data = stats[["Negative (0)", "Positive (1)"]].T
short_labels = [c.replace("CTDT_", "")
                 .replace("_G1G2", "\nG1↔G2")
                 .replace("_G1G3", "\nG1↔G3")
                 .replace("_G2G3", "\nG2↔G3")
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
    cbar_kws={"label": "Mean Transition Rate"}
)
ax.set_title("CTDT Feature Heatmap — Mean Values (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("CTDT Features", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "CTDT_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Heatmap saved")


# ============================================================
#   CELL 11 — Visualization 3: Discriminative Feature Bar Plot
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5))

diff   = stats["Difference"]
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in diff]
short  = [c.replace("CTDT_", "")
           .replace("_G1G2", "\nG1↔G2")
           .replace("_G1G3", "\nG1↔G3")
           .replace("_G2G3", "\nG2↔G3")
          for c in diff.index]

ax.bar(range(len(diff)), diff.values, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(diff)))
ax.set_xticklabels(short, fontsize=7)
ax.set_xlabel("CTDT Feature", fontsize=12)
ax.set_ylabel("Mean Difference (Pos − Neg)", fontsize=12)
ax.set_title("Discriminative Power of CTDT Features",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="Higher in AIP"),
                   Patch(facecolor="#e74c3c", label="Higher in non-AIP")]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "CTDT_discriminative_features.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Discriminative feature plot saved")


# ============================================================
#   CELL 12 — Visualization 4: CTDC vs CTDT Comparison
#             Load CTDC if available and compare feature types
# ============================================================

ctdc_path = os.path.join(OUTPUT_DIR, "CTDC_features.csv")

if os.path.exists(ctdc_path):
    df_ctdc   = pd.read_csv(ctdc_path)
    ctdc_cols = [c for c in df_ctdc.columns if c.startswith("CTDC_")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # CTDC spread
    ctdc_std = df_ctdc[ctdc_cols].std()
    axes[0].bar(range(len(ctdc_std)), ctdc_std.values, color="#3498db", alpha=0.8)
    axes[0].set_title("CTDC — Feature Std Dev (variability)", fontweight="bold")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Std Deviation")
    axes[0].grid(axis="y", alpha=0.3)

    # CTDT spread
    ctdt_std = df_ctdt[ctdt_cols].std()
    axes[1].bar(range(len(ctdt_std)), ctdt_std.values, color="#e67e22", alpha=0.8)
    axes[1].set_title("CTDT — Feature Std Dev (variability)", fontweight="bold")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Std Deviation")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Feature Variability: CTDC vs CTDT",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "CTDC_vs_CTDT_variability.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ CTDC vs CTDT comparison saved")
else:
    print("ℹ️  CTDC file not found — skipping comparison plot")
    print(f"   Run 03_CTDC_feature_extraction.py first to enable this plot")


# ============================================================
#   CELL 13 — Save Final CSV
# ============================================================

df_ctdt.to_csv(OUTPUT_CSV, index=False)

print("=" * 55)
print("✅ CTDT Feature Extraction COMPLETE")
print(f"   File     : {OUTPUT_CSV}")
print(f"   Shape    : {df_ctdt.shape}")
print(f"   Features : 7 properties × 3 transitions = 21 CTDT features")
print(f"   Figures  : {FIGURES_DIR}")
print("=" * 55)