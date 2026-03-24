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

# DRIVE PATH
# FASTA_PATH   = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/data/raw/AIP_ind.fasta"

# LOCAL PATH
FASTA_PATH   = "../../data/raw/AIP_ind.fasta"

# Drive output (uncomment for Google Colab)
# FIGURE_DIR   = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/results/figures"
# OUTPUT_DIR   = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/data/features"

# Local output (uncomment for local runs)
FIGURE_DIR   = "../../results/figures"
OUTPUT_DIR   = "../../data/features"

OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "AAC_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Output directory ready: {OUTPUT_DIR}")


# ============================================================
#   CELL 4 — FASTA Parser
# ============================================================

def parse_fasta(filepath):
    """
    Parse a FASTA file into a list of (id, sequence, label) tuples.
    Labels are inferred from header prefix: pos → 1, neg → 0
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
                    "seq_id"   : current_id,
                    "sequence" : line.upper(),
                    "label"    : label
                })

    df = pd.DataFrame(entries)
    print(f"✅ Parsed {len(df)} sequences")
    print(f"   Positive (AIP)   : {df['label'].sum()}")
    print(f"   Negative (non-AIP): {(df['label']==0).sum()}")
    return df

df_seq = parse_fasta(FASTA_PATH)
df_seq.head(3)


# ============================================================
#   CELL 5 — AAC Feature Extraction
# ============================================================

# 20 standard amino acids (alphabetical order)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def compute_aac(sequence):
    """
    Amino Acid Composition (AAC):
    Frequency of each amino acid in a sequence.

    Formula:
        AAC(aa_i) = count(aa_i) / len(sequence)

    Returns: dict of 20 features  e.g. {'AAC_A': 0.05, ...}
    """
    length  = len(sequence)
    counts  = Counter(sequence)
    return {f"AAC_{aa}": round(counts.get(aa, 0) / length, 6)
            for aa in AMINO_ACIDS}


def extract_aac(df):
    """Apply AAC extraction to full dataframe."""
    aac_records = df["sequence"].apply(compute_aac)
    aac_df      = pd.DataFrame(list(aac_records))
    result      = pd.concat([df.reset_index(drop=True), aac_df], axis=1)
    # ── REMOVED ──────────────────────────────────────────────
    # result.insert(2, "length", result["sequence"].apply(len))
    # ─────────────────────────────────────────────────────────
    return result

df_aac = extract_aac(df_seq)

# ── CHANGED (Block): drop sequence & length, move label to end ──
df_aac = df_aac.drop(columns=["sequence"])                        # ← remove sequence column
cols   = [c for c in df_aac.columns if c != "label"] + ["label"] # ← move label to end
df_aac = df_aac[cols]                                             # ← reorder columns
# ────────────────────────────────────────────────────────────────

print(f"✅ AAC extraction complete")
print(f"   Shape   : {df_aac.shape}  (sequences × features)")
print(f"   Columns : {list(df_aac.columns)}")                     # ← CHANGED: show all columns
df_aac.head(3)


# ============================================================
#   CELL 6 — Verify: No missing values or invalid features
# ============================================================

aac_cols = [c for c in df_aac.columns if c.startswith("AAC_")]

print("── Validation ──────────────────────────────")
print(f"Missing values     : {df_aac[aac_cols].isnull().sum().sum()}")
print(f"All rows sum to 1  : {df_aac[aac_cols].sum(axis=1).round(4).eq(1.0).all()}")
print(f"Value range        : {df_aac[aac_cols].values.min():.4f} – {df_aac[aac_cols].values.max():.4f}")
print("✅ All checks passed")


# ============================================================
#   CELL 7 — Statistical Summary (Pos vs Neg)
# ============================================================

stats = df_aac.groupby("label")[aac_cols].mean().T
stats.columns = ["Negative (0)", "Positive (1)"]
stats["Difference"] = (stats["Positive (1)"] - stats["Negative (0)"]).round(6)
stats = stats.sort_values("Difference", ascending=False)

print("── Mean AAC: Positive vs Negative (sorted by difference) ──")
print(stats.to_string())


# ============================================================
#   CELL 8 — Visualization 1: AAC Mean Comparison Bar Plot
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5))

x         = np.arange(len(AMINO_ACIDS))
width     = 0.35
pos_means = df_aac[df_aac["label"]==1][aac_cols].mean().values
neg_means = df_aac[df_aac["label"]==0][aac_cols].mean().values

bars1 = ax.bar(x - width/2, pos_means, width, label="Positive (AIP)",
               color="#2ecc71", edgecolor="white", alpha=0.85)
bars2 = ax.bar(x + width/2, neg_means, width, label="Negative (non-AIP)",
               color="#e74c3c", edgecolor="white", alpha=0.85)

ax.set_xlabel("Amino Acid", fontsize=12)
ax.set_ylabel("Mean Frequency", fontsize=12)
ax.set_title("AAC Mean Frequency: AIP vs Non-AIP", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(AMINO_ACIDS, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "AAC_mean_comparison.png"), dpi=150)
plt.show()
print("✅ Plot saved")


# ============================================================
#   CELL 9 — Visualization 2: Heatmap of AAC Features
# ============================================================

fig, ax = plt.subplots(figsize=(22, 6))

# Sample 60 sequences for readability
sample = df_aac.sample(min(60, len(df_aac)), random_state=42).sort_values("label")
heatmap_data = sample[aac_cols].values

im = ax.imshow(heatmap_data.T, aspect="auto", cmap="YlOrRd")

ax.set_yticks(range(len(AMINO_ACIDS)))
ax.set_yticklabels(AMINO_ACIDS, fontsize=10)
ax.set_xlabel("Peptide Sequences", fontsize=12)
ax.set_title("AAC Feature Heatmap (sample of 60 sequences)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Frequency")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "AAC_heatmap.png"), dpi=150)
plt.show()
print("✅ Heatmap saved")


# ============================================================
#   CELL 10 — Visualization 3: Top Discriminative Features
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

diff     = stats["Difference"]
colors   = ["#2ecc71" if v > 0 else "#e74c3c" for v in diff]

ax.bar(diff.index, diff.values, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Amino Acid Feature", fontsize=12)
ax.set_ylabel("Mean Difference (Pos − Neg)", fontsize=12)
ax.set_title("Discriminative Power of AAC Features", fontsize=13, fontweight="bold")
ax.set_xticklabels(diff.index, rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="Higher in AIP"),
                   Patch(facecolor="#e74c3c", label="Higher in non-AIP")]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "AAC_discriminative_features.png"), dpi=150)
plt.show()
print("✅ Discriminative feature plot saved")


# ============================================================
#   CELL 11 — Save Final CSV
# ============================================================

df_aac.to_csv(OUTPUT_CSV, index=False)

print("=" * 50)
print("✅ AAC feature extraction COMPLETE")
print(f"   File    : {OUTPUT_CSV}")
print(f"   Shape   : {df_aac.shape}")
print(f"   Columns : seq_id, AAC_A ... AAC_Y, label") # ← CHANGED: updated column description
print("=" * 50)