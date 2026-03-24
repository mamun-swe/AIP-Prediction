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

OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "PAAC_features.csv")

# Output will be saved here (ONLY for Google Colab, ignored in local runs)
# FIGURES_DIR = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/results/figures"

# Output will be saved here (ONLY for local runs, ignored in Google Colab)
FIGURES_DIR = "../../results/figures"

# PAAC hyperparameter: sequence-order lag (1 ≤ λ ≤ L-1)
# Recommended: 3–5 for short peptides (avg length ~16 aa)
LAMBDA      = 3

# Weight factor: controls balance between AAC and order info
# Recommended: 0.05–0.1
WEIGHT      = 0.05

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"✅ Features dir : {OUTPUT_DIR}")
print(f"✅ Figures dir  : {FIGURES_DIR}")
print(f"✅ Lambda (λ)   : {LAMBDA}")
print(f"✅ Weight (w)   : {WEIGHT}")
print(f"✅ Total features per sequence: 20 + {LAMBDA} = {20 + LAMBDA}")


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
#   CELL 5 — Define Physicochemical Properties for PAAC
# ============================================================
#
#  PAAC uses 3 standard physicochemical properties (Chou 2001)
#  to compute sequence-order correlation:
#    1. Hydrophobicity  (Tanford scale, normalized)
#    2. Hydrophilicity  (Hopp-Woods scale, normalized)
#    3. Side-chain mass (Radzicka-Wolfenden, normalized)
#
# ============================================================

RAW_PROPERTIES = {
    # aa : [Hydrophobicity, Hydrophilicity, SideChainMass]
    "A": [ 0.62, -0.5,  15.0],
    "C": [ 0.29, -1.0,  47.0],
    "D": [-0.90,  3.0,  59.0],
    "E": [-0.74,  3.0,  73.0],
    "F": [ 1.19, -2.5,  91.0],
    "G": [ 0.48, -0.5,   1.0],
    "H": [-0.40, -0.5,  82.0],
    "I": [ 1.38, -1.8,  57.0],
    "K": [-1.50,  3.0,  73.0],
    "L": [ 1.06, -1.8,  57.0],
    "M": [ 0.64, -1.3,  75.0],
    "N": [-0.78,  2.0,  58.0],
    "P": [ 0.12,  0.0,  42.0],
    "Q": [-0.85,  0.2,  72.0],
    "R": [-2.53,  3.0, 101.0],
    "S": [-0.18,  0.3,  31.0],
    "T": [-0.05, -0.4,  45.0],
    "V": [ 1.08, -1.5,  43.0],
    "W": [ 0.81, -3.4, 130.0],
    "Y": [ 0.26, -2.3, 107.0],
}

# ✅ Fixed: always define exactly 20 AAs in consistent order
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
N_PROPS     = 3


def normalize_properties(raw_props):
    """Z-score normalize each property across all 20 AAs."""
    prop_matrix = np.array([raw_props[aa] for aa in AMINO_ACIDS])
    means       = prop_matrix.mean(axis=0)
    stds        = prop_matrix.std(axis=0)
    norm_matrix = (prop_matrix - means) / stds
    return {aa: norm_matrix[i] for i, aa in enumerate(AMINO_ACIDS)}

NORM_PROPS = normalize_properties(RAW_PROPERTIES)

print("── Normalized Properties (first 5 AAs) ────────────")
print(f"  {'AA':4s} {'Hydrophob':>12s} {'Hydrophil':>12s} {'SideChain':>12s}")
for aa in AMINO_ACIDS[:5]:
    h, r, m = NORM_PROPS[aa]
    print(f"  {aa:4s} {h:12.4f} {r:12.4f} {m:12.4f}")
print("  ...")


# ============================================================
#   CELL 6 — PAAC Feature Extraction
# ============================================================

def sequence_order_correlation(sequence, lag, norm_props):
    """
    Compute Sequence-Order Correlation Factor θ_λ:

    θ_λ = (1/(L-λ)) × Σ_{i=1}^{L-λ} Θ(R_i, R_{i+λ})

    Θ(R_i, R_j) = (1/N_PROPS) × Σ_k [H_k(R_i) - H_k(R_j)]²
    """
    L = len(sequence)
    if L <= lag:
        return 0.0

    theta = 0.0
    count = 0
    for i in range(L - lag):
        aa_i = sequence[i]
        aa_j = sequence[i + lag]
        if aa_i not in norm_props or aa_j not in norm_props:
            continue
        props_i = norm_props[aa_i]
        props_j = norm_props[aa_j]
        theta  += np.mean((props_i - props_j) ** 2)
        count  += 1

    return round(theta / count, 6) if count > 0 else 0.0


def compute_paac(sequence, lam=LAMBDA, weight=WEIGHT,
                 norm_props=NORM_PROPS):
    """
    PAAC — Pseudo Amino Acid Composition (Type 1, Chou 2001):

    Feature vector of length (20 + λ):
      p_i      = f_i / [Σf_k + w×Σθ_j]     i = 1..20
      p_{20+j} = w×θ_j / [Σf_k + w×Σθ_j]   j = 1..λ

    ✅ Fix 1: Iterate over AMINO_ACIDS (not sequence chars)
              → guarantees all 20 keys always exist
    ✅ Fix 2: Use .get(aa, 0) so missing AAs default to 0.0
    """
    L      = len(sequence)
    counts = Counter(sequence)

    # ✅ All 20 AAs always present via explicit iteration
    f = {aa: counts.get(aa, 0) / L for aa in AMINO_ACIDS}

    # Sequence-order correlation factors for lag 1..λ
    thetas = [sequence_order_correlation(sequence, j, norm_props)
              for j in range(1, lam + 1)]

    denominator = sum(f.values()) + weight * sum(thetas)
    if denominator == 0:
        denominator = 1.0  # safety guard

    features = {}

    # ✅ AAC part — always 20 keys in fixed order
    for aa in AMINO_ACIDS:
        features[f"PAAC_{aa}"] = round(f[aa] / denominator, 6)

    # Sequence-order part — λ keys
    for j, theta in enumerate(thetas, start=1):
        features[f"PAAC_T{j}"] = round((weight * theta) / denominator, 6)

    return features


def extract_paac(df, lam=LAMBDA, weight=WEIGHT):
    """Apply PAAC extraction to full dataframe."""
    print(f"Extracting PAAC features... ({20 + lam} per sequence, λ={lam}, w={weight})")

    paac_records = df["sequence"].apply(
        lambda seq: compute_paac(seq, lam=lam, weight=weight)
    )
    paac_df = pd.DataFrame(list(paac_records))

    # ✅ Fix 3: Reindex to enforce column order and fill any gaps
    aa_cols = [f"PAAC_{aa}" for aa in AMINO_ACIDS]
    t_cols  = [f"PAAC_T{j}" for j in range(1, lam + 1)]
    paac_df = paac_df.reindex(columns=aa_cols + t_cols, fill_value=0.0)

    result  = pd.concat([df.reset_index(drop=True), paac_df], axis=1)
    # ── REMOVED ──────────────────────────────────────────────
    # result.insert(2, "length", result["sequence"].apply(len))
    # ─────────────────────────────────────────────────────────────────
    return result

df_paac = extract_paac(df_seq)

# ── CHANGED (Block): drop sequence & length, move label to end ──
df_paac = df_paac.drop(columns=["sequence"])                        # ← remove sequence column
cols   = [c for c in df_paac.columns if c != "label"] + ["label"] # ← move label to end
df_paac = df_paac[cols]                                             # ← reorder columns
# ────────────────────────────────────────────────────────────────

# ✅ Fix 4: Column lists always derived from actual dataframe
paac_cols    = [c for c in df_paac.columns if c.startswith("PAAC_")]
paac_aa_cols = [c for c in paac_cols if not c.startswith("PAAC_T")]
paac_t_cols  = [c for c in paac_cols if c.startswith("PAAC_T")]

print(f"\n✅ PAAC extraction complete")
print(f"   Shape          : {df_paac.shape}")
print(f"   AAC features   : {len(paac_aa_cols)}  (expected 20)")
print(f"   Order features : {len(paac_t_cols)}  (expected {LAMBDA})")
df_paac.head(3)


# ============================================================
#   CELL 7 — Validate Output
# ============================================================

print("── Validation ──────────────────────────────────────")
print(f"Total PAAC features  : {len(paac_cols)}")
print(f"AAC features         : {len(paac_aa_cols)} {'✅' if len(paac_aa_cols)==20 else '⚠️'}")
print(f"Order (θ) features   : {len(paac_t_cols)} {'✅' if len(paac_t_cols)==LAMBDA else '⚠️'}")
print(f"Missing values       : {df_paac[paac_cols].isnull().sum().sum()}")
print(f"Value range          : {df_paac[paac_cols].values.min():.6f} – "
      f"{df_paac[paac_cols].values.max():.6f}")

neg_vals = (df_paac[paac_cols] < 0).sum().sum()
print(f"Negative values      : {neg_vals} {'✅' if neg_vals == 0 else '⚠️'}")

row_sums = df_paac[paac_cols].sum(axis=1).round(4)
all_one  = row_sums.between(0.999, 1.001).all()
print(f"All rows sum ≈ 1.0   : {'✅ Yes' if all_one else '⚠️  Check'}")
print(f"  Row sum range      : {row_sums.min()} – {row_sums.max()}")


# ============================================================
#   CELL 8 — Statistical Summary: Pos vs Neg
# ============================================================

stats = df_paac.groupby("label")[paac_cols].mean().T
stats.columns = ["Negative (0)", "Positive (1)"]
stats["Difference"] = (stats["Positive (1)"] - stats["Negative (0)"]).round(6)
stats_sorted = stats.sort_values("Difference", ascending=False)

print("── Top 10 PAAC features higher in AIP ─────────────")
print(stats_sorted.head(10).to_string())
print("\n── Top 10 PAAC features higher in non-AIP ─────────")
print(stats_sorted.tail(10).to_string())


# ============================================================
#   CELL 9 — Visualization 1: PAAC Component Comparison
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
width = 0.35

# --- Left: AAC part of PAAC ---
ax    = axes[0]
# ✅ Use actual column count — never hardcode 20
n_aa  = len(paac_aa_cols)
x     = np.arange(n_aa)

pos_means = df_paac[df_paac["label"] == 1][paac_aa_cols].mean().values
neg_means = df_paac[df_paac["label"] == 0][paac_aa_cols].mean().values

# ✅ x.shape == pos_means.shape guaranteed by reindex in extract_paac
ax.bar(x - width/2, pos_means, width, label="AIP (pos)",
       color="#2ecc71", alpha=0.85, edgecolor="white")
ax.bar(x + width/2, neg_means, width, label="Non-AIP (neg)",
       color="#e74c3c", alpha=0.85, edgecolor="white")
ax.set_xticks(x)
# ✅ Labels from actual column names, not hardcoded alphabet
ax.set_xticklabels([c.replace("PAAC_", "") for c in paac_aa_cols], fontsize=9)
ax.set_title(f"PAAC — Amino Acid Component ({n_aa} features)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Mean PAAC Value")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# --- Right: Sequence-order part ---
ax    = axes[1]
n_t   = len(paac_t_cols)
x     = np.arange(n_t)

pos_means = df_paac[df_paac["label"] == 1][paac_t_cols].mean().values
neg_means = df_paac[df_paac["label"] == 0][paac_t_cols].mean().values

ax.bar(x - width/2, pos_means, width, label="AIP (pos)",
       color="#3498db", alpha=0.85, edgecolor="white")
ax.bar(x + width/2, neg_means, width, label="Non-AIP (neg)",
       color="#e67e22", alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"θ{j+1}\n(lag {j+1})" for j in range(n_t)], fontsize=10)
ax.set_title(f"PAAC — Sequence-Order Component ({n_t} features, λ={LAMBDA})",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Mean PAAC Value")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("PAAC Feature Breakdown: AAC Component vs Sequence-Order Component",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "PAAC_component_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Component comparison saved")


# ============================================================
#   CELL 10 — Visualization 2: Theta (θ) Values Across Lambda
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

max_lambda      = 10
sample_pos      = df_seq[df_seq["label"] == 1]["sequence"].tolist()   # ← FIXED: use df_seq (sequence dropped from df_paac)
sample_neg      = df_seq[df_seq["label"] == 0]["sequence"].tolist()   # ← FIXED: use df_seq (sequence dropped from df_paac)
theta_pos_means = []
theta_neg_means = []

for lag in range(1, max_lambda + 1):
    pos_thetas = [sequence_order_correlation(s, lag, NORM_PROPS)
                  for s in sample_pos if len(s) > lag]
    neg_thetas = [sequence_order_correlation(s, lag, NORM_PROPS)
                  for s in sample_neg if len(s) > lag]
    theta_pos_means.append(np.mean(pos_thetas) if pos_thetas else 0)
    theta_neg_means.append(np.mean(neg_thetas) if neg_thetas else 0)

ax.plot(range(1, max_lambda + 1), theta_pos_means, "o-",
        color="#2ecc71", linewidth=2, markersize=7, label="AIP (pos)")
ax.plot(range(1, max_lambda + 1), theta_neg_means, "s-",
        color="#e74c3c", linewidth=2, markersize=7, label="Non-AIP (neg)")
ax.axvline(LAMBDA, color="grey", linestyle="--", linewidth=1.2,
           label=f"Current λ = {LAMBDA}")

ax.set_xlabel("Lag (λ)", fontsize=12)
ax.set_ylabel("Mean θ_λ", fontsize=12)
ax.set_title("Sequence-Order Correlation Factor θ_λ vs Lag Distance",
             fontsize=13, fontweight="bold")
ax.set_xticks(range(1, max_lambda + 1))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "PAAC_theta_vs_lambda.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Theta vs Lambda plot saved")


# ============================================================
#   CELL 11 — Visualization 3: Discriminative Feature Bar Plot
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5))

diff   = stats_sorted["Difference"]
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in diff]
short  = [c.replace("PAAC_", "") for c in diff.index]

ax.bar(range(len(diff)), diff.values, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(diff)))
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("PAAC Feature", fontsize=12)
ax.set_ylabel("Mean Difference (Pos − Neg)", fontsize=12)
ax.set_title("Discriminative Power of All PAAC Features",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="Higher in AIP"),
                   Patch(facecolor="#e74c3c", label="Higher in non-AIP")]
ax.legend(handles=legend_elements, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "PAAC_discriminative_features.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Discriminative feature plot saved")


# ============================================================
#   CELL 12 — Visualization 4: Heatmap of All PAAC Features
# ============================================================

fig, ax = plt.subplots(figsize=(16, 6))

heatmap_data = stats[["Negative (0)", "Positive (1)"]].T
short_labels = [c.replace("PAAC_", "") for c in stats.index]

sns.heatmap(
    heatmap_data,
    xticklabels=short_labels,
    yticklabels=["Non-AIP (neg)", "AIP (pos)"],
    cmap="YlOrRd",
    annot=True,
    fmt=".4f",
    linewidths=0.4,
    linecolor="grey",
    ax=ax,
    cbar_kws={"label": "Mean PAAC Value"},
    annot_kws={"size": 7}
)
ax.set_title(f"PAAC Feature Heatmap — Mean Values (λ={LAMBDA}, w={WEIGHT})",
             fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "PAAC_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Heatmap saved")


# ============================================================
#   CELL 13 — Lambda Sensitivity Analysis
# ============================================================

print("── Lambda Sensitivity Analysis ─────────────────────")
print(f"{'λ':>5s}  {'Features':>10s}  {'Mean θ (pos)':>14s}  {'Mean θ (neg)':>14s}")
print("-" * 50)

for lam in range(1, 8):
    valid_seqs = df_seq[df_seq["sequence"].apply(len) > lam]
    pos_theta  = np.mean([
        sequence_order_correlation(s, lam, NORM_PROPS)
        for s in valid_seqs[valid_seqs["label"] == 1]["sequence"]
    ])
    neg_theta  = np.mean([
        sequence_order_correlation(s, lam, NORM_PROPS)
        for s in valid_seqs[valid_seqs["label"] == 0]["sequence"]
    ])
    marker = " ◄ current" if lam == LAMBDA else ""
    print(f"{lam:>5d}  {20+lam:>10d}  {pos_theta:>14.6f}  "
          f"{neg_theta:>14.6f}{marker}")


# ============================================================
#   CELL 14 — Save Final CSV
# ============================================================

df_paac.to_csv(OUTPUT_CSV, index=False)

print("=" * 55)
print("✅ PAAC Feature Extraction COMPLETE")
print(f"   File     : {OUTPUT_CSV}")
print(f"   Shape    : {df_paac.shape}")
print(f"   Features : 20 (AAC) + {LAMBDA} (θ lag) = {20 + LAMBDA} total")
print(f"   λ used   : {LAMBDA}   |   weight : {WEIGHT}")
print(f"   Figures  : {FIGURES_DIR}")
print("=" * 55)