# ============================================================
#   CELL 1 — Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')


# ============================================================
#   CELL 2 — Install Required Libraries
# ============================================================
# ESMC requires esm >= 3.x (different from ESM-2's fair-esm)
# Uninstall fair-esm first if you ran the ESM-2 script in this
# same runtime — they conflict with each other.
!pip uninstall fair-esm -y -q
!pip install esm umap-learn -q

print("✅ Libraries installed")


# ============================================================
#   CELL 3 — HuggingFace Login (Required for ESMC weights)
# ============================================================
#
#  ESMC model weights are gated on HuggingFace.
#  You MUST accept the license and log in before downloading.
#
#  Steps (one-time):
#  1. Visit https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12
#  2. Click "Agree and access repository" to accept the license
#  3. Go to https://huggingface.co/settings/tokens
#  4. Create a token with "read" permissions
#  5. Paste it below
#
# ============================================================
from huggingface_hub import login

HF_TOKEN = "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"   # ← paste your token here

login(token=HF_TOKEN, add_to_git_credential=False)
print("✅ HuggingFace login successful")


# ============================================================
#   CELL 4 — Import Libraries
# ============================================================
import os
import gc
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Check GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Libraries loaded")
print(f"   Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ⚠️  No GPU — running on CPU (slower)")


# ============================================================
#   CELL 5 — Configuration (Edit paths here)
# ============================================================

# FOR DRIVE FILE LOCATION
FASTA_PATH  = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/data/raw/AIP_ind.fasta"

# FOR LOCAL
# FASTA_PATH  = "../../data/raw/AIP_ind.fasta"

# FOR DRIVE FILE LOCATION
OUTPUT_DIR  = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/data/features"

# FOR LOCAL
# OUTPUT_DIR  = "../../data/features"

OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "ESMC_features.csv")
OUTPUT_NPY  = os.path.join(OUTPUT_DIR, "ESMC_embeddings.npy")

# FOR DRIVE FILE LOCATION
FIGURES_DIR = "/content/drive/MyDrive/CANADA/Thesis/API-Prediction/results/figures"

# FOR LOCAL
# OUTPUT_DIR  = "../../results/figures"

# ── ESMC Model Variants ──────────────────────────────────────
#
#  Model           Params  Embed-dim  VRAM       Notes
#  esmc_300m       300M    960        ~2 GB      ✅ recommended (Colab free)
#  esmc_600m       600M    1152       ~4 GB      better quality
#  esmc-6b-2024-12  6B     ~4096      ~24 GB     Forge API only (not local)
#
#  ESMC vs ESM-2 efficiency:
#  esmc_300m  ≈ ESM-2 650M quality  (2x smaller, faster)
#  esmc_600m  ≈ ESM-2 3B   quality  (5x smaller, faster)
#
#  ✅ Recommended for Colab T4 (16 GB): esmc_600m
#  ✅ Recommended for Colab free (15 GB RAM): esmc_300m
#  🔁 If OOM: switch to esmc_300m
# ────────────────────────────────────────────────────────────
MODEL_NAME  = "esmc_600m"    # or "esmc_300m"

# Embedding dimensions per model:
#   esmc_300m → 960
#   esmc_600m → 1152
EMBED_DIM_MAP = {
    "esmc_300m": 960,
    "esmc_600m": 1152,
}

# Pooling strategy
#   "mean" — average over all residue tokens (recommended ✅)
#   "cls"  — first token (BOS)
#   "max"  — max over all residue tokens
POOLING = "mean"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"✅ Features dir  : {OUTPUT_DIR}")
print(f"✅ Figures dir   : {FIGURES_DIR}")
print(f"✅ Model         : {MODEL_NAME}")
print(f"✅ Embedding dim : {EMBED_DIM_MAP.get(MODEL_NAME, 'see model card')}")
print(f"✅ Pooling       : {POOLING}")


# ============================================================
#   CELL 6 — FASTA Parser
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
    print(f"   Length range      : {df['sequence'].apply(len).min()} – "
          f"{df['sequence'].apply(len).max()} aa")
    return df

df_seq = parse_fasta(FASTA_PATH)
df_seq.head(3)


# ============================================================
#   CELL 7 — Load ESMC Model
# ============================================================
#
#  ESMC (ESM Cambrian) — released December 2024 by EvolutionaryScale.
#
#  Key differences from ESM-2:
#  ✅ Completely new API: ESMC, ESMProtein, LogitsConfig
#  ✅ No batch_converter — each sequence is wrapped in ESMProtein()
#  ✅ Embeddings returned via logits_output.embeddings
#     (NOT results["representations"][layer] like ESM-2)
#  ✅ Embeddings are ALREADY mean-pooled per sequence by default
#     shape: (sequence_length+2, embed_dim) — includes BOS/EOS tokens
#  ✅ Architecture: Pre-LN, rotary embeddings, SwiGLU activations
#  ✅ No biases in linear layers or layer norms
#
#  API pattern:
#    protein        = ESMProtein(sequence="ACDE")
#    protein_tensor = client.encode(protein)
#    output         = client.logits(protein_tensor,
#                         LogitsConfig(sequence=True,
#                                      return_embeddings=True))
#    embeddings     = output.embeddings  # (L+2, embed_dim)
#
# ============================================================

print(f"Loading ESMC model: {MODEL_NAME}")
print("(Downloads ~1–2 GB on first run — cached afterwards)\n")

# Load model — requires HuggingFace login (Cell 3)
esmc_client = ESMC.from_pretrained(MODEL_NAME).to(DEVICE)
esmc_client.eval()

print(f"✅ ESMC model loaded on {DEVICE}")
print(f"   Model        : {MODEL_NAME}")
print(f"   Embed dim    : {EMBED_DIM_MAP.get(MODEL_NAME, 'check model card')}")


# ============================================================
#   CELL 8 — Sequence Preprocessing for ESMC
# ============================================================
#
#  ✅ ESMC takes RAW sequences — NO space formatting needed
#     (same as ESM-2, unlike ProtBERT/ProtT5)
#
#  Replace non-standard amino acids:
#     U (Selenocysteine) → C
#     B, Z, O            → X
#
# ============================================================

def preprocess_sequence(sequence):
    """
    Minimal preprocessing for ESMC.
    Raw sequences — no space formatting needed.
    """
    sequence = sequence.upper()
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return sequence


sample    = df_seq["sequence"].iloc[0]
processed = preprocess_sequence(sample)
print(f"Original  : {sample}")
print(f"Processed : {processed}")
print(f"\n✅ No space formatting needed for ESMC")


# ============================================================
#   CELL 9 — ESMC Embedding Extraction
# ============================================================
#
#  ESMC embedding shape after client.logits():
#
#    output.embeddings : tensor (seq_len + 2, embed_dim)
#      position 0          = BOS token  (<cls>-equivalent)
#      position 1..L       = amino acid tokens
#      position L+1        = EOS token
#
#  We pool over residue positions [1..L] only (skip BOS/EOS).
#
# ============================================================

def pool_esmc_embeddings(embeddings_tensor, seq_len, strategy="mean"):
    """
    Pool ESMC per-residue embeddings → one vector per sequence.

    Args:
        embeddings_tensor : (seq_len+2, embed_dim)  — includes BOS/EOS
        seq_len           : actual amino acid count
        strategy          : "mean", "cls", or "max"

    Returns:
        pooled : 1D tensor (embed_dim,)
    """
    # Residue tokens are at positions 1..seq_len (skip BOS at 0, EOS at end)
    residue_emb = embeddings_tensor[1: seq_len + 1]  # (L, embed_dim)

    if strategy == "mean":
        return residue_emb.mean(dim=0)
    elif strategy == "cls":
        # BOS token at position 0 (analogous to [CLS])
        return embeddings_tensor[0]
    elif strategy == "max":
        return residue_emb.max(dim=0).values
    else:
        raise ValueError(f"Unknown pooling: '{strategy}'. "
                         f"Choose 'mean', 'cls', or 'max'.")


def extract_esmc_embeddings(df, client, device,
                             pooling=POOLING):
    """
    Extract ESMC embeddings for all sequences one-by-one.

    ⚠️  ESMC API processes ONE sequence at a time via ESMProtein().
        There is no batch tokenizer like ESM-2's batch_converter.
        For 245 short peptides this is fast enough (~1-2 min on GPU).

    Returns:
        embeddings : np.ndarray shape (N, embed_dim)
    """
    sequences   = df["sequence"].tolist()
    n_sequences = len(sequences)
    all_vectors = []

    print(f"Extracting ESMC embeddings...")
    print(f"  Sequences  : {n_sequences}")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Pooling    : {pooling}")
    print(f"  Device     : {device}\n")

    with torch.no_grad():
        for idx, raw_seq in enumerate(sequences):
            seq = preprocess_sequence(raw_seq)

            # Wrap in ESMProtein container
            protein = ESMProtein(sequence=seq)

            # Encode sequence → tokenised protein tensor
            protein_tensor = client.encode(protein)

            # Forward pass — request embeddings
            # LogitsConfig(return_embeddings=True) returns per-token
            # embeddings of shape (seq_len+2, embed_dim)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )

            # logits_output.embeddings: (seq_len+2, embed_dim)
            emb_tensor = logits_output.embeddings  # on CPU or GPU

            # Move to CPU for pooling
            emb_cpu = emb_tensor.squeeze(0).cpu().float()

            # Pool → (embed_dim,)
            pooled = pool_esmc_embeddings(
                emb_cpu, len(seq), strategy=pooling
            )

            all_vectors.append(pooled.numpy())

            # Progress every 25 sequences
            if (idx + 1) % 25 == 0 or (idx + 1) == n_sequences:
                print(f"  [{idx + 1:>4}/{n_sequences}] done")

            # Periodic GPU cache flush
            if device.type == "cuda" and (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    embeddings = np.stack(all_vectors, axis=0)
    print(f"\n✅ Extraction complete")
    print(f"   Shape : {embeddings.shape}")
    return embeddings


embeddings = extract_esmc_embeddings(df_seq, esmc_client, DEVICE)


# ============================================================
#   CELL 10 — Build Feature DataFrame
# ============================================================

embedding_dim = embeddings.shape[1]
col_names     = [f"ESMC_{i}" for i in range(embedding_dim)]

df_emb    = pd.DataFrame(embeddings, columns=col_names)
df_esmc   = pd.concat([
    df_seq[["seq_id", "sequence", "label"]].reset_index(drop=True),
    df_emb
], axis=1)
# ── REMOVED ──────────────────────────────────────────────
# df_esmc.insert(2, "length", df_esmc["sequence"].apply(len))
# ─────────────────────────────────────────────────────────────────

# ── CHANGED (Block): drop sequence & length, move label to end ──
df_esmc = df_esmc.drop(columns=["sequence"])                        # ← remove sequence column
cols   = [c for c in df_esmc.columns if c != "label"] + ["label"] # ← move label to end
df_esmc = df_esmc[cols]                                             # ← reorder columns
# ────────────────────────────────────────────────────────────────

emb_cols = [c for c in df_esmc.columns if c.startswith("ESMC_")]

print(f"✅ Feature DataFrame built")
print(f"   Shape   : {df_esmc.shape}")
print(f"      Columns : seq_id, ESMC_0...ESMC_N (embedding features), label")  # ← CHANGED: updated column description
df_esmc.iloc[:3, :8]


# ============================================================
#   CELL 11 — Validate Output
# ============================================================

print("── Validation ──────────────────────────────────────")
print(f"Embedding dimension  : {len(emb_cols)}")
print(f"Missing values       : {df_esmc[emb_cols].isnull().sum().sum()}")
print(f"Value range          : {df_esmc[emb_cols].values.min():.4f} – "
      f"{df_esmc[emb_cols].values.max():.4f}")
print(f"Mean (pos class)     : "
      f"{df_esmc[df_esmc['label']==1][emb_cols].values.mean():.6f}")
print(f"Mean (neg class)     : "
      f"{df_esmc[df_esmc['label']==0][emb_cols].values.mean():.6f}")
print(f"Std  (pos class)     : "
      f"{df_esmc[df_esmc['label']==1][emb_cols].values.std():.6f}")
print(f"Std  (neg class)     : "
      f"{df_esmc[df_esmc['label']==0][emb_cols].values.std():.6f}")
print("✅ Validation complete")


# ============================================================
#   CELL 12 — Scale Embeddings for Visualization
# ============================================================

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_esmc[emb_cols].values)
print(f"✅ Scaled — shape: {X_scaled.shape}  mean≈0  std≈1")


# ============================================================
#   CELL 13 — Visualization 1: PCA 2D Plot
# ============================================================

pca_2d  = PCA(n_components=2, random_state=42)
X_pca   = pca_2d.fit_transform(X_scaled)
var_exp = pca_2d.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))
for label, color, name in [(1, "#2ecc71", "AIP (pos)"),
                            (0, "#e74c3c", "Non-AIP (neg)")]:
    mask = df_esmc["label"] == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=color, label=name, alpha=0.75,
               s=65, edgecolors="white", linewidths=0.5)

ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=12)
ax.set_title("PCA of ESMC Embeddings (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ESMC_PCA.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ PCA plot saved")


# ============================================================
#   CELL 14 — Visualization 2: t-SNE 2D Plot
# ============================================================

print("Running t-SNE...")
tsne   = TSNE(n_components=2, random_state=42,
              perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))
for label, color, name in [(1, "#2ecc71", "AIP (pos)"),
                            (0, "#e74c3c", "Non-AIP (neg)")]:
    mask = df_esmc["label"] == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=color, label=name, alpha=0.75,
               s=65, edgecolors="white", linewidths=0.5)

ax.set_xlabel("t-SNE Dim 1", fontsize=12)
ax.set_ylabel("t-SNE Dim 2", fontsize=12)
ax.set_title("t-SNE of ESMC Embeddings (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ESMC_tSNE.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ t-SNE plot saved")


# ============================================================
#   CELL 15 — Visualization 3: UMAP 2D Plot
# ============================================================

print("Running UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42,
                    n_neighbors=15, min_dist=0.1)
X_umap  = reducer.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))
for label, color, name in [(1, "#2ecc71", "AIP (pos)"),
                            (0, "#e74c3c", "Non-AIP (neg)")]:
    mask = df_esmc["label"] == label
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               c=color, label=name, alpha=0.75,
               s=65, edgecolors="white", linewidths=0.5)

ax.set_xlabel("UMAP Dim 1", fontsize=12)
ax.set_ylabel("UMAP Dim 2", fontsize=12)
ax.set_title("UMAP of ESMC Embeddings (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ESMC_UMAP.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ UMAP plot saved")


# ============================================================
#   CELL 16 — Visualization 4: All PLM Models PCA Comparison
#             ESMC + ESM-2 + ProtBERT + ProtT5 + BioBERT
# ============================================================

all_model_files = {
    "ProtBERT" : os.path.join(OUTPUT_DIR, "ProtBERT_embeddings.npy"),
    "ProtT5"   : os.path.join(OUTPUT_DIR, "ProtT5_embeddings.npy"),
    "ESM-2"    : os.path.join(OUTPUT_DIR, "ESM2_embeddings.npy"),
    "BioBERT"  : os.path.join(OUTPUT_DIR, "BioBERT_embeddings.npy"),
    "ESMC"     : None    # already in memory
}

available = {k: v for k, v in all_model_files.items()
             if v is None or os.path.exists(v)}

n_models = len(available)
fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
if n_models == 1:
    axes = [axes]

for ax, (model_label, npy_path) in zip(axes, available.items()):
    emb   = embeddings if npy_path is None else np.load(npy_path)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(emb)
    pca_m = PCA(n_components=2, random_state=42)
    X_m   = pca_m.fit_transform(X_sc)
    var_m = pca_m.explained_variance_ratio_

    for label, color, name in [(1, "#2ecc71", "AIP"),
                                (0, "#e74c3c", "Non-AIP")]:
        mask = df_esmc["label"] == label
        ax.scatter(X_m[mask, 0], X_m[mask, 1],
                   c=color, label=name, alpha=0.7,
                   s=50, edgecolors="white", linewidths=0.4)

    ax.set_xlabel(f"PC1 ({var_m[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_m[1]*100:.1f}%)", fontsize=10)
    ax.set_title(f"PCA — {model_label}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("PCA Comparison: All Protein Language Models",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ESMC_AllPLM_comparison_PCA.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ All-PLM comparison plot saved ({n_models} models shown)")


# ============================================================
#   CELL 17 — Visualization 5: PCA Variance Explained
# ============================================================

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_95     = int(np.argmax(cum_var >= 95)) + 1
n_99     = int(np.argmax(cum_var >= 99)) + 1

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cum_var) + 1), cum_var,
        color="#9b59b6", linewidth=2)
ax.axhline(95, color="#e74c3c", linestyle="--",
           label=f"95% variance → {n_95} components")
ax.axhline(99, color="#e67e22", linestyle="--",
           label=f"99% variance → {n_99} components")
ax.axvline(n_95, color="#e74c3c", linestyle="--", alpha=0.5)
ax.axvline(n_99, color="#e67e22", linestyle="--", alpha=0.5)
ax.fill_between(range(1, len(cum_var) + 1), cum_var,
                alpha=0.1, color="#9b59b6")
ax.set_xlabel("Number of PCA Components", fontsize=12)
ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
ax.set_title("PCA Variance Explained — ESMC Embeddings",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(1, min(200, len(cum_var)))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ESMC_PCA_variance.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Variance plot saved")
print(f"   {n_95} components → 95% variance (vs full {embedding_dim} dims)")
print(f"   {n_99} components → 99% variance")


# ============================================================
#   CELL 18 — Optional: PCA-Compressed Features (95% variance)
# ============================================================

pca_red   = PCA(n_components=n_95, random_state=42)
X_reduced = pca_red.fit_transform(X_scaled)

col_r        = [f"ESMC_PCA_{i}" for i in range(n_95)]
df_reduced   = pd.DataFrame(X_reduced, columns=col_r)
df_pca_final = pd.concat([
    df_seq[["seq_id", "sequence", "label"]].reset_index(drop=True),
    df_reduced
], axis=1)

pca_csv = os.path.join(OUTPUT_DIR, "ESMC_PCA_features.csv")
df_pca_final.to_csv(pca_csv, index=False)

print(f"✅ PCA-reduced features saved")
print(f"   File  : {pca_csv}")
print(f"   Shape : {df_pca_final.shape} "
      f"({n_95} dims vs original {embedding_dim})")


# ============================================================
#   CELL 19 — Save Final Outputs
# ============================================================

df_esmc.to_csv(OUTPUT_CSV, index=False)
np.save(OUTPUT_NPY, embeddings)

print("=" * 55)
print("✅ ESMC Feature Extraction COMPLETE")
print(f"   Full CSV   : {OUTPUT_CSV}")
print(f"   Shape      : {df_esmc.shape}")
print(f"   Raw .npy   : {OUTPUT_NPY}")
print(f"   PCA CSV    : {pca_csv}")
print(f"   PCA Shape  : {df_pca_final.shape}")
print(f"   Figures    : {FIGURES_DIR}")
print(f"   Model      : {MODEL_NAME}")
print(f"   Embed dim  : {embedding_dim}")
print(f"   Pooling    : {POOLING}")
print("=" * 55)