# ============================================================
#   CELL 3 — Import Libraries
# ============================================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset

# Check GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Libraries loaded")
print(f"   Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ⚠️  No GPU detected — running on CPU (will be slower)")


# ============================================================
#   CELL 4 — Configuration (Edit paths here)
# ============================================================

FASTA_PATH  = "../../data/raw/AIP_ind.fasta"

OUTPUT_DIR  = "../../data/features"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "ProtBERT_features.csv")
OUTPUT_NPY  = os.path.join(OUTPUT_DIR, "ProtBERT_embeddings.npy")

FIGURES_DIR = "../../results/figures"

# ProtBERT model name from HuggingFace Hub
# Options:
#   "Rostlab/prot_bert"        — base model (768-dim embeddings)
#   "Rostlab/prot_bert_bfd"    — trained on BFD database (recommended)
MODEL_NAME  = "Rostlab/prot_bert_bfd"

# Batch size for inference (reduce if OOM error on GPU)
BATCH_SIZE  = 16

# Pooling strategy: how to get a fixed-size vector per sequence
#   "mean"  — average all token embeddings (recommended)
#   "cls"   — use [CLS] token embedding only
#   "max"   — max pooling across all tokens
POOLING     = "mean"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"✅ Features dir : {OUTPUT_DIR}")
print(f"✅ Figures dir  : {FIGURES_DIR}")
print(f"✅ Model        : {MODEL_NAME}")
print(f"✅ Batch size   : {BATCH_SIZE}")
print(f"✅ Pooling      : {POOLING}")
print(f"✅ Embedding dim: 1024 (ProtBERT-BFD) or 768 (ProtBERT)")


# ============================================================
#   CELL 5 — FASTA Parser
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
#   CELL 6 — Load ProtBERT Model & Tokenizer
# ============================================================
#
#  ProtBERT is a BERT model pre-trained on 216 million protein
#  sequences from UniRef100. It produces contextual embeddings
#  that capture evolutionary and structural information.
#
#  ⚠️  IMPORTANT: ProtBERT requires sequences to be formatted
#  with spaces between each amino acid:
#      "ACDE" → "A C D E"
#  Unknown/special characters should be replaced with X.
#
# ============================================================

print(f"Loading ProtBERT model: {MODEL_NAME}")
print("(First run will download ~1.6 GB — will be cached afterwards)")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case=False   # protein sequences are uppercase
)

model = BertModel.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()              # inference mode — disable dropout

print(f"✅ Model loaded successfully")
print(f"   Hidden size    : {model.config.hidden_size}")
print(f"   Num layers     : {model.config.num_hidden_layers}")
print(f"   Attention heads: {model.config.num_attention_heads}")


# ============================================================
#   CELL 7 — Sequence Preprocessing for ProtBERT
# ============================================================

def preprocess_sequence(sequence):
    """
    Format a protein sequence for ProtBERT:
    1. Replace unknown amino acids (B, Z, U, O) with X
    2. Insert spaces between every amino acid
       "ACDE" → "A C D E"
    This is mandatory for the ProtBERT tokenizer.
    """
    # Replace non-standard amino acids with X (unknown token)
    sequence = re.sub(r"[BUZOJ]", "X", sequence.upper())
    # Space-separated format required by ProtBERT
    return " ".join(list(sequence))


# Preview preprocessing
sample_seq = df_seq["sequence"].iloc[0]
print(f"Original  : {sample_seq}")
print(f"Processed : {preprocess_sequence(sample_seq)}")


# ============================================================
#   CELL 8 — PyTorch Dataset for Batched Inference
# ============================================================

class PeptideDataset(Dataset):
    """Simple dataset for batched ProtBERT inference."""

    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences  = [preprocess_sequence(s) for s in sequences]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.sequences[idx],
            return_tensors     = "pt",
            padding            = "max_length",
            truncation         = True,
            max_length         = self.max_length,
            add_special_tokens = True    # adds [CLS] and [SEP]
        )
        return {
            "input_ids"     : encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


# ============================================================
#   CELL 9 — ProtBERT Embedding Extraction
# ============================================================

def pool_embeddings(hidden_states, attention_mask, strategy="mean"):
    """
    Aggregate token-level embeddings into a single
    fixed-size vector per sequence.

    Strategies:
      mean — average over non-padding tokens (recommended)
             ignores [CLS], [SEP], and padding positions
      cls  — use [CLS] token (position 0) embedding only
      max  — element-wise max over non-padding tokens
    """
    if strategy == "cls":
        # [CLS] token is always at position 0
        return hidden_states[:, 0, :]

    # Create mask to ignore padding, [CLS] at pos 0, [SEP] at last pos
    # attention_mask: 1 = real token, 0 = padding
    mask = attention_mask.unsqueeze(-1).float()

    if strategy == "mean":
        # Sum of real token embeddings / number of real tokens
        sum_embeddings  = (hidden_states * mask).sum(dim=1)
        count_tokens    = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / count_tokens

    elif strategy == "max":
        # Replace padding positions with -inf before max
        hidden_states[mask == 0] = -1e9
        return hidden_states.max(dim=1).values

    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


def extract_protbert_embeddings(df, tokenizer, model, device,
                                batch_size=BATCH_SIZE,
                                pooling=POOLING):
    """
    Extract ProtBERT embeddings for all sequences.

    Returns:
        embeddings : np.ndarray of shape (N, hidden_size)
                     e.g. (245, 1024) for prot_bert_bfd
    """
    sequences = df["sequence"].tolist()
    dataset   = PeptideDataset(sequences, tokenizer)
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)

    all_embeddings = []
    total_batches  = len(loader)

    print(f"Extracting embeddings...")
    print(f"  Sequences   : {len(sequences)}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Batches     : {total_batches}")
    print(f"  Pooling     : {pooling}")
    print(f"  Device      : {device}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass through ProtBERT
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )

            # Last hidden state: (batch_size, seq_len, hidden_size)
            hidden_states = outputs.last_hidden_state

            # Pool to get one vector per sequence
            pooled = pool_embeddings(
                hidden_states, attention_mask, strategy=pooling
            )

            all_embeddings.append(pooled.cpu().numpy())

            # Progress update every 5 batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                done = (batch_idx + 1) * batch_size
                print(f"  [{batch_idx+1}/{total_batches}] "
                      f"{min(done, len(sequences))}/{len(sequences)} sequences done")

    embeddings = np.vstack(all_embeddings)
    print(f"\n✅ Embedding extraction complete")
    print(f"   Embedding shape : {embeddings.shape}")
    return embeddings

embeddings = extract_protbert_embeddings(
    df_seq, tokenizer, model, DEVICE
)


# ============================================================
#   CELL 10 — Build Feature DataFrame
# ============================================================

embedding_dim = embeddings.shape[1]
col_names     = [f"ProtBERT_{i}" for i in range(embedding_dim)]

df_emb = pd.DataFrame(embeddings, columns=col_names)

df_protbert = pd.concat([
    df_seq[["seq_id", "sequence", "label"]].reset_index(drop=True),
    df_emb
], axis=1)
df_protbert.insert(2, "length", df_protbert["sequence"].apply(len))

print(f"✅ Feature DataFrame built")
print(f"   Shape   : {df_protbert.shape}")
print(f"   Columns : seq_id, sequence, length, label, "
      f"ProtBERT_0 ... ProtBERT_{embedding_dim-1}")
df_protbert.iloc[:3, :8]


# ============================================================
#   CELL 11 — Validate Output
# ============================================================

emb_cols = [c for c in df_protbert.columns if c.startswith("ProtBERT_")]

print("── Validation ──────────────────────────────────────")
print(f"Embedding dimension : {len(emb_cols)}")
print(f"Missing values      : {df_protbert[emb_cols].isnull().sum().sum()}")
print(f"Value range         : {df_protbert[emb_cols].values.min():.4f} – "
      f"{df_protbert[emb_cols].values.max():.4f}")
print(f"Mean (pos)          : {df_protbert[df_protbert['label']==1][emb_cols].values.mean():.4f}")
print(f"Mean (neg)          : {df_protbert[df_protbert['label']==0][emb_cols].values.mean():.4f}")
print("✅ Validation complete")


# ============================================================
#   CELL 12 — Visualization 1: PCA Plot (2D)
# ============================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Running PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_protbert[emb_cols].values)

pca     = PCA(n_components=2, random_state=42)
X_pca   = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))

for label, color, name in [(1, "#2ecc71", "AIP (pos)"),
                            (0, "#e74c3c", "Non-AIP (neg)")]:
    mask = df_protbert["label"] == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=color, label=name, alpha=0.7,
               s=60, edgecolors="white", linewidths=0.5)

ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=12)
ax.set_title("PCA of ProtBERT Embeddings (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ProtBERT_PCA.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ PCA plot saved")


# ============================================================
#   CELL 13 — Visualization 2: t-SNE Plot (2D)
# ============================================================
from sklearn.manifold import TSNE

print("Running t-SNE (this may take ~30 seconds)...")

tsne    = TSNE(n_components=2, random_state=42,
               perplexity=30, n_iter=1000)
X_tsne  = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))

for label, color, name in [(1, "#2ecc71", "AIP (pos)"),
                            (0, "#e74c3c", "Non-AIP (neg)")]:
    mask = df_protbert["label"] == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=color, label=name, alpha=0.7,
               s=60, edgecolors="white", linewidths=0.5)

ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
ax.set_title("t-SNE of ProtBERT Embeddings (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ProtBERT_tSNE.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ t-SNE plot saved")


# ============================================================
#   CELL 14 — Visualization 3: UMAP Plot (2D)
# ============================================================
try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    import subprocess
    subprocess.run(["pip", "install", "umap-learn", "-q"])
    import umap

print("Running UMAP...")

reducer = umap.UMAP(n_components=2, random_state=42,
                    n_neighbors=15, min_dist=0.1)
X_umap  = reducer.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))

for label, color, name in [(1, "#2ecc71", "AIP (pos)"),
                            (0, "#e74c3c", "Non-AIP (neg)")]:
    mask = df_protbert["label"] == label
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               c=color, label=name, alpha=0.7,
               s=60, edgecolors="white", linewidths=0.5)

ax.set_xlabel("UMAP Dimension 1", fontsize=12)
ax.set_ylabel("UMAP Dimension 2", fontsize=12)
ax.set_title("UMAP of ProtBERT Embeddings (AIP vs Non-AIP)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ProtBERT_UMAP.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ UMAP plot saved")


# ============================================================
#   CELL 15 — Visualization 4: Embedding Heatmap (sample)
# ============================================================

# Sample 60 sequences for readability
sample     = df_protbert.sample(min(60, len(df_protbert)),
                                 random_state=42).sort_values("label")
# Show first 100 dimensions for visualization
n_show     = 100
sample_emb = sample[[f"ProtBERT_{i}" for i in range(n_show)]].values

fig, ax = plt.subplots(figsize=(20, 6))
im = ax.imshow(sample_emb.T, aspect="auto", cmap="coolwarm")

ax.set_xlabel("Peptide Sequences (sorted by label)", fontsize=12)
ax.set_ylabel(f"ProtBERT Embedding Dimensions (first {n_show})", fontsize=12)
ax.set_title(f"ProtBERT Embedding Heatmap "
             f"(sample of {len(sample)} sequences, first {n_show} dims)",
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="Embedding Value")

# Add label boundary line
n_neg  = (sample["label"] == 0).sum()
n_pos  = (sample["label"] == 1).sum()
ax.axvline(n_neg - 0.5, color="yellow", linewidth=2, linestyle="--",
           label=f"← Non-AIP ({n_neg}) | AIP ({n_pos}) →")
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ProtBERT_embedding_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("✅ Embedding heatmap saved")


# ============================================================
#   CELL 16 — Visualization 5: PCA Variance Explained
# ============================================================

pca_full   = PCA(random_state=42)
pca_full.fit(X_scaled)
cum_var    = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_95       = np.argmax(cum_var >= 95) + 1

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(cum_var) + 1), cum_var,
        color="#3498db", linewidth=2)
ax.axhline(95, color="#e74c3c", linestyle="--",
           label=f"95% variance → {n_95} components")
ax.axvline(n_95, color="#e74c3c", linestyle="--")
ax.fill_between(range(1, len(cum_var) + 1), cum_var,
                alpha=0.15, color="#3498db")

ax.set_xlabel("Number of PCA Components", fontsize=12)
ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
ax.set_title("PCA Variance Explained — ProtBERT Embeddings",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(1, min(200, len(cum_var)))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ProtBERT_PCA_variance.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ PCA variance plot saved")
print(f"   {n_95} components explain 95% of variance")
print(f"   (vs full {embedding_dim} dimensions)")


# ============================================================
#   CELL 17 — Optional: Dimensionality Reduction with PCA
#             Save a compressed version for ML models
# ============================================================

# Use n_95 components to retain 95% variance
pca_reduced  = PCA(n_components=n_95, random_state=42)
X_reduced    = pca_reduced.fit_transform(X_scaled)

col_names_r  = [f"ProtBERT_PCA_{i}" for i in range(n_95)]
df_reduced   = pd.DataFrame(X_reduced, columns=col_names_r)
df_pca_final = pd.concat([
    df_seq[["seq_id", "sequence", "label"]].reset_index(drop=True),
    df_reduced
], axis=1)

pca_csv = os.path.join(OUTPUT_DIR, "ProtBERT_PCA_features.csv")
df_pca_final.to_csv(pca_csv, index=False)

print(f"✅ PCA-reduced features saved")
print(f"   File  : {pca_csv}")
print(f"   Shape : {df_pca_final.shape}  "
      f"({n_95} dims vs original {embedding_dim})")


# ============================================================
#   CELL 18 — Save Final Outputs
# ============================================================

# Save full embedding CSV
df_protbert.to_csv(OUTPUT_CSV, index=False)

# Save raw numpy array (faster to load for DL models)
np.save(OUTPUT_NPY, embeddings)

print("=" * 55)
print("✅ ProtBERT Feature Extraction COMPLETE")
print(f"   Full CSV   : {OUTPUT_CSV}")
print(f"   Shape      : {df_protbert.shape}")
print(f"   Raw .npy   : {OUTPUT_NPY}")
print(f"   PCA CSV    : {pca_csv}")
print(f"   PCA Shape  : {df_pca_final.shape}")
print(f"   Figures    : {FIGURES_DIR}")
print(f"   Model used : {MODEL_NAME}")
print(f"   Pooling    : {POOLING}")
print("=" * 55)