"""
Protein Language Model (PLM) embedding extractor.

Supports:
  - ESM-2 (any variant) via fair-esm
  - ProtT5 via HuggingFace transformers (optional)

Embeddings are extracted as mean-pooled last hidden states,
giving a fixed-size vector regardless of peptide length.
All models are used in frozen (no-gradient) inference mode.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


def _mean_pool(token_representations: torch.Tensor,
               seq_len: int) -> np.ndarray:
    """Mean-pool token embeddings, excluding BOS/EOS tokens."""
    # token_representations shape: (seq_len + 2,  embed_dim)
    emb = token_representations[1: seq_len + 1]  # exclude BOS and EOS
    return emb.mean(dim=0).cpu().numpy()


class ESM2Extractor:
    """
    Extract mean-pooled embeddings from any ESM-2 model.

    Args:
        model_name: one of the ESM-2 model strings, e.g.
            'esm2_t6_8M_UR50D'   → dim=480
            'esm2_t12_35M_UR50D'  → dim=640
            'esm2_t30_150M_UR50D' → dim=640
            'esm2_t33_650M_UR50D' → dim=1280
        layer: which transformer layer to extract from (default: last layer)
        device: 'cpu' or 'cuda'
        batch_size: number of sequences per forward pass
    """

    def __init__(self, model_name: str = "esm2_t6_8M_UR50D",
                 layer: Optional[int] = None,
                 device: str = "cpu",
                 batch_size: int = 16):
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._layer = layer

    def _load(self):
        if self._model is not None:
            return
        try:
            import esm as esm_lib
        except ImportError:
            raise ImportError(
                "fair-esm is not installed. Run: pip install fair-esm"
            )
        log.info(f"Loading {self.model_name}...")
        loader = getattr(esm_lib.pretrained, self.model_name)
        model, alphabet = loader()
        model = model.to(self.device).eval()
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        if self._layer is None:
            self._layer = model.num_layers
        log.info(f"Loaded {self.model_name} — extracting layer {self._layer}")

    @torch.no_grad()
    def extract_batch(self, sequences: List[str],
                      show_progress: bool = True) -> np.ndarray:
        self._load()
        from tqdm import tqdm

        all_embeddings = []
        batches = [sequences[i: i + self.batch_size]
                   for i in range(0, len(sequences), self.batch_size)]

        it = tqdm(batches, desc=f"ESM2[{self.model_name}]") if show_progress else batches

        for batch_seqs in it:
            data = [(f"seq{i}", s) for i, s in enumerate(batch_seqs)]
            _, _, tokens = self._batch_converter(data)
            tokens = tokens.to(self.device)

            results = self._model(
                tokens,
                repr_layers=[self._layer],
                return_contacts=False
            )
            token_reps = results["representations"][self._layer]  # (B, L+2, D)

            for j, seq in enumerate(batch_seqs):
                emb = _mean_pool(token_reps[j], len(seq))
                all_embeddings.append(emb)

        return np.stack(all_embeddings)  # (N, D)

    def extract(self, sequence: str) -> np.ndarray:
        return self.extract_batch([sequence], show_progress=False)[0]

    @property
    def embed_dim(self) -> int:
        self._load()
        return self._model.embed_dim


class ProtT5Extractor:
    """
    Extract mean-pooled embeddings from ProtT5-XL-UniRef50 (encoder only).
    Uses HuggingFace transformers.
    """

    def __init__(self, model_id: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
                 device: str = "cpu",
                 batch_size: int = 8):
        self.model_id = model_id
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers sentencepiece"
            )
        log.info(f"Loading {self.model_id} (this may take a while)...")
        self._tokenizer = T5Tokenizer.from_pretrained(
            self.model_id, do_lower_case=False
        )
        self._model = T5EncoderModel.from_pretrained(self.model_id)
        self._model = self._model.to(self.device).eval()
        log.info("ProtT5 loaded.")

    @torch.no_grad()
    def extract_batch(self, sequences: List[str],
                      show_progress: bool = True) -> np.ndarray:
        self._load()
        from tqdm import tqdm

        # ProtT5 expects space-separated amino acids
        spaced = [" ".join(list(seq)) for seq in sequences]
        all_embeddings = []

        batches_seq = [sequences[i: i + self.batch_size]
                       for i in range(0, len(sequences), self.batch_size)]
        batches_sp = [spaced[i: i + self.batch_size]
                      for i in range(0, len(spaced), self.batch_size)]

        it = zip(batches_seq, batches_sp)
        if show_progress:
            it = tqdm(list(it), desc="ProtT5")

        for seqs_b, spaced_b in it:
            enc = self._tokenizer.batch_encode_plus(
                spaced_b,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)

            out = self._model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = out.last_hidden_state  # (B, L, 1024)

            for j, seq in enumerate(seqs_b):
                # mean pool over actual sequence positions
                mask_j = attn_mask[j].bool()
                emb = hidden[j][mask_j].mean(dim=0).cpu().numpy()
                all_embeddings.append(emb)

        return np.stack(all_embeddings)

    def extract(self, sequence: str) -> np.ndarray:
        return self.extract_batch([sequence], show_progress=False)[0]


# ─── combined PLM extractor ──────────────────────────────────────────────────

class PLMFeatureExtractor:
    """
    Concatenates embeddings from multiple PLM models.

    Default config uses two ESM-2 variants (no ProtT5 by default
    to keep dependencies lighter — set use_prot_t5=True to add it).

    Output dim = sum of individual model dims.
    """

    def __init__(self, device: str = "cpu",
                 batch_size: int = 16,
                 use_esm2_small: bool = True,
                 use_esm2_medium: bool = True,
                 use_prot_t5: bool = False):
        self.extractors: Dict[str, object] = {}

        if use_esm2_small:
            self.extractors["esm2_small"] = ESM2Extractor(
                model_name="esm2_t6_8M_UR50D",
                layer=6, device=device, batch_size=batch_size
            )
        if use_esm2_medium:
            self.extractors["esm2_medium"] = ESM2Extractor(
                model_name="esm2_t12_35M_UR50D",
                layer=12, device=device, batch_size=batch_size
            )
        if use_prot_t5:
            self.extractors["prot_t5"] = ProtT5Extractor(
                device=device, batch_size=max(1, batch_size // 4)
            )

        if not self.extractors:
            raise ValueError("At least one PLM model must be enabled.")

    def extract_batch(self, sequences: List[str],
                      show_progress: bool = True) -> np.ndarray:
        """Extract and concatenate embeddings from all models."""
        parts = []
        for name, extractor in self.extractors.items():
            log.info(f"Extracting {name} embeddings ({len(sequences)} sequences)...")
            emb = extractor.extract_batch(sequences, show_progress=show_progress)
            parts.append(emb)
            log.info(f"  {name}: shape={emb.shape}")

        return np.concatenate(parts, axis=1)  # (N, sum_dims)

    def extract(self, sequence: str) -> np.ndarray:
        return self.extract_batch([sequence], show_progress=False)[0]

    def feature_names(self, dims: Optional[Dict[str, int]] = None) -> List[str]:
        """Generate feature names — requires knowing each model's output dim."""
        names = []
        defaults = {"esm2_small": 480, "esm2_medium": 640, "prot_t5": 1024}
        for name in self.extractors:
            d = (dims or defaults).get(name, 512)
            names.extend([f"PLM_{name}_{i}" for i in range(d)])
        return names

    def save(self, embeddings: np.ndarray, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        log.info(f"Saved PLM embeddings → {path} (shape={embeddings.shape})")

    @staticmethod
    def load(path: str) -> np.ndarray:
        emb = np.load(path)
        log.info(f"Loaded PLM embeddings from {path} (shape={emb.shape})")
        return emb
