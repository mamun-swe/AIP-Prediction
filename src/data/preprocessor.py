"""
Sequence preprocessing: filtering, cleaning, redundancy removal.
All logic operates on the TRAINING set only — test set is passed through
the same rules but without re-fitting any thresholds.
"""
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
NON_STANDARD = set("BOUJXZ")


def load_fasta(filepath: str) -> List[Tuple[str, str]]:
    """Parse a FASTA file → list of (header, sequence)."""
    records = []
    header, seq_parts = None, []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts).upper()))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts).upper()))
    return records


def load_csv(filepath: str, seq_col: str = "sequence",
             label_col: str = "label") -> pd.DataFrame:
    """Load CSV with sequence and label columns."""
    df = pd.read_csv(filepath)
    assert seq_col in df.columns, f"Column '{seq_col}' not found in {filepath}"
    assert label_col in df.columns, f"Column '{label_col}' not found in {filepath}"
    df = df[[seq_col, label_col]].rename(columns={seq_col: "sequence", label_col: "label"})
    return df


def validate_sequence(seq: str, min_len: int = 5, max_len: int = 40) -> bool:
    """Return True only if sequence passes all filters."""
    if not (min_len <= len(seq) <= max_len):
        return False
    if not set(seq).issubset(STANDARD_AAS):
        return False
    return True


def clean_dataframe(df: pd.DataFrame,
                    min_len: int = 5,
                    max_len: int = 40) -> pd.DataFrame:
    """Apply all sequence-level filters and return cleaned DataFrame."""
    original = len(df)
    df = df.copy()
    df["sequence"] = df["sequence"].str.upper().str.strip()

    # remove non-standard AAs
    mask_nonstandard = df["sequence"].apply(
        lambda s: bool(set(s) & NON_STANDARD)
    )
    df = df[~mask_nonstandard]

    # length filter
    mask_len = df["sequence"].apply(lambda s: min_len <= len(s) <= max_len)
    df = df[mask_len]

    # drop duplicates (keep first)
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)

    log.info(f"Cleaned {original} → {len(df)} sequences "
             f"(removed {original - len(df)})")
    return df


def remove_redundancy_python(df: pd.DataFrame,
                              threshold: float = 0.8) -> pd.DataFrame:
    """
    Pure-Python sequence-identity clustering (CD-HIT replacement).
    Uses pairwise local alignment identity for short peptides.
    This is O(N^2) — acceptable for AIP dataset size (~4000 sequences).
    """
    log.info("Running redundancy removal (Python implementation)...")
    sequences = df["sequence"].tolist()
    n = len(sequences)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            identity = _sequence_identity(sequences[i], sequences[j])
            if identity >= threshold:
                keep[j] = False

    result = df[keep].reset_index(drop=True)
    log.info(f"Redundancy removal: {n} → {len(result)} sequences "
             f"(threshold={threshold})")
    return result


def _sequence_identity(s1: str, s2: str) -> float:
    """Compute pairwise sequence identity using simple alignment."""
    # Use Hamming distance for same-length seqs; use overlap for different lengths
    if len(s1) == len(s2):
        matches = sum(a == b for a, b in zip(s1, s2))
        return matches / len(s1)
    # For different lengths: slide shorter over longer, take max identity
    short, long = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    max_id = 0.0
    for start in range(len(long) - len(short) + 1):
        segment = long[start: start + len(short)]
        matches = sum(a == b for a, b in zip(short, segment))
        identity = matches / len(short)
        max_id = max(max_id, identity)
    return max_id


def build_dataset_from_fastas(pos_fasta: str, neg_fasta: str,
                               min_len: int = 5, max_len: int = 40,
                               redundancy_threshold: float = 0.8,
                               output_csv: str = None) -> pd.DataFrame:
    """
    Full pipeline: load FASTAs → clean → remove redundancy → save CSV.

    Args:
        pos_fasta: path to positive (AIP) FASTA
        neg_fasta: path to negative (non-AIP) FASTA
        min_len, max_len: length filters
        redundancy_threshold: CD-HIT equivalent threshold
        output_csv: if set, saves the combined DataFrame

    Returns:
        DataFrame with columns [sequence, label]
    """
    log.info("Loading positive sequences...")
    pos_records = load_fasta(pos_fasta)
    pos_df = pd.DataFrame(pos_records, columns=["header", "sequence"])
    pos_df["label"] = 1

    log.info("Loading negative sequences...")
    neg_records = load_fasta(neg_fasta)
    neg_df = pd.DataFrame(neg_records, columns=["header", "sequence"])
    neg_df["label"] = 0

    df = pd.concat([pos_df, neg_df], ignore_index=True)[["sequence", "label"]]

    log.info(f"Raw: {len(pos_df)} positives, {len(neg_df)} negatives")

    # Clean
    df = clean_dataframe(df, min_len=min_len, max_len=max_len)

    # Separate for per-class redundancy removal
    pos_clean = remove_redundancy_python(
        df[df.label == 1], threshold=redundancy_threshold)
    neg_clean = remove_redundancy_python(
        df[df.label == 0], threshold=redundancy_threshold)

    final = pd.concat([pos_clean, neg_clean], ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Final dataset: {len(final[final.label==1])} pos, "
             f"{len(final[final.label==0])} neg, "
             f"total={len(final)}")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_csv, index=False)
        log.info(f"Saved processed dataset → {output_csv}")

    return final


def build_dataset_from_csv(csv_path: str,
                            seq_col: str = "sequence",
                            label_col: str = "label",
                            min_len: int = 5,
                            max_len: int = 40,
                            redundancy_threshold: float = 0.8,
                            output_csv: str = None) -> pd.DataFrame:
    """Load from CSV, clean, remove redundancy, return DataFrame."""
    log.info(f"Loading from CSV: {csv_path}")
    df = load_csv(csv_path, seq_col=seq_col, label_col=label_col)
    df = clean_dataframe(df, min_len=min_len, max_len=max_len)

    pos_clean = remove_redundancy_python(
        df[df.label == 1], threshold=redundancy_threshold)
    neg_clean = remove_redundancy_python(
        df[df.label == 0], threshold=redundancy_threshold)

    final = pd.concat([pos_clean, neg_clean], ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Final: {len(final[final.label==1])} pos, "
             f"{len(final[final.label==0])} neg")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_csv, index=False)
        log.info(f"Saved → {output_csv}")

    return final
