"""
Shared utility functions for the AIP-ML-Stack project.
"""

import os
import random
import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from datetime import datetime


# ─────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config file and return as dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return cfg


# ─────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ─────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_artifact(obj, path: str):
    """Save any Python object with joblib (models, arrays, etc.)."""
    ensure_dir(str(Path(path).parent))
    joblib.dump(obj, path)
    logger.info(f"Saved artifact → {path}")


def load_artifact(path: str):
    """Load a joblib artifact."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    obj = joblib.load(path)
    logger.info(f"Loaded artifact ← {path}")
    return obj


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False):
    """Save DataFrame to CSV."""
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=index)
    logger.info(f"Saved DataFrame ({df.shape}) → {path}")


def load_dataframe(path: str) -> pd.DataFrame:
    """Load DataFrame from CSV."""
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded DataFrame ({df.shape}) ← {path}")
    return df


# ─────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_sequence(seq: str, valid_aas: set = VALID_AAS) -> bool:
    """Return True if sequence contains only standard amino acids."""
    return bool(seq) and all(aa in valid_aas for aa in seq.upper())


def clean_sequence(seq: str) -> str:
    """Uppercase and strip whitespace from a sequence."""
    return seq.strip().upper().replace(" ", "")


def filter_by_length(sequences: list, min_len: int = 5,
                     max_len: int = 40) -> list:
    """Return sequences within [min_len, max_len]."""
    return [s for s in sequences if min_len <= len(s) <= max_len]


# ─────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────

def setup_logger(log_dir: str = "results/logs", level: str = "INFO"):
    """Configure loguru to write to file and console."""
    ensure_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"
    logger.add(log_file, level=level, rotation="10 MB")
    logger.info(f"Logger initialized. Log file: {log_file}")


# ─────────────────────────────────────────────────────────
# Numpy / array helpers
# ─────────────────────────────────────────────────────────

def safe_concat(arrays: list, axis: int = 1) -> np.ndarray:
    """Concatenate list of numpy arrays; skip None entries."""
    valid = [a for a in arrays if a is not None]
    if not valid:
        raise ValueError("No valid arrays to concatenate.")
    return np.concatenate(valid, axis=axis)


def check_for_nan(X: np.ndarray, name: str = "Feature matrix") -> bool:
    """Warn if NaN/Inf present; return True if clean."""
    if np.isnan(X).any():
        logger.warning(f"{name} contains NaN values.")
        return False
    if np.isinf(X).any():
        logger.warning(f"{name} contains Inf values.")
        return False
    return True


def replace_nan(X: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN and Inf with fill value."""
    X = np.nan_to_num(X, nan=fill, posinf=fill, neginf=fill)
    return X
