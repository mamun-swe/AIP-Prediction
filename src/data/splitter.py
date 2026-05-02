"""
Stratified train/test split.
CRITICAL: All feature extraction must be fitted on train only, 
then applied to test. This module enforces that by saving splits 
before any feature work begins.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

log = get_logger(__name__)


def split_and_save(df: pd.DataFrame,
                   splits_dir: str,
                   train_ratio: float = 0.8,
                   random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split → save train.csv and test.csv.

    Args:
        df: DataFrame with columns [sequence, label]
        splits_dir: directory to save train.csv and test.csv
        train_ratio: fraction for training
        random_state: reproducibility seed

    Returns:
        (train_df, test_df)
    """
    Path(splits_dir).mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = Path(splits_dir) / "train.csv"
    test_path = Path(splits_dir) / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    pos_train = (train_df.label == 1).sum()
    neg_train = (train_df.label == 0).sum()
    pos_test = (test_df.label == 1).sum()
    neg_test = (test_df.label == 0).sum()

    log.info(f"Train: {len(train_df)} ({pos_train} pos / {neg_train} neg)")
    log.info(f"Test : {len(test_df)} ({pos_test} pos / {neg_test} neg)")
    log.info(f"Saved → {splits_dir}")

    return train_df, test_df


def load_splits(splits_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved train/test splits."""
    train_df = pd.read_csv(Path(splits_dir) / "train.csv")
    test_df = pd.read_csv(Path(splits_dir) / "test.csv")
    log.info(f"Loaded train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df
