"""
CHANGES:
    - added cross-validation (k-fold)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:

    
    idx = np.arange(len(y))
    
    # Keep test set for running solution.py
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    # K-fold
    n_folds = 5
    y_train_val = y[idx_train_val]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx_fold, val_idx_fold in skf.split(idx_train_val, y_train_val):
        train_original = idx_train_val[train_idx_fold]
        val_original = idx_train_val[val_idx_fold]
        splits.append((train_original, val_original, idx_test))
    
    return splits