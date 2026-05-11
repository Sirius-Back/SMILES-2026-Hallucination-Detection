"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
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

    # Фиксированный hold-out test (15%)
    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 5-fold по оставшимся данным
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []
    for idx_tr, idx_val in skf.split(idx_trainval, y[idx_trainval]):
        splits.append((
            idx_trainval[idx_tr],
            idx_trainval[idx_val],
            idx_test,
        ))
    return splits