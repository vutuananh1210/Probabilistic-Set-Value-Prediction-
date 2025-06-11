# data/data_prepare.py
"""
Create one <name>.npz per dataset containing:
    • proba      – list[ndarray]  (prob. preds per fold)
    • fold_idx   – list[ndarray]  (indices of the corresponding test rows)
    • classes    – ndarray (for MCC)  OR  None (for MLC/MDC)

The details of *how* we read the CSV and *which* Random-Forest engine
we use depend on the task type and are encapsulated in two plug-ins:
    1) DatasetLoader  – knows how to parse raw CSV   (MCC vs MLC/MDC)
    2) ModelRunner    – knows how to fit/predict RF (RF vs BRF)
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


# ==========================================================================
# 1. CSV → ndarray              (two loaders)
# ==========================================================================
#just an abstract base class for loaders

class DatasetLoader(ABC):
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    @abstractmethod
    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Return X, y  (both numpy arrays)."""

#This is the loader for MCC datasets. Here the last column of the csv is the class label.
class MCCLoader(DatasetLoader):
    """Label = last column; may be categorical."""

    def load(self):
        df = pd.read_csv(self.csv_path, delimiter=";")
        X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
        y = df.iloc[:, -1].to_numpy()
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y)
        return X, y

# This is the loader for MLC and MDC datasets. Here the labels are in multiple columns.
#the csv files are expected to have feature columns starting with 'feature' and label columns starting with 'class'

class MultiLabelLoader(DatasetLoader):
    """
    Works for both MLC and MDC.

    * feature columns start with 'feature'
    * label  columns start with 'class'  or 'y'
    """

    def load(self):
        df = pd.read_csv(self.csv_path, delimiter=";")
        feat_mask = df.columns.str.lower().str.startswith("feature")
        lab_mask  = (
            df.columns.str.lower().str.startswith("class")
        )
        X = df.loc[:, feat_mask].to_numpy(dtype=np.float32)
        y = df.loc[:, lab_mask].to_numpy()
        return X, y



# 2. RF runner plug-ins          (single-label RF vs binary-relevance RF)


class ModelRunner(ABC):
    """Wraps model instantiation, fitting and probability prediction."""

    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...
    @property
    @abstractmethod
    def classes_(self): ...


# ---- 2.1 single-label (MCC) ---------------------------------------------
class RFRunner(ModelRunner):
    def __init__(self, n_estimators=300, random_state=42, n_jobs=-1):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    
    def fit(self, X, y):
        self.rf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.rf.predict_proba(X)

    @property
    def classes_(self):
        return self.rf.classes_


# ---- 2.2 binary relevance (MLC & MDC) -----------------------------------
class BRF(ModelRunner, BaseEstimator, ClassifierMixin):
    """
    Binary Relevance wrapper around Random-Forest – one RF per label.
    Parallelised with joblib.
    """

    def __init__(self, n_estimators=400, n_jobs=-1, random_state=42):
        self.n_estimators = n_estimators
        self.n_jobs       = n_jobs
        self.random_state = random_state

        self._base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    # ------------- sklearn API ------------------------------------------
    def fit(self, X, y):
        Xc, yc = check_X_y(X, y, multi_output=True)
        n_labels = yc.shape[1]

        def _fit_one(j):
            clf = clone(self._base_rf)
            clf.fit(Xc, yc[:, j])
            return clf

        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(j) for j in range(n_labels)
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "models_")
        Xc = check_array(X, dtype=np.float32)

        probas = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda m, X: m.predict_proba(X))(m, Xc) for m in self.models_
        )
        
        # return np.stack(probas, axis=1)
        return probas

    # @property
    # def classes_(self):
    #     #  not  useful for MLC/MDC, but required by RFRunner interface
    #     return None


    @property
    def classes_(self):
        # Gather the classes_ array from each binary RF
        return [clf.classes_ for clf in self.models_]



# 3. public interface ---------------------------------------------------------

_LOADER = {"MCC": MCCLoader, "MLC": MultiLabelLoader, "MDC": MultiLabelLoader}
_RUNNER = {"MCC": RFRunner,  "MLC": BRF,              "MDC": BRF}


def prepare_all(
    task_type: str,
    dataset_names: Sequence[str],
    data_dir: str | Path = "data",
    out_dir: str | Path  = "cache",
    k: int = 10,
    random_state: int = 42,
) -> list[Path]:
    """
    Iterate over `dataset_names` (without .csv extension), run 10-fold CV
    with the task-specific Random-Forest, and dump
    <out_dir>/<name>_<task_type>.npz  for each dataset.

    Returns list of created NPZ paths.
    """
    task_type = task_type.upper()
    if task_type not in _LOADER:
        raise ValueError(f"Unknown task_type: {task_type}, it must be one of {list(_LOADER.keys())}")

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for name in dataset_names:
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        # 1) X, y
        X, y = _LOADER[task_type](csv_path).load()

        # 2) folds
        # if task_type == "MCC":
        #     splitter = KFold(
        #         n_splits=k, shuffle=True, random_state=random_state
        #     )
        #     folds = list(splitter.split(X, y))
        # else:

        splitter = KFold(n_splits=k, shuffle=True, random_state=random_state)
        folds = list(splitter.split(X))

        # 3) cross-val loop
        fold_probs, fold_indices = [], []
        for tr, te in folds:
            model = _RUNNER[task_type]()   # fresh model each fold
            model.fit(X[tr], y[tr])
            fold_probs.append(model.predict_proba(X[te]))
            fold_indices.append(te)

        # 4) write .npz
        npz_path = out_dir / f"{name}_{task_type}.npz"
        np.savez_compressed(
            npz_path,
            probas=np.array(fold_probs, dtype=object),
            fold_indices=np.array(fold_indices, dtype=object),
            y = y,
            classes= np.array(model.classes_, dtype=object)
        )
        print(f"done ! {csv_path.name:<25s} to {npz_path.name}")
        created.append(npz_path)

    return created
