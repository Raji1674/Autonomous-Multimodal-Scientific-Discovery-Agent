"""Small built-in datasets: offline, free, versioned with scikit-learn."""
from __future__ import annotations

import numpy as np
from sklearn import datasets as skd

REGISTRY = {
    "iris": skd.load_iris,
    "wine": skd.load_wine,
    "breast_cancer": skd.load_breast_cancer,
    "digits": skd.load_digits,
}


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return X, y and a small metadata dict (recorded in the experiment manifest)."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {sorted(REGISTRY)}")
    X, y = REGISTRY[name](return_X_y=True)
    meta = {"name": name, "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))), "source": "scikit-learn built-in"}
    return X, y, meta
