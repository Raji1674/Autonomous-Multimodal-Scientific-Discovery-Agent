"""Reproducible randomness helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy. (Libraries like scikit-learn take their own random_state.)"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # only affects child processes
