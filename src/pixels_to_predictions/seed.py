"""Deterministic-seed helpers.

We don't promise bit-identical reruns (torch's non-determinism on GPU makes that
expensive), but we do want reproducible data splits, prompt orderings, and
hyperparameter samples.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs.

    Also sets PYTHONHASHSEED so dict iteration is stable across subprocesses in the
    overnight optimizer.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)  # noqa: NPY002 — legacy API is fine here
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
