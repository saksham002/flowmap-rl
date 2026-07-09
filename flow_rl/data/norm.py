"""
Modular state/action normalization.

Statistics (per-dim min/max/mean/std) are computed once from the dataset and
saved to a .json (path comes from the config). A `Normalizer` loads them and is
applied as a transform inside the dataloader (and for un-normalizing actions /
normalizing states at eval time).
"""

import json
import os
import numpy as np


def compute_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype = np.float64)
    return {
        "min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
        "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
    }


def build_and_save(path: str, state: np.ndarray, action: np.ndarray) -> dict:
    stats = {"state": compute_stats(state), "action": compute_stats(action)}
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w") as f:
        json.dump(stats, f, indent = 2)
    return stats


class Normalizer:
    """Applies min_max (-> [-1, 1]) or mean_std normalization per field."""

    def __init__(self, stats: dict, mode: str = "min_max"):
        assert mode in ("min_max", "mean_std"), mode
        self.mode = mode
        self.s = {
            field: {k: np.asarray(v, dtype = np.float32) for k, v in st.items()}
            for field, st in stats.items()
        }

    @classmethod
    def from_json(cls, path: str, mode: str = "min_max") -> "Normalizer":
        with open(path) as f:
            return cls(json.load(f), mode)

    def normalize(self, x: np.ndarray, field: str) -> np.ndarray:
        st = self.s[field]
        x = np.asarray(x, dtype = np.float32)
        if self.mode == "min_max":
            return (2.0 * (x - st["min"]) / (st["max"] - st["min"] + 1e-8) - 1.0).astype(np.float32)
        return ((x - st["mean"]) / (st["std"] + 1e-8)).astype(np.float32)

    def unnormalize(self, x: np.ndarray, field: str) -> np.ndarray:
        st = self.s[field]
        x = np.asarray(x, dtype = np.float32)
        if self.mode == "min_max":
            return ((x + 1.0) / 2.0 * (st["max"] - st["min"]) + st["min"]).astype(np.float32)
        return (x * (st["std"] + 1e-8) + st["mean"]).astype(np.float32)
