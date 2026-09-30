"""Approval cut-off simulator: portfolio outcomes at each score cut-off.

``scored`` needs columns score, pd, ead, el, bad and realised_loss (one row per loan).
A loan is approved when ``score >= cutoff``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scorecard.config import CUTOFF_STEP, SCORE_MAX, SCORE_MIN

COLUMNS = [
    "cutoff", "n_total", "n_approved", "n_rejected", "approval_rate", "rejection_rate",
    "expected_bad_rate", "observed_bad_rate", "approved_exposure", "expected_loss",
    "expected_loss_rate", "realised_loss",
]


def _suffix_sums(sorted_values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sum of ``sorted_values[i:]`` for each i in ``idx`` (one cumulative sum)."""
    csum = np.concatenate([[0.0], np.cumsum(sorted_values, dtype=float)])
    return csum[-1] - csum[idx]


def _cutoff_rows(scored: pd.DataFrame, cutoffs: np.ndarray) -> pd.DataFrame:
    order = np.argsort(scored["score"].to_numpy(float), kind="stable")
    score = scored["score"].to_numpy(float)[order]
    n = len(score)
    idx = np.searchsorted(score, cutoffs, side="left")  # first approved position
    approved = n - idx
    sums = {c: _suffix_sums(scored[c].to_numpy(float)[order], idx)
            for c in ["pd", "ead", "el", "bad", "realised_loss"]}
    has = approved > 0
    safe_n = np.where(has, approved, 1)
    safe_exposure = np.where(sums["ead"] > 0, sums["ead"], 1.0)
    return pd.DataFrame({
        "cutoff": cutoffs, "n_total": n, "n_approved": approved, "n_rejected": idx,
        "approval_rate": approved / n if n else np.nan,
        "rejection_rate": idx / n if n else np.nan,
        "expected_bad_rate": np.where(has, sums["pd"] / safe_n, np.nan),
        "observed_bad_rate": np.where(has, sums["bad"] / safe_n, np.nan),
        "approved_exposure": sums["ead"], "expected_loss": sums["el"],
        "expected_loss_rate": np.where(sums["ead"] > 0, sums["el"] / safe_exposure, np.nan),
        "realised_loss": sums["realised_loss"],
    })[COLUMNS]


def cutoff_table(scored: pd.DataFrame, step: int = CUTOFF_STEP) -> pd.DataFrame:
    """Outcomes for every cut-off from SCORE_MIN to SCORE_MAX in ``step`` increments."""
    return _cutoff_rows(scored, np.arange(SCORE_MIN, SCORE_MAX + 1, step, dtype=float))


def cutoff_summary(scored: pd.DataFrame, cutoff: float) -> dict:
    """Same fields as ``cutoff_table`` for a single cut-off."""
    row = _cutoff_rows(scored, np.array([float(cutoff)])).iloc[0]
    return {k: (int(v) if k.startswith("n_") else float(v)) for k, v in row.items()}
