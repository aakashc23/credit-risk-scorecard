"""Discrimination, calibration and stability metrics.

Convention: ``y`` is 1 for bad and ``risk`` is anything where higher = riskier (PD, or
``-score``). Decile/PSI helpers take the score itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from scorecard.config import N_DECILES

PSI_EPS = 1e-4


def auc(y, risk) -> float:
    return float(roc_auc_score(y, risk))


def gini(y, risk) -> float:
    return 2 * auc(y, risk) - 1


def ks(y, risk) -> float:
    """Max gap between the cumulative bad and good distributions."""
    fpr, tpr, _ = roc_curve(y, risk)
    return float(np.max(tpr - fpr))


def decile_table(y, score, pd_, n_bins: int = N_DECILES) -> pd.DataFrame:
    """Score deciles, worst (lowest score) first, with cumulative bad capture."""
    frame = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score, float),
                          "pd": np.asarray(pd_, float)})
    frame["decile"] = pd.qcut(frame["score"].rank(method="first"), n_bins, labels=False) + 1
    grp = frame.groupby("decile")
    table = pd.DataFrame({
        "n": grp.size(), "bads": grp["y"].sum(), "bad_rate": grp["y"].mean(),
        "mean_pd": grp["pd"].mean(), "min_score": grp["score"].min(),
        "max_score": grp["score"].max(),
    })
    table["cum_bad_capture"] = table["bads"].cumsum() / table["bads"].sum()
    goods = table["n"] - table["bads"]
    table["cum_good"] = goods.cumsum() / goods.sum()
    return table.reset_index()


def calibration_table(y, pd_, n_bins: int = N_DECILES) -> pd.DataFrame:
    """Mean predicted PD vs observed bad rate by PD decile (lowest PD first)."""
    frame = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_, float)})
    frame["decile"] = pd.qcut(frame["pd"].rank(method="first"), n_bins, labels=False) + 1
    table = frame.groupby("decile").agg(n=("y", "size"), mean_pd=("pd", "mean"),
                                        observed_bad_rate=("y", "mean")).reset_index()
    table["abs_error"] = (table["mean_pd"] - table["observed_bad_rate"]).abs()
    return table


def psi(expected, actual, n_bins: int = N_DECILES) -> float:
    """Population stability index; bin edges are deciles of the ``expected`` sample."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], edges, [np.inf]])
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.maximum(e, PSI_EPS), np.maximum(a, PSI_EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def grade_to_ordinal(sub_grade: pd.Series) -> pd.Series:
    """Map LC sub-grades A1..G5 to 1..35 (higher = riskier); unparsable -> NaN."""
    s = sub_grade.astype("string").str.strip().str.upper()
    parts = s.str.extract(r"^([A-G])([1-5])$")
    letter = parts[0].map(lambda c: ord(c) - ord("A") if isinstance(c, str) else np.nan)
    number = pd.to_numeric(parts[1], errors="coerce")
    return (letter * 5 + number).astype("float64")
