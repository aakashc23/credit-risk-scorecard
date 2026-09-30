"""Score scaling: PD <-> score and the points-per-attribute scorecard.

``score = offset + factor * ln(odds_good)``, ``factor = PDO / ln 2``,
``offset = BASE_SCORE - factor * ln(BASE_ODDS)``. Scores are clipped to SCORE_MIN..SCORE_MAX.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scorecard.config import BASE_ODDS, BASE_SCORE, PDO, SCORE_MAX, SCORE_MIN
from scorecard.woe import WoeBinner

EPS = 1e-12


def factor() -> float:
    return PDO / np.log(2)


def offset() -> float:
    return BASE_SCORE - factor() * np.log(BASE_ODDS)


def pd_to_score(pd_: np.ndarray | pd.Series, clip: bool = True) -> np.ndarray:
    """Score from probability of default; higher score = lower risk."""
    p = np.clip(np.asarray(pd_, float), EPS, 1 - EPS)
    score = offset() + factor() * np.log((1 - p) / p)
    return np.clip(score, SCORE_MIN, SCORE_MAX) if clip else score


def score_to_pd(score: np.ndarray | pd.Series | float) -> np.ndarray:
    """Inverse of ``pd_to_score`` (ignoring clipping)."""
    return 1 / (1 + np.exp((np.asarray(score, float) - offset()) / factor()))


def scorecard_points(binner: WoeBinner, model: LogisticRegression) -> pd.DataFrame:
    """Points per attribute: ``-(beta_j * WOE_ij + alpha / n) * factor + offset / n``, rounded.

    Summing a row's points over the model features reproduces its (unclipped) score up to
    rounding (at most about one point per feature).
    """
    features = list(model.feature_names_in_)
    n = len(features)
    alpha = float(model.intercept_[0])
    coef = dict(zip(features, model.coef_[0], strict=True))
    table = binner.bin_table()
    table = table[table["feature"].isin(features)].copy()
    beta = table["feature"].map(coef)
    table["coefficient"] = beta
    table["points"] = np.round(-(beta * table["woe"] + alpha / n) * factor() + offset() / n)
    cols = ["feature", "bin", "n", "bad_rate", "woe", "coefficient", "points"]
    return table[cols].reset_index(drop=True)


def points_score(binner: WoeBinner, points: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Sum scorecard points for each row of ``df`` (bin lookup, no model needed)."""
    features = points["feature"].unique().tolist()
    labels = binner.bin_labels(df, features)
    total = pd.Series(0.0, index=df.index)
    for feat in features:
        lookup = points[points["feature"] == feat].set_index("bin")["points"]
        total += labels[feat].map(lookup).fillna(0.0)
    return total
