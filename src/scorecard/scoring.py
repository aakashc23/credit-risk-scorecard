"""Score scaling: PD <-> score and the points-per-attribute scorecard.

``score = offset + factor * ln(odds_good)``, ``factor = PDO / ln 2``,
``offset = BASE_SCORE - factor * ln(BASE_ODDS)``. Scores are clipped to SCORE_MIN..SCORE_MAX.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scorecard.config import BASE_ODDS, BASE_SCORE, PDO, RISK_BANDS, SCORE_MAX, SCORE_MIN
from scorecard.woe import OTHER, WoeBinner

if TYPE_CHECKING:  # sklearn is only needed to *fit*; scoring/the dashboard must not import it
    from sklearn.linear_model import LogisticRegression

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


def risk_band(score: float) -> str:
    """Label of the ``RISK_BANDS`` band a score falls in."""
    for minimum, label in RISK_BANDS:
        if score >= minimum:
            return label
    return RISK_BANDS[-1][1]


UNBINNED = "Unbinned"  # label of the catch-all row (same label as woe.label_feature uses)


def neutral_points(alpha: float, n_features: int) -> float:
    """Points for a value that falls in no bin (WOE = 0): ``-(alpha / n) * factor + offset / n``."""
    return -(alpha / n_features) * factor() + offset() / n_features


def _has_catch_all(spec: dict) -> bool:
    """True if missing / unseen values already land in a real bin of this feature."""
    if spec["kind"] == "numeric":
        return any(b.get("is_missing") for b in spec["bins"])
    return any(OTHER in b["categories"] for b in spec["bins"])


def scorecard_points(binner: WoeBinner, model: LogisticRegression) -> pd.DataFrame:
    """Points per attribute: ``-(beta_j * WOE_ij + alpha / n) * factor + offset / n``, rounded.

    Summing a row's points over the model features reproduces its (unclipped) score up to
    rounding (at most about one point per feature). A feature with no bin for missing /
    unseen values gets an extra ``Unbinned`` row (0 training loans, WOE 0): that is the
    neutral-points rule ``transform_feature`` implies, so scoring stays consistent.
    """
    features = list(model.feature_names_in_)
    n = len(features)
    alpha = float(model.intercept_[0])
    coef = dict(zip(features, model.coef_[0], strict=True))
    bins = binner.bin_table()
    parts = []
    for feat in features:
        part = bins[bins["feature"] == feat].copy()
        if not _has_catch_all(binner.bins_[feat]):
            extra = {"feature": feat, "bin": UNBINNED, "n": 0, "bads": 0, "bad_rate": np.nan,
                     "woe": 0.0, "iv_contrib": 0.0}
            part = pd.concat([part, pd.DataFrame([extra])], ignore_index=True)
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    beta = table["feature"].map(coef)
    table["coefficient"] = beta
    table["points"] = np.round(-(beta * table["woe"] + alpha / n) * factor() + offset() / n)
    cols = ["feature", "bin", "n", "bad_rate", "woe", "coefficient", "points"]
    return table[cols].reset_index(drop=True)


def points_score(binner: WoeBinner, points: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Sum scorecard points for each row of ``df`` (bin lookup, no model needed).

    A label missing from the table (only possible for hand-built tables) scores the
    ``Unbinned`` row of that feature, or 0 if it has none.
    """
    features = points["feature"].unique().tolist()
    labels = binner.bin_labels(df, features)
    total = pd.Series(0.0, index=df.index)
    for feat in features:
        part = points[points["feature"] == feat].set_index("bin")["points"]
        total += labels[feat].map(part).fillna(part.get(UNBINNED, 0.0))
    return total
