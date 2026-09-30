"""Feature selection and the sign-constrained logistic regression on WOE features."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scorecard.config import (
    MAX_CORR,
    MAX_MODEL_FEATURES,
    MIN_IV,
    RANDOM_STATE,
    SUSPICIOUS_IV,
)

logger = logging.getLogger(__name__)


def select_features(iv: pd.DataFrame, woe_train: pd.DataFrame) -> tuple[list[str], dict]:
    """Pick model features from the IV table.

    Steps: IV >= MIN_IV; flag (but keep) IV > SUSPICIOUS_IV; drop the lower-IV member of any
    pair with |corr| > MAX_CORR on the train WOE matrix; cap at MAX_MODEL_FEATURES by IV.
    Returns the selected features (strongest first) and a dict describing what was dropped.
    """
    ranked = iv.sort_values("iv", ascending=False)
    weak = ranked.loc[ranked["iv"] < MIN_IV, "feature"].tolist()
    strong = ranked[ranked["iv"] >= MIN_IV]
    suspicious = strong.loc[strong["iv"] > SUSPICIOUS_IV, "feature"].tolist()
    if suspicious:
        logger.warning("IV > %.2f, review for leakage: %s", SUSPICIOUS_IV, suspicious)

    corr = woe_train[strong["feature"].tolist()].corr().abs()
    kept: list[str] = []
    dropped_corr: dict[str, str] = {}
    for feat in strong["feature"]:
        partner = next((k for k in kept if corr.loc[feat, k] > MAX_CORR), None)
        if partner is None:
            kept.append(feat)
        else:
            dropped_corr[feat] = partner
    selected = kept[:MAX_MODEL_FEATURES]
    info = {"dropped_low_iv": weak, "suspicious_iv": suspicious,
            "dropped_correlated": dropped_corr, "dropped_cap": kept[MAX_MODEL_FEATURES:]}
    return selected, info


def _fit(X: pd.DataFrame, y) -> LogisticRegression:
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000,
                               random_state=RANDOM_STATE)
    return model.fit(X, y)


def fit_logit(X: pd.DataFrame, y) -> tuple[LogisticRegression, list[str], pd.DataFrame]:
    """Unpenalised logistic regression with a sign check.

    With ``WOE = ln(good/bad)`` and ``y = bad``, every coefficient must be negative. While
    any is >= 0, drop the most positive one and refit. Returns the model, the final feature
    list and a coefficient table (feature, coefficient; the intercept is the last row).
    """
    features = list(X.columns)
    while features:
        model = _fit(X[features], y)
        coefs = model.coef_[0]
        if (coefs < 0).all():
            table = pd.DataFrame({"feature": features + ["intercept"],
                                  "coefficient": np.append(coefs, model.intercept_[0])})
            return model, features, table
        worst = features[int(np.argmax(coefs))]
        logger.info("dropping %s: non-negative coefficient %.4f", worst, coefs.max())
        features.remove(worst)
    raise ValueError("no feature has a negative coefficient; check the WOE convention")


def predict_pd(model: LogisticRegression, woe: pd.DataFrame) -> np.ndarray:
    """Probability of bad for each row of a WOE frame."""
    return model.predict_proba(woe[list(model.feature_names_in_)])[:, 1]
