import numpy as np
import pandas as pd
import pytest

from scorecard.config import BASE_ODDS, BASE_SCORE, PDO, SCORE_MAX, SCORE_MIN
from scorecard.model import fit_logit, predict_pd
from scorecard.scoring import (
    factor,
    neutral_points,
    offset,
    pd_to_score,
    points_score,
    score_to_pd,
    scorecard_points,
)
from scorecard.woe import WoeBinner


def test_score_at_base_odds_is_base_score():
    pd_at_base = 1 / (1 + BASE_ODDS)
    assert pd_to_score(pd_at_base) == pytest.approx(BASE_SCORE)


def test_doubling_odds_adds_pdo():
    p = 0.1
    odds = (1 - p) / p
    p_double = 1 / (1 + 2 * odds)
    assert pd_to_score(p_double) - pd_to_score(p) == pytest.approx(PDO)


def test_higher_pd_gives_lower_score():
    scores = pd_to_score(np.array([0.02, 0.1, 0.3]))
    assert np.all(np.diff(scores) < 0)


def test_clipping():
    assert pd_to_score(1e-9) == SCORE_MAX
    assert pd_to_score(0.999999) == SCORE_MIN
    assert pd_to_score(0.0) == SCORE_MAX
    assert pd_to_score(1.0) == SCORE_MIN
    assert pd_to_score(1e-9, clip=False) > SCORE_MAX


def test_score_to_pd_is_inverse():
    p = np.array([0.03, 0.09, 0.2, 0.4])
    assert np.allclose(score_to_pd(pd_to_score(p)), p)
    assert score_to_pd(BASE_SCORE) == pytest.approx(1 / (1 + BASE_ODDS))


def test_factor_and_offset():
    assert factor() == pytest.approx(PDO / np.log(2))
    assert offset() == pytest.approx(BASE_SCORE - factor() * np.log(BASE_ODDS))


def test_points_sum_to_score(prepared):
    train = prepared[prepared["split"] == "train"]
    feats = ["fico_mid", "dti", "inq_last_6mths", "purpose", "revol_util"]
    binner = WoeBinner().fit(train, train["bad"], feats)
    woe = binner.transform(train, feats)
    model, final, _ = fit_logit(woe, train["bad"])
    assert len(final) >= 3
    points = scorecard_points(binner, model)
    assert set(points["feature"]) == set(final)
    total = points_score(binner, points, train)
    unclipped = pd.Series(pd_to_score(predict_pd(model, woe), clip=False), index=train.index)
    assert (total - unclipped).abs().max() <= len(final)  # rounding tolerance
    assert abs((total - unclipped).mean()) < 1.0


def test_unbinned_values_score_neutral_points_not_zero():
    """A feature with no Missing/Other bin in train: NaN / unseen at scoring time gets the
    neutral points (WOE 0), so the points still add up to the model score."""
    rng = np.random.default_rng(3)
    n = 3000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n)  # every category is common, so there is no Other bin
    p = 1 / (1 + np.exp(-(-1.5 - 1.0 * x1 - 0.8 * x2 + 0.6 * (cat == "c"))))
    y = (rng.random(n) < p).astype(int)
    train = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
    feats = ["x1", "x2", "cat"]
    binner = WoeBinner().fit(train, y, feats)
    model, final, _ = fit_logit(binner.transform(train, feats), pd.Series(y))
    points = scorecard_points(binner, model)
    assert (points["bin"] == "Unbinned").sum() == len(final)  # one catch-all row per feature

    test = pd.DataFrame({"x1": [np.nan, 0.3], "x2": [np.nan, np.nan], "cat": ["zzz", None]})
    woe = binner.transform(test, final)
    assert (woe == 0).sum().sum() >= 3  # several values really are unbinned
    unclipped = pd.Series(pd_to_score(predict_pd(model, woe), clip=False))
    total = points_score(binner, points, test)
    assert (total - unclipped).abs().max() <= len(final)
    neutral = neutral_points(float(model.intercept_[0]), len(final))
    unbinned = points[points["bin"] == "Unbinned"]["points"]
    assert np.allclose(unbinned, np.round(neutral))
