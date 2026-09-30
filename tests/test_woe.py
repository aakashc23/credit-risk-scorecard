import json

import numpy as np
import pandas as pd
import pytest

from scorecard.config import MAX_COARSE_BINS, MIN_BIN_SHARE, WOE_SMOOTHING
from scorecard.woe import WoeBinner, fit_bins, woe_iv


def _monotone_data(n=5000, seed=1, direction=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1.5 + direction * 1.2 * x)))
    return pd.Series(x), (rng.random(n) < p).astype(int)


def _cat_data(n=5000, seed=6):
    rng = np.random.default_rng(seed)
    cats = rng.choice(["a", "b", "c", "rare1", "rare2"], n, p=[0.4, 0.3, 0.295, 0.0025, 0.0025])
    base = {"a": 0.05, "b": 0.15, "c": 0.3, "rare1": 0.2, "rare2": 0.2}
    y = (rng.random(n) < pd.Series(cats).map(base).to_numpy()).astype(int)
    return pd.DataFrame({"cat": cats}), y


def test_iv_hand_computed():
    # two bins: 100 loans / 10 bad and 100 loans / 40 bad
    woe, contrib = woe_iv(np.array([100, 100]), np.array([10, 40]))
    good = np.array([90, 60]) + WOE_SMOOTHING
    bad = np.array([10, 40]) + WOE_SMOOTHING
    pct_good, pct_bad = good / good.sum(), bad / bad.sum()
    assert np.allclose(woe, np.log(pct_good / pct_bad))
    assert contrib.sum() == pytest.approx(((pct_good - pct_bad) * np.log(pct_good / pct_bad)).sum())
    assert woe[0] > 0 > woe[1]  # the low-risk bin has positive WOE


def test_iv_value_on_tiny_example():
    # 4 obvious bins: check IV against a direct computation with smoothing 0.5
    x = pd.Series([1.0] * 50 + [2.0] * 50 + [3.0] * 50 + [4.0] * 50)
    y = np.array([0] * 45 + [1] * 5 + [0] * 40 + [1] * 10 + [0] * 30 + [1] * 20 + [0] * 20 + [1] * 30)
    spec = fit_bins(x, y)
    bads = np.array([5, 10, 20, 30]) + 0.5
    goods = np.array([45, 40, 30, 20]) + 0.5
    expected = ((goods / goods.sum() - bads / bads.sum())
                * np.log((goods / goods.sum()) / (bads / bads.sum()))).sum()
    assert len(spec["bins"]) == 4
    assert spec["iv"] == pytest.approx(expected)


@pytest.mark.parametrize("direction", [1, -1])
def test_numeric_bins_monotone_and_bounded(direction):
    x, y = _monotone_data(direction=direction)
    spec = fit_bins(x, y)
    rates = [b["bad_rate"] for b in spec["bins"]]
    assert 1 < len(rates) <= MAX_COARSE_BINS
    assert np.all(np.diff(rates) * direction >= 0)
    assert min(b["n"] for b in spec["bins"]) >= MIN_BIN_SHARE * len(x)
    assert spec["iv"] > 0.1


def test_non_monotone_relationship_is_merged_to_monotone():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=6000))
    y = (rng.random(6000) < 1 / (1 + np.exp(-(-1.5 + 0.9 * x**2)))).astype(int)  # U shape
    diffs = np.diff([b["bad_rate"] for b in fit_bins(x, y)["bins"]])
    assert np.all(diffs >= 0) or np.all(diffs <= 0)


def test_heavy_ties_and_constant_feature():
    rng = np.random.default_rng(4)
    zeros = pd.Series(np.where(rng.random(3000) < 0.8, 0.0, rng.integers(1, 6, 3000)))
    y = (rng.random(3000) < 0.1 + 0.05 * (zeros > 0)).astype(int)
    assert len(fit_bins(zeros, y)["bins"]) >= 2
    const = fit_bins(pd.Series(np.ones(500)), (rng.random(500) < 0.2).astype(int))
    assert len(const["bins"]) == 1
    assert const["iv"] == pytest.approx(0.0)


def test_missing_gets_own_bin_and_transform_uses_it():
    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(size=4000))
    x[rng.random(4000) < 0.2] = np.nan
    y = np.where(x.isna(), rng.random(4000) < 0.5, rng.random(4000) < 0.1).astype(int)
    frame = pd.DataFrame({"x": x})
    binner = WoeBinner().fit(frame, y, ["x"])
    table = binner.bin_table()
    miss = table[table["bin"] == "Missing"]
    assert len(miss) == 1
    assert miss["n"].iloc[0] == x.isna().sum()
    assert (binner.transform(frame)["x"][x.isna()] == miss["woe"].iloc[0]).all()
    # missing at transform time but no Missing bin at fit time -> neutral WOE
    clean = WoeBinner().fit(pd.DataFrame({"x": x.fillna(0)}), y, ["x"])
    assert clean.transform(pd.DataFrame({"x": [np.nan]}))["x"].iloc[0] == 0.0


def test_categorical_pools_rare_and_handles_unseen():
    df, y = _cat_data()
    binner = WoeBinner().fit(df, y, ["cat"])
    cats = {c for b in binner.bins_["cat"]["bins"] for c in b["categories"]}
    assert "Other" in cats
    assert not {"rare1", "rare2"} & cats
    woe_rare = binner.transform(pd.DataFrame({"cat": ["rare1"]}))["cat"].iloc[0]
    unseen = binner.transform(pd.DataFrame({"cat": ["never_seen", None]}))["cat"]
    assert (unseen == woe_rare).all()
    assert binner.transform(df)["cat"].nunique() <= MAX_COARSE_BINS


def test_unseen_category_without_other_bin_gets_zero():
    df = pd.DataFrame({"cat": ["a", "b"] * 500})
    binner = WoeBinner().fit(df, np.array([0, 1] * 500), ["cat"])
    assert binner.transform(pd.DataFrame({"cat": ["zzz"]}))["cat"].iloc[0] == 0.0


def test_iv_table_and_bin_table_consistent():
    x, y = _monotone_data()
    cats, _ = _cat_data()
    frame = pd.DataFrame({"x": x, "cat": cats["cat"]})
    binner = WoeBinner().fit(frame, y, ["x", "cat"])
    iv, bins = binner.iv_table(), binner.bin_table()
    assert iv["iv"].is_monotonic_decreasing
    by_feature = bins.groupby("feature")["iv_contrib"].sum()
    for feat, total in by_feature.items():
        assert total == pytest.approx(iv.set_index("feature").loc[feat, "iv"])
    assert {"feature", "bin", "n", "bads", "bad_rate", "woe", "iv_contrib"} <= set(bins.columns)


def test_json_round_trip(tmp_path):
    x, y = _monotone_data()
    cats, _ = _cat_data()
    frame = pd.DataFrame({"x": x.where(x.index % 10 != 0), "cat": cats["cat"]})
    binner = WoeBinner().fit(frame, y, ["x", "cat"])
    path = tmp_path / "bins.json"
    binner.to_json(path)
    json.loads(path.read_text())  # valid JSON
    loaded = WoeBinner.from_json(path)
    pd.testing.assert_frame_equal(binner.transform(frame), loaded.transform(frame))
    pd.testing.assert_frame_equal(binner.bin_table(), loaded.bin_table())
