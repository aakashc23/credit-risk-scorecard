import numpy as np
import pandas as pd
import pytest

from scorecard.validation import (
    auc,
    calibration_table,
    decile_table,
    gini,
    grade_to_ordinal,
    ks,
    psi,
)

Y = np.array([0, 0, 1, 0, 1, 1])
RISK = np.array([0.1, 0.4, 0.35, 0.2, 0.8, 0.7])


def test_auc_hand_checked():
    # positives 0.35, 0.8, 0.7 vs negatives 0.1, 0.4, 0.2: 8 of 9 pairs ordered correctly
    assert auc(Y, RISK) == pytest.approx(8 / 9)
    assert gini(Y, RISK) == pytest.approx(2 * 8 / 9 - 1)


def test_ks_hand_checked():
    # sorted by risk desc: 0.8(b) 0.7(b) 0.4(g) 0.35(b) 0.2(g) 0.1(g); best gap after 0.4 step
    assert ks(Y, RISK) == pytest.approx(2 / 3)


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    risk = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks(y, risk) == pytest.approx(1.0)
    assert auc(y, risk) == pytest.approx(1.0)
    assert gini(y, risk) == pytest.approx(1.0)


def test_random_scores_have_no_separation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20000)
    assert abs(auc(y, rng.random(20000)) - 0.5) < 0.02


def test_psi_identical_is_zero_and_shift_is_positive():
    rng = np.random.default_rng(1)
    base = rng.normal(600, 50, 10000)
    assert psi(base, base) == pytest.approx(0.0, abs=1e-12)
    assert psi(base, base + 40) > 0.25


def test_decile_table():
    rng = np.random.default_rng(2)
    n = 1000
    score = rng.normal(650, 50, n)
    p = 1 / (1 + np.exp((score - 650) / 40))
    y = (rng.random(n) < p).astype(int)
    table = decile_table(y, score, p)
    assert len(table) == 10
    assert table["n"].sum() == n
    assert table["bads"].sum() == y.sum()
    assert table["decile"].tolist() == list(range(1, 11))
    assert table["max_score"].iloc[0] <= table["min_score"].iloc[1]  # worst first
    assert table["bad_rate"].iloc[0] > table["bad_rate"].iloc[-1]
    assert table["cum_bad_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["cum_good"].iloc[-1] == pytest.approx(1.0)
    assert table["cum_bad_capture"].is_monotonic_increasing


def test_calibration_table_well_calibrated():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.5, 50000)
    y = (rng.random(50000) < p).astype(int)
    table = calibration_table(y, p)
    assert len(table) == 10
    assert table["mean_pd"].is_monotonic_increasing
    assert (table["abs_error"] < 0.03).all()


def test_grade_to_ordinal():
    ordinal = grade_to_ordinal(pd.Series(["A1", "A5", "B1", "G5", "x", None]))
    assert ordinal.iloc[:4].tolist() == [1, 5, 6, 35]
    assert ordinal.iloc[4:].isna().all()
