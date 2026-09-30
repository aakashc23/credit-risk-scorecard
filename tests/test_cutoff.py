import numpy as np
import pandas as pd
import pytest

from scorecard.config import SCORE_MAX, SCORE_MIN
from scorecard.cutoff import COLUMNS, cutoff_summary, cutoff_table


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 3000
    score = np.round(rng.normal(640, 60, n)).clip(SCORE_MIN, SCORE_MAX)
    p = 1 / (1 + np.exp((score - 650) / 45))
    bad = (rng.random(n) < p).astype(int)
    ead = rng.uniform(2000, 15000, n)
    return pd.DataFrame({"score": score, "pd": p, "ead": ead, "el": p * 0.8 * ead, "bad": bad,
                         "realised_loss": bad * ead * 0.7})


def test_columns_and_grid(scored):
    table = cutoff_table(scored)
    assert list(table.columns) == COLUMNS
    assert table["cutoff"].iloc[0] == SCORE_MIN
    assert table["cutoff"].iloc[-1] == SCORE_MAX
    assert (table["cutoff"].diff().dropna() == 5).all()


def test_approval_rate_non_increasing(scored):
    assert cutoff_table(scored)["approval_rate"].is_monotonic_decreasing


def test_min_cutoff_approves_everyone(scored):
    row = cutoff_table(scored).iloc[0]
    assert row["approval_rate"] == 1.0
    assert row["n_rejected"] == 0


def test_counts_add_up(scored):
    table = cutoff_table(scored)
    assert (table["n_approved"] + table["n_rejected"] == table["n_total"]).all()
    assert np.allclose(table["approval_rate"] + table["rejection_rate"], 1.0)


def test_nothing_approved_above_max_score(scored):
    row = cutoff_table(scored.assign(score=scored["score"].clip(upper=700))).iloc[-1]
    assert row["n_approved"] == 0
    assert np.isnan(row["expected_bad_rate"])
    assert np.isnan(row["observed_bad_rate"])
    assert row["approved_exposure"] == 0


@pytest.mark.parametrize("cutoff", [300, 555, 640, 700, 851])
def test_vectorised_matches_naive(scored, cutoff):
    approved = scored[scored["score"] >= cutoff]
    summary = cutoff_summary(scored, cutoff)
    assert summary["n_approved"] == len(approved)
    assert summary["approval_rate"] == pytest.approx(len(approved) / len(scored))
    assert summary["approved_exposure"] == pytest.approx(approved["ead"].sum())
    assert summary["expected_loss"] == pytest.approx(approved["el"].sum())
    assert summary["realised_loss"] == pytest.approx(approved["realised_loss"].sum())
    if len(approved):
        assert summary["expected_bad_rate"] == pytest.approx(approved["pd"].mean())
        assert summary["observed_bad_rate"] == pytest.approx(approved["bad"].mean())
        assert summary["expected_loss_rate"] == pytest.approx(
            approved["el"].sum() / approved["ead"].sum())
