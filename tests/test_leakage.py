"""Guards against post-origination leakage, benchmark and fair-lending columns."""
import pandas as pd
import pytest

from scorecard import features
from scorecard.config import (
    BENCHMARK_COLS,
    CANDIDATE_FEATURES,
    EXCLUDED_FAIR_LENDING,
    LEAKAGE_COLUMNS,
    LOSS_CALIBRATION_COLS,
)
from scorecard.features import feature_columns, forbidden_columns

FORBIDDEN_EXAMPLES = ["recoveries", "total_pymnt", "last_pymnt_d", "last_fico_range_high",
                      "grade", "sub_grade", "int_rate", "zip_code", "funded_amnt"]


def test_whitelist_disjoint_from_leakage():
    assert not set(CANDIDATE_FEATURES) & LEAKAGE_COLUMNS


@pytest.mark.parametrize("group", [BENCHMARK_COLS, EXCLUDED_FAIR_LENDING, LOSS_CALIBRATION_COLS])
def test_whitelist_disjoint_from_other_forbidden(group):
    assert not set(CANDIDATE_FEATURES) & set(group)


def test_engineered_feature_set_is_clean(prepared):
    cols = feature_columns(prepared)
    assert cols
    assert not set(cols) & forbidden_columns()


def test_injected_forbidden_columns_are_ignored(prepared):
    df = prepared.copy()
    for col in FORBIDDEN_EXAMPLES:
        if col not in df:
            df[col] = 1.0
    cols = feature_columns(df)
    assert not set(cols) & set(FORBIDDEN_EXAMPLES)
    assert set(cols) <= set(features.candidate_columns())


def test_polluted_whitelist_raises(monkeypatch):
    monkeypatch.setattr(features, "CANDIDATE_FEATURES", [*CANDIDATE_FEATURES, "recoveries"])
    with pytest.raises(AssertionError, match="recoveries"):
        feature_columns(pd.DataFrame({"recoveries": [1.0]}))
