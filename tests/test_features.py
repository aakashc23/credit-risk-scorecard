import numpy as np
import pandas as pd

from scorecard.features import (
    ENGINEERED,
    RAW_REPLACED,
    engineer,
    feature_columns,
    high_missing_features,
    parse_emp_length,
)


def _row(**kw):
    base = {"issue_d": pd.Timestamp("2014-03-01"), "earliest_cr_line": "Mar-2004",
            "fico_range_low": 680.0, "fico_range_high": 684.0, "emp_length": "3 years",
            "loan_amnt": 10000.0, "annual_inc": 50000.0, "revol_bal": 5000.0}
    return pd.DataFrame([{**base, **kw}])


def test_engineered_values():
    out = engineer(_row()).iloc[0]
    assert out["credit_history_months"] == 120
    assert out["fico_mid"] == 682
    assert out["emp_length_years"] == 3
    assert out["loan_to_income"] == 0.2
    assert out["revol_bal_to_income"] == 0.1


def test_raw_fields_dropped_and_new_present():
    out = engineer(_row())
    assert not set(RAW_REPLACED) & set(out.columns)
    assert set(ENGINEERED) <= set(out.columns)


def test_non_positive_income_gives_nan_ratio():
    frame = pd.concat([_row(annual_inc=0.0), _row(annual_inc=-5.0), _row(annual_inc=np.nan)])
    out = engineer(frame)
    assert out[["loan_to_income", "revol_bal_to_income"]].isna().all().all()


def test_emp_length_parsing():
    parsed = parse_emp_length(pd.Series(["< 1 year", "1 year", "10+ years", "n/a", None, "7 years"]))
    assert parsed.iloc[:3].tolist() == [0, 1, 10]
    assert parsed.iloc[3:5].isna().all()
    assert parsed.iloc[5] == 7


def test_missing_raw_inputs_skip_feature():
    out = engineer(_row().drop(columns=["emp_length"]))
    assert "emp_length_years" not in out.columns
    assert "fico_mid" in out.columns


def test_feature_columns_on_synthetic(prepared):
    cols = feature_columns(prepared)
    assert {"fico_mid", "credit_history_months", "purpose"} <= set(cols)
    assert not set(RAW_REPLACED) & set(cols)


def test_high_missing_features(prepared):
    train = prepared[prepared["split"] == "train"]
    dropped = high_missing_features(train, feature_columns(prepared), max_share=0.30)
    # mths_since_* is mostly blank (= "never happened") but is exempt: missing is informative.
    assert "mths_since_last_record" not in dropped
    assert "fico_mid" not in dropped
    assert high_missing_features(train.assign(junk=float("nan")), ["junk"]) == ["junk"]
