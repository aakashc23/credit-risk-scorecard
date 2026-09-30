"""Feature engineering and the leakage-safe list of model candidate columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scorecard.config import (
    BENCHMARK_COLS,
    CANDIDATE_FEATURES,
    EXCLUDED_FAIR_LENDING,
    LEAKAGE_COLUMNS,
    LOSS_CALIBRATION_COLS,
    MAX_MISSING_SHARE,
)

DATE_FORMAT = "%b-%Y"
# Raw fields replaced by engineered ones (dropped after engineering).
RAW_REPLACED = ["fico_range_low", "fico_range_high", "earliest_cr_line", "emp_length"]
ENGINEERED = [
    "credit_history_months",
    "fico_mid",
    "emp_length_years",
    "loan_to_income",
    "revol_bal_to_income",
]


def forbidden_columns() -> set[str]:
    """Columns that may never be model inputs."""
    return (set(LEAKAGE_COLUMNS) | set(BENCHMARK_COLS) | set(EXCLUDED_FAIR_LENDING)
            | set(LOSS_CALIBRATION_COLS))


def parse_emp_length(s: pd.Series) -> pd.Series:
    """"< 1 year" -> 0, "10+ years" -> 10, "n/a"/missing -> NaN."""
    text = s.astype("string").str.strip()
    years = pd.to_numeric(text.str.extract(r"(\d+)")[0], errors="coerce").astype("float64")
    return years.mask(text.str.startswith("<", na=False), 0.0)


def _safe_ratio(num: pd.Series, income: pd.Series) -> pd.Series:
    """num / income, NaN where income <= 0 or missing."""
    return num / income.where(income > 0)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features and drop the raw fields they replace.

    A feature is skipped (with no error) when its raw inputs are absent from ``df``.
    """
    out = df.copy()

    def has(*cols: str) -> bool:
        return all(c in out.columns for c in cols)

    if has("earliest_cr_line", "issue_d"):
        earliest = pd.to_datetime(out["earliest_cr_line"], format=DATE_FORMAT, errors="coerce")
        issue = pd.to_datetime(out["issue_d"])
        out["credit_history_months"] = ((issue.dt.year - earliest.dt.year) * 12
                                        + (issue.dt.month - earliest.dt.month)).astype("float64")
    if has("fico_range_low", "fico_range_high"):
        out["fico_mid"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    if has("emp_length"):
        out["emp_length_years"] = parse_emp_length(out["emp_length"])
    if has("loan_amnt", "annual_inc"):
        out["loan_to_income"] = _safe_ratio(out["loan_amnt"], out["annual_inc"])
    if has("revol_bal", "annual_inc"):
        out["revol_bal_to_income"] = _safe_ratio(out["revol_bal"], out["annual_inc"])
    made = [c for c in ENGINEERED if c in out.columns]
    out[made] = out[made].replace([np.inf, -np.inf], np.nan)
    return out.drop(columns=[c for c in RAW_REPLACED if c in out.columns])


def candidate_columns() -> list[str]:
    """Model candidates after engineering, derived from the whitelist only."""
    kept = [c for c in CANDIDATE_FEATURES if c not in RAW_REPLACED]
    return kept + [c for c in ENGINEERED if c not in kept]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-candidate columns present in ``df``. Raises if any is a forbidden column."""
    cols = candidate_columns()
    clash = set(cols) & forbidden_columns()
    if clash:
        raise AssertionError(f"leakage/forbidden columns in candidate features: {sorted(clash)}")
    return [c for c in cols if c in df.columns]


def high_missing_features(train: pd.DataFrame, columns: list[str],
                          max_share: float = MAX_MISSING_SHARE) -> list[str]:
    """Columns whose missing share in ``train`` exceeds ``max_share`` (to be dropped)."""
    share = train[columns].isna().mean()
    return [c for c in columns if share[c] > max_share]
