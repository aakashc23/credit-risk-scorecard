"""Shared fixtures: a synthetic raw file that mimics the Lending Club schema."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scorecard.config import (
    BENCHMARK_COLS,
    CANDIDATE_FEATURES,
    ID_COLS,
    LOSS_CALIBRATION_COLS,
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "small_business", "other",
            "major_purchase", "moving"]


def _fmt_date(year: np.ndarray, month: np.ndarray) -> np.ndarray:
    return np.array([f"{MONTHS[m - 1]}-{y}" for y, m in zip(year, month, strict=True)])


def make_raw_frame(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    """Raw-looking Lending Club frame. Default risk depends on fico, dti, inq, purpose."""
    rng = np.random.default_rng(seed)
    year = rng.integers(2011, 2017, n)
    month = rng.integers(1, 13, n)
    fico = np.clip(rng.normal(695, 30, n), 620, 830).round(-1) // 5 * 5
    dti = np.clip(rng.gamma(4.0, 4.5, n), 0, 45).round(2)
    inq = rng.poisson(0.8, n)
    income = rng.lognormal(11.0, 0.5, n).round(0)
    loan = (rng.integers(10, 400, n) * 100).astype(float)
    purpose = rng.choice(PURPOSES, n, p=[0.45, 0.2, 0.1, 0.05, 0.1, 0.05, 0.05])
    home = rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], n, p=[0.45, 0.4, 0.149, 0.001])
    revol_util = np.clip(rng.normal(50, 25, n), 0, 120).round(1)
    logit = (-1.7 - 0.025 * (fico - 695) + 0.04 * (dti - 18) + 0.3 * inq
             + 0.5 * (purpose == "small_business") + 0.01 * (revol_util - 50)
             + 0.000004 * (loan - 20000))
    is_bad = rng.random(n) < 1 / (1 + np.exp(-logit))

    status = np.where(is_bad, "Charged Off", "Fully Paid").astype(object)
    status[is_bad & (rng.random(n) < 0.05)] = "Default"
    other = rng.random(n)
    status[~is_bad & (other < 0.03)] = "Current"
    status[~is_bad & (other > 0.98)] = "Late (31-120 days)"
    status[is_bad & (year <= 2013) & (rng.random(n) < 0.03)] = \
        "Does not meet the credit policy. Status:Charged Off"
    bad_final = np.isin(status, ["Charged Off", "Default",
                                 "Does not meet the credit policy. Status:Charged Off"])

    paid = np.where(bad_final, rng.uniform(0.1, 0.7, n), 1.0)
    recov = np.where(bad_final, loan * rng.uniform(0, 0.08, n), 0.0).round(2)

    sub_letter = np.clip(((740 - fico) / 25 + rng.normal(0, 1, n)).round().astype(int), 0, 6)
    sub_grade = np.array([f"{'ABCDEFG'[g]}{rng.integers(1, 6)}" for g in sub_letter])

    def maybe_missing(values: np.ndarray, share: float) -> np.ndarray:
        out = values.astype(float)
        out[rng.random(n) < share] = np.nan
        return out

    emp_choices = ["< 1 year", "1 year", "2 years", "5 years", "9 years", "10+ years", "n/a"]
    frame = pd.DataFrame({
        "id": np.arange(1000, 1000 + n).astype(str),
        "issue_d": _fmt_date(year, month),
        "term": rng.choice([" 36 months", " 60 months"], n, p=[0.75, 0.25]),
        "loan_status": status,
        "loan_amnt": loan,
        "emp_length": rng.choice(emp_choices, n),
        "home_ownership": home,
        "annual_inc": income,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "purpose": purpose,
        "application_type": rng.choice(["Individual", "Joint App"], n, p=[0.97, 0.03]),
        "dti": maybe_missing(dti, 0.01),
        "delinq_2yrs": rng.poisson(0.3, n),
        "earliest_cr_line": _fmt_date(year - rng.integers(4, 25, n), rng.integers(1, 13, n)),
        "fico_range_low": fico,
        "fico_range_high": fico + 4,
        "inq_last_6mths": inq,
        "mths_since_last_delinq": maybe_missing(rng.integers(1, 100, n), 0.5),
        "mths_since_last_record": maybe_missing(rng.integers(1, 100, n), 0.85),
        "open_acc": rng.poisson(10, n),
        "pub_rec": rng.poisson(0.2, n),
        "revol_bal": rng.lognormal(9, 1, n).round(0),
        "revol_util": [f"{v}%" for v in revol_util],  # percentage strings, as in some exports
        "total_acc": rng.poisson(25, n),
        "mort_acc": maybe_missing(rng.poisson(1.2, n), 0.05),
        "pub_rec_bankruptcies": rng.poisson(0.1, n),
        "acc_open_past_24mths": rng.poisson(4, n),
        "bc_util": maybe_missing(np.clip(rng.normal(55, 28, n), 0, 100), 0.02),
        "num_actv_rev_tl": rng.poisson(6, n),
        "percent_bc_gt_75": maybe_missing(np.clip(rng.normal(45, 30, n), 0, 100), 0.02),
        "tot_cur_bal": rng.lognormal(10.5, 1.2, n).round(0),
        "total_rev_hi_lim": rng.lognormal(9.8, 0.9, n).round(0),
        "mo_sin_rcnt_tl": rng.poisson(8, n),
        "num_tl_op_past_12m": rng.poisson(2, n),
        "avg_cur_bal": rng.lognormal(8.5, 1.0, n).round(0),
        "mths_since_recent_inq": maybe_missing(rng.integers(0, 24, n), 0.1),
        "tax_liens": rng.poisson(0.02, n),
        "collections_12_mths_ex_med": rng.poisson(0.02, n),
        "grade": [g[0] for g in sub_grade],
        "sub_grade": sub_grade,
        "int_rate": (6 + 2.2 * sub_letter + rng.normal(0, 0.5, n)).round(2),
        "installment": (loan / 33).round(2),
        "funded_amnt": loan,
        "total_rec_prncp": (loan * paid).round(2),
        "recoveries": recov,
        "collection_recovery_fee": (recov * 0.1).round(2),
        # extra columns the loader must ignore (leakage / fair-lending fields)
        "total_pymnt": loan * 1.1,
        "last_pymnt_d": "Jan-2018",
        "last_fico_range_high": fico + 50,
        "zip_code": "123xx",
        "addr_state": "CA",
    })
    assert set(ID_COLS + CANDIDATE_FEATURES + BENCHMARK_COLS + LOSS_CALIBRATION_COLS) \
        <= set(frame.columns)
    return frame


def add_footer_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the junk rows the real file ends with (text in `id`, everything else NaN)."""
    footer = pd.DataFrame({"id": ["Total amount funded in policy code 1: 123",
                                  "Total amount funded in policy code 2: 456"]})
    return pd.concat([frame, footer], ignore_index=True)


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return add_footer_rows(make_raw_frame())


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory, raw_frame) -> Path:
    path = tmp_path_factory.mktemp("raw") / "synthetic_accepted.csv.gz"
    raw_frame.to_csv(path, index=False, compression="gzip")
    return path


@pytest.fixture(scope="session")
def prepared(raw_csv) -> pd.DataFrame:
    """Loaded, parsed, target-labelled, split and engineered synthetic data."""
    from scorecard.data import assign_split, load_raw, make_target, parse_types
    from scorecard.features import engineer

    df, _ = make_target(parse_types(load_raw(raw_csv)))
    return engineer(assign_split(df))
