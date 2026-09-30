import numpy as np
import pandas as pd
import pytest

from scorecard.config import (
    BAD_STATUSES,
    DEV_END,
    DEV_START,
    GOOD_STATUSES,
    OOT_END,
    OOT_START,
    TEST_SIZE,
)
from scorecard.data import assign_split, load_raw, make_target, parse_types


@pytest.fixture(scope="module")
def loaded(raw_csv):
    return load_raw(raw_csv)


def test_term_and_window_filter(loaded, raw_frame):
    assert (loaded["term"] == 36).all()
    assert loaded["issue_d"].between(pd.Timestamp(DEV_START), pd.Timestamp(OOT_END)).all()
    year = raw_frame["issue_d"].str[-4:]
    expected = raw_frame["term"].eq(" 36 months") & year.between(DEV_START[:4], OOT_END[:4])
    assert len(loaded) == expected.sum()
    assert loaded.attrs["n_rows_read"] == len(raw_frame)


def test_footer_rows_dropped(loaded):
    assert loaded["id"].notna().all()
    assert not loaded["id"].str.startswith("Total amount").any()


def test_only_needed_columns_read(loaded):
    assert not {"total_pymnt", "last_pymnt_d", "zip_code", "addr_state"} & set(loaded.columns)


def test_sample_reads_first_rows_only(raw_csv):
    small = load_raw(raw_csv, sample=1000)
    assert small.attrs["n_rows_read"] == 1000
    assert 0 < len(small) < 1000


def test_parquet_input_matches_csv(tmp_path, raw_frame, raw_csv):
    """Parquet extract: numeric id, stripped term strings, optional columns missing."""
    extract = raw_frame.iloc[:-2].copy()  # the extract has no footer rows
    extract["id"] = extract["id"].astype("int64")
    extract["term"] = extract["term"].str.strip()
    extract["revol_util"] = extract["revol_util"].str.rstrip("%").astype(float)
    path = tmp_path / "extract.parquet"
    extract.drop(columns=["tax_liens", "mort_acc"]).to_parquet(path, index=False)

    from_parquet = load_raw(path)
    from_csv = load_raw(raw_csv)
    assert len(from_parquet) == len(from_csv)
    assert "tax_liens" not in from_parquet.columns  # missing optional column skipped
    assert (from_parquet["term"] == 36).all()
    assert from_parquet.attrs["n_rows_read"] == len(extract)
    assert load_raw(path, sample=500).attrs["n_rows_read"] == 500


def test_parse_types_handles_percent_strings_and_rare_home(loaded):
    df = parse_types(loaded)
    assert df["revol_util"].dtype == "float64"
    assert df["revol_util"].between(0, 130).all()
    assert not df["home_ownership"].isin(["ANY", "NONE"]).any()
    numeric_input = parse_types(loaded.assign(revol_util=df["revol_util"]))
    assert np.allclose(numeric_input["revol_util"], df["revol_util"], equal_nan=True)


def test_target_mapping_and_indeterminate_excluded(loaded):
    parsed = parse_types(loaded)
    df, stats = make_target(parsed)
    status = df["loan_status"]
    assert (df.loc[status.isin(BAD_STATUSES), "bad"] == 1).all()
    assert (df.loc[status.isin(GOOD_STATUSES), "bad"] == 0).all()
    assert set(status) <= BAD_STATUSES | GOOD_STATUSES
    n_indet = (~parsed["loan_status"].isin(BAD_STATUSES | GOOD_STATUSES)).sum()
    assert stats["n_indeterminate"] == n_indet
    assert n_indet > 0
    assert stats["n_bad"] + stats["n_good"] + n_indet == len(parsed) == stats["n_total"]
    assert sum(stats["status_counts"].values()) == len(parsed)


def test_split_proportions_and_stratification(prepared):
    dev = prepared[prepared["split"].isin(["train", "test"])]
    assert abs((dev["split"] == "test").mean() - TEST_SIZE) < 0.01
    rates = dev.groupby("split")["bad"].mean()
    assert abs(rates["train"] - rates["test"]) < 0.01


def test_oot_is_purely_by_date(prepared):
    oot = prepared["split"] == "oot"
    in_oot = prepared["issue_d"].between(pd.Timestamp(OOT_START), pd.Timestamp(OOT_END))
    in_dev = prepared["issue_d"].between(pd.Timestamp(DEV_START), pd.Timestamp(DEV_END))
    assert oot.equals(in_oot)
    assert in_dev[~oot].all()


def test_assign_split_is_deterministic(prepared):
    again = assign_split(prepared.drop(columns="split"))
    assert again["split"].equals(prepared["split"])
