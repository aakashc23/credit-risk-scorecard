"""Loading, type parsing, target construction and train/test/OOT split."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from scorecard.config import (
    BAD_STATUSES,
    BENCHMARK_COLS,
    CANDIDATE_FEATURES,
    CATEGORICAL_FEATURES,
    DEV_END,
    DEV_START,
    GOOD_STATUSES,
    ID_COLS,
    LOSS_CALIBRATION_COLS,
    OOT_END,
    OOT_START,
    RANDOM_STATE,
    RAW_PATH,
    TARGET,
    TERM_MONTHS,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

CHUNK_ROWS = 200_000
DATE_FORMAT = "%b-%Y"  # raw dates look like "Dec-2015"
# Raw string columns that are parsed in features.engineer, not here.
STRING_COLS = ["emp_length", "earliest_cr_line"]
RARE_HOME_OWNERSHIP = {"ANY", "NONE"}


def raw_columns() -> list[str]:
    """Every raw column the pipeline wants (and nothing else)."""
    cols = ID_COLS + CANDIDATE_FEATURES + BENCHMARK_COLS + LOSS_CALIBRATION_COLS
    return list(dict.fromkeys(cols))


def _resolve_columns(available: list[str]) -> list[str]:
    """Wanted columns that exist in the file. The key ID_COLS are required; others are
    optional and skipped with a warning."""
    missing_key = [c for c in ID_COLS if c not in available]
    if missing_key:
        raise ValueError(f"raw file lacks required columns: {missing_key}")
    skipped = [c for c in raw_columns() if c not in available]
    if skipped:
        logger.warning("optional columns missing from raw file, skipped: %s", skipped)
    return [c for c in raw_columns() if c in available]


def _filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep 36-month loans issued inside the development + OOT windows; drop junk rows."""
    chunk = chunk.dropna(subset=["id", "issue_d", "term"]).copy()
    chunk["id"] = chunk["id"].astype(str)
    if not pd.api.types.is_datetime64_any_dtype(chunk["issue_d"]):
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"].astype(str).str.strip(),
                                          format=DATE_FORMAT, errors="coerce")
    chunk["term"] = pd.to_numeric(chunk["term"].astype(str).str.extract(r"(\d+)")[0],
                                  errors="coerce")
    in_window = chunk["issue_d"].between(pd.Timestamp(DEV_START), pd.Timestamp(OOT_END))
    return chunk[(chunk["term"] == TERM_MONTHS) & in_window]


def _iter_csv(path: Path, sample: int | None):
    available = list(pd.read_csv(path, nrows=0).columns)
    yield from pd.read_csv(path, usecols=_resolve_columns(available), dtype={"id": str},
                           low_memory=False, chunksize=CHUNK_ROWS, nrows=sample)


def _iter_parquet(path: Path, sample: int | None):
    pf = pq.ParquetFile(path)
    columns = _resolve_columns(pf.schema_arrow.names)
    remaining = sample
    for batch in pf.iter_batches(batch_size=CHUNK_ROWS, columns=columns):
        chunk = batch.to_pandas()
        if remaining is not None:
            chunk, remaining = chunk.head(remaining), remaining - len(chunk)
        yield chunk
        if remaining is not None and remaining <= 0:
            return


def load_raw(path: Path | str = RAW_PATH, sample: int | None = None) -> pd.DataFrame:
    """Read the raw Lending Club file in chunks, keeping only the modelling sample.

    Dispatches on suffix: ``.parquet`` (row-group batches) or CSV / ``.csv.gz`` (chunked
    ``read_csv``). ``sample`` reads only the first N raw rows (fast development runs).
    ``issue_d`` is returned as datetime and ``term`` as int; everything else is still raw
    (see ``parse_types``). Raw rows read are stored in ``df.attrs["n_rows_read"]``.
    """
    path = Path(path)
    chunks = _iter_parquet(path, sample) if path.suffix == ".parquet" else _iter_csv(path, sample)
    parts, n_read = [], 0
    for chunk in chunks:
        n_read += len(chunk)
        parts.append(_filter_chunk(chunk))
        logger.info("read %s raw rows, kept %s so far", f"{n_read:,}",
                    f"{sum(map(len, parts)):,}")
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=raw_columns())
    df.attrs["n_rows_read"] = n_read
    return df


def _to_float(s: pd.Series) -> pd.Series:
    """Numeric parse that tolerates strings such as "45.3%" or " 13.99 "."""
    if s.dtype == object or pd.api.types.is_string_dtype(s):  # object (pandas 2) or str (pandas 3)
        s = s.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns (incl. percentage strings) and tidy categorical labels."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["issue_d"]):
        df["issue_d"] = pd.to_datetime(df["issue_d"], format=DATE_FORMAT, errors="coerce")
    text_cols = set(CATEGORICAL_FEATURES) | set(STRING_COLS) | set(ID_COLS) | {"grade", "sub_grade"}
    for col in raw_columns():
        if col in df.columns and col not in text_cols:
            df[col] = _to_float(df[col])
    for col in [*CATEGORICAL_FEATURES, "loan_status"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().astype(object)
    if "home_ownership" in df.columns:
        df["home_ownership"] = df["home_ownership"].replace(
            dict.fromkeys(RARE_HOME_OWNERSHIP, "OTHER"))
    return df


def make_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add ``bad`` (1 = charged off / default, 0 = fully paid); drop indeterminate loans."""
    status = df["loan_status"]
    bad = pd.Series(pd.NA, index=df.index, dtype="Int8")
    bad[status.isin(BAD_STATUSES)] = 1
    bad[status.isin(GOOD_STATUSES)] = 0
    known = bad.notna()
    out = df[known].copy()
    out[TARGET] = bad[known].astype("int8")
    stats = {
        "n_total": len(df),
        "status_counts": {str(k): int(v) for k, v in status.value_counts().items()},
        "n_bad": int(out[TARGET].sum()),
        "n_good": int((out[TARGET] == 0).sum()),
        "n_indeterminate": int((~known).sum()),
        "bad_rate": float(out[TARGET].mean()) if len(out) else float("nan"),
    }
    logger.info("target: %d bad, %d good, %d indeterminate excluded",
                stats["n_bad"], stats["n_good"], stats["n_indeterminate"])
    return out, stats


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``split``: OOT by issue date; development rows split train/test, stratified on bad."""
    issue = df["issue_d"]
    is_oot = issue.between(pd.Timestamp(OOT_START), pd.Timestamp(OOT_END))
    is_dev = issue.between(pd.Timestamp(DEV_START), pd.Timestamp(DEV_END))
    out = df[is_oot | is_dev].copy()
    out["split"] = "oot"
    dev = out.index[is_dev[out.index]]
    if len(dev):
        train_idx, _ = train_test_split(dev, test_size=TEST_SIZE, stratify=out.loc[dev, TARGET],
                                        random_state=RANDOM_STATE)
        out.loc[dev, "split"] = "test"
        out.loc[train_idx, "split"] = "train"
    return out
