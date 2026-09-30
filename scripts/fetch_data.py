"""Stream the Lending Club accepted-loans CSV from Hugging Face and build slim local extracts.

The 1.68 GB CSV is NEVER written to disk (not even a temp file): bytes are streamed over HTTP,
hashed on the fly and parsed in chunks by pandas. Outputs:

  data/raw/lc_accepted_2007_2018Q4_extract.parquet     slim extract (gitignored)
  data/interim/lc_all_columns_sample_5pct.parquet      ~5% sample, all columns, as strings
  artifacts/data_profile/*                             small full-file profile (committed)
  data/raw/SOURCE.md and docs/data_source.md           provenance note

Usage:  python scripts/fetch_data.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import urllib3

PROJECT_ROOT = Path(__file__).resolve().parents[1]

URL = (
    "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/"
    "accepted_2007_to_2018Q4.csv"
)
LICENSE = "CC0-1.0 (public domain dedication)"
EXPECTED_BYTES = 1_675_133_810
EXPECTED_SHA256 = "3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a"
CHUNK_ROWS = 200_000
ATTEMPTS = 3
SAMPLE_MOD = 20  # id % 20 == 0  ->  ~5% deterministic sample

EXTRACT_NAME = "lc_accepted_2007_2018Q4_extract.parquet"
SAMPLE_NAME = "lc_all_columns_sample_5pct.parquet"

# Post-origination / misc fields kept ONLY for the leakage demonstration.
LEAKAGE_DEMO_COLS = [
    "total_pymnt", "total_rec_int", "total_rec_late_fee", "last_pymnt_d", "last_pymnt_amnt",
    "last_fico_range_high", "last_fico_range_low", "out_prncp", "last_credit_pull_d",
    "debt_settlement_flag", "hardship_flag", "pymnt_plan", "funded_amnt_inv",
    "initial_list_status", "policy_code", "title", "emp_title",
]

# Columns stored as strings in the extract; id -> int64; every other kept column -> float64
# (coerced with pd.to_numeric; any value lost by coercion is counted in the profile).
STRING_COLS = {
    "issue_d", "term", "loan_status", "emp_length", "home_ownership", "verification_status",
    "purpose", "application_type", "earliest_cr_line", "grade", "sub_grade", "zip_code",
    "addr_state", "last_pymnt_d", "last_credit_pull_d", "debt_settlement_flag", "hardship_flag",
    "pymnt_plan", "initial_list_status", "title", "emp_title",
}
DATE_FMT = "%b-%Y"


def load_config(root: Path):
    """Load config.py directly from its file (avoids importing the whole package)."""
    spec = importlib.util.spec_from_file_location("_sc_config", root / "src" / "scorecard" / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wanted_columns(cfg) -> tuple[list[str], list[str]]:
    """(config columns, all wanted columns incl. leakage-demo extras), de-duplicated in order."""
    cfg_cols = (list(cfg.ID_COLS) + list(cfg.CANDIDATE_FEATURES) + list(cfg.BENCHMARK_COLS)
                + list(cfg.LOSS_CALIBRATION_COLS) + list(cfg.EXCLUDED_FAIR_LENDING))
    cfg_cols = list(dict.fromkeys(cfg_cols))
    allc = list(dict.fromkeys(cfg_cols + LEAKAGE_DEMO_COLS))
    return cfg_cols, allc


# --------------------------------------------------------------------------- streaming
class HashingReader(io.RawIOBase):
    """Raw binary stream over an HTTP response; hashes and counts every byte it hands out.

    If the connection drops mid-stream it transparently reconnects with an HTTP Range request
    from the current offset (the hash/counter just continue), so a drop does not restart the run.
    """

    MAX_RECONNECTS = 40

    def __init__(self, resp: requests.Response):
        self.resp = resp
        self.sha = hashlib.sha256()
        self.nbytes = 0
        self.reconnects = 0

    def readable(self) -> bool:
        return True

    def _reconnect(self):
        self.reconnects += 1
        if self.reconnects > self.MAX_RECONNECTS:
            raise ConnectionError("too many reconnects")
        time.sleep(min(30, 2 * self.reconnects))
        try:
            self.resp.close()
        except Exception:
            pass
        r = requests.get(URL, stream=True, allow_redirects=True, timeout=(30, 300),
                         headers={"Accept-Encoding": "identity", "Range": f"bytes={self.nbytes}-",
                                  "User-Agent": "credit-risk-scorecard-fetch/1.0"})
        if r.status_code != 206:
            raise ConnectionError(f"Range request returned HTTP {r.status_code}")
        self.resp = r
        print(f"  reconnected at byte {self.nbytes:,} (reconnect #{self.reconnects})", flush=True)

    def readinto(self, b) -> int:
        while True:
            try:
                data = self.resp.raw.read(len(b), decode_content=False)
                if not data and self.nbytes < EXPECTED_BYTES:
                    raise ConnectionError("stream ended early")
                break
            except (urllib3.exceptions.HTTPError, requests.RequestException, OSError) as e:
                print(f"  stream error at byte {self.nbytes:,}: {type(e).__name__}", flush=True)
                self._reconnect()
        n = len(data)
        if n:
            b[:n] = data
            self.sha.update(data)
            self.nbytes += n
        return n


class State:
    """Accumulators for the full-file profile (one per download attempt)."""

    def __init__(self):
        self.columns: list[str] | None = None
        self.rows = 0
        self.valid_rows = 0
        self.junk_rows = 0
        self.junk_examples: list[str] = []
        self.junk_bad_id = 0
        self.junk_null_issue_d = 0
        self.issue_d_unparseable = 0
        self.non_null = None
        self.numeric_ok = None
        self.examples: dict[str, list[str]] = {}
        self.nn_year = None
        self.rows_year = None
        self.vint_year: list[pd.DataFrame] = []
        self.vint_qtr: list[pd.DataFrame] = []
        self.ids: list[np.ndarray] = []
        self.hashes: list[np.ndarray] = []
        self.issue_min = self.issue_max = None
        self.last_pymnt_max = self.last_credit_max = None
        self.coerce_loss: dict[str, int] = {}
        self.kept: list[str] = []
        self.missing_cfg: list[str] = []
        self.missing_extra: list[str] = []


def _upd(cur, new, fn):
    if pd.isna(new):
        return cur
    return new if cur is None else fn(cur, new)


def process_chunk(df: pd.DataFrame, st: State, cfg_cols, all_cols, writers: dict):
    n = len(df)
    if st.columns is None:
        st.columns = list(df.columns)
        st.non_null = pd.Series(0, index=df.columns, dtype="int64")
        st.numeric_ok = pd.Series(0, index=df.columns, dtype="int64")
        present = set(df.columns)
        st.missing_cfg = [c for c in cfg_cols if c not in present]
        st.missing_extra = [c for c in all_cols if c not in cfg_cols and c not in present]
        st.kept = [c for c in all_cols if c in present]
        if "id" not in st.kept or "issue_d" not in st.kept:
            raise RuntimeError("id / issue_d missing from file")
        st.examples = {c: [] for c in df.columns}
        fields = []
        for c in st.kept:
            if c == "id":
                fields.append(pa.field(c, pa.int64()))
            elif c in STRING_COLS:
                fields.append(pa.field(c, pa.string()))
            else:
                fields.append(pa.field(c, pa.float64()))
        writers["extract_schema"] = pa.schema(fields)
        writers["sample_schema"] = pa.schema([pa.field(c, pa.string()) for c in df.columns])
        writers["extract"] = pq.ParquetWriter(writers["extract_path"], writers["extract_schema"],
                                              compression="zstd", compression_level=9)
        writers["sample"] = pq.ParquetWriter(writers["sample_path"], writers["sample_schema"],
                                             compression="zstd", compression_level=9)
    elif list(df.columns) != st.columns:
        raise RuntimeError("column mismatch between chunks")

    st.rows += n
    st.non_null += df.notna().sum()

    # numeric share + examples
    for c in df.columns:
        s = df[c]
        st.numeric_ok[c] += int(pd.to_numeric(s, errors="coerce").notna().sum())
        ex = st.examples[c]
        if len(ex) < 3:
            for v in s.dropna().unique()[:10]:
                if v not in ex and len(ex) < 3:
                    ex.append(str(v)[:60])

    # junk rows
    id_s = df["id"].str.strip()
    id_ok = id_s.str.fullmatch(r"\d+").fillna(False).astype(bool)
    issue_ok = df["issue_d"].notna()
    valid = id_ok & issue_ok
    junk = ~valid
    nj = int(junk.sum())
    st.junk_rows += nj
    st.junk_bad_id += int((~id_ok).sum())
    st.junk_null_issue_d += int((~issue_ok).sum())
    st.valid_rows += n - nj
    if nj and len(st.junk_examples) < 10:
        for v in df.loc[junk, "id"].head(10 - len(st.junk_examples)):
            st.junk_examples.append("<null>" if pd.isna(v) else str(v)[:100])

    # duplicates
    st.ids.append(df["id"].dropna().str.strip().to_numpy(dtype=object))
    st.hashes.append(pd.util.hash_pandas_object(df, index=False).to_numpy())

    # dates
    issue_dt = pd.to_datetime(df["issue_d"], format=DATE_FMT, errors="coerce")
    st.issue_d_unparseable += int((valid & issue_dt.isna()).sum())
    vdt = valid & issue_dt.notna()
    st.issue_min = _upd(st.issue_min, issue_dt[vdt].min(), min)
    st.issue_max = _upd(st.issue_max, issue_dt[vdt].max(), max)
    if "last_pymnt_d" in df.columns:
        st.last_pymnt_max = _upd(st.last_pymnt_max,
                                 pd.to_datetime(df["last_pymnt_d"], format=DATE_FMT, errors="coerce").max(), max)
    if "last_credit_pull_d" in df.columns:
        st.last_credit_max = _upd(st.last_credit_max,
                                  pd.to_datetime(df["last_credit_pull_d"], format=DATE_FMT, errors="coerce").max(), max)

    # missingness by year + vintage tables (valid rows with parseable issue_d)
    sub = df[vdt]
    yr = issue_dt[vdt].dt.year
    qt = issue_dt[vdt].dt.quarter
    nny = sub.notna().groupby(yr).sum()
    st.nn_year = nny if st.nn_year is None else st.nn_year.add(nny, fill_value=0)
    ry = yr.value_counts()
    st.rows_year = ry if st.rows_year is None else st.rows_year.add(ry, fill_value=0)
    term = sub["term"].str.strip().fillna("(missing)")
    status = sub["loan_status"].fillna("(missing)")
    st.vint_year.append(pd.DataFrame({"issue_year": yr, "term": term, "loan_status": status})
                        .groupby(["issue_year", "term", "loan_status"]).size().rename("n").reset_index())
    st.vint_qtr.append(pd.DataFrame({"issue_year": yr, "issue_quarter": qt, "term": term, "loan_status": status})
                       .groupby(["issue_year", "issue_quarter", "term", "loan_status"]).size()
                       .rename("n").reset_index())

    # ---- extract (valid rows only)
    v = df[valid]
    cols = {}
    for c in st.kept:
        s = v[c]
        if c == "id":
            cols[c] = pd.to_numeric(s.str.strip()).astype("int64")
        elif c in STRING_COLS:
            s = s.str.strip()
            cols[c] = s.where(s != "", None)
        else:
            num = pd.to_numeric(s, errors="coerce").astype("float64")
            loss = int(s.notna().sum() - num.notna().sum())
            if loss:
                st.coerce_loss[c] = st.coerce_loss.get(c, 0) + loss
            cols[c] = num
    writers["extract"].write_table(pa.Table.from_pandas(pd.DataFrame(cols), schema=writers["extract_schema"],
                                                        preserve_index=False))
    # ---- 5% sample, all columns as strings
    ids_int = pd.to_numeric(v["id"].str.strip()).astype("int64")
    smp = v[(ids_int % SAMPLE_MOD == 0).to_numpy()]
    if len(smp):
        writers["sample"].write_table(pa.Table.from_pandas(smp, schema=writers["sample_schema"], preserve_index=False))


def run_pass(paths: dict, cfg_cols, all_cols, max_chunks: int | None):
    st = State()
    writers = dict(paths)
    t0 = time.time()
    with requests.get(URL, stream=True, allow_redirects=True, timeout=(30, 300),
                      headers={"Accept-Encoding": "identity", "User-Agent": "credit-risk-scorecard-fetch/1.0"}) as resp:
        resp.raise_for_status()
        if resp.headers.get("Content-Encoding", "identity") not in ("identity", ""):
            raise RuntimeError(f"unexpected Content-Encoding {resp.headers.get('Content-Encoding')}")
        print(f"HTTP {resp.status_code}; content-length={resp.headers.get('Content-Length')}; final={resp.url[:80]}...",
              flush=True)
        reader = HashingReader(resp)
        buf = io.BufferedReader(reader, buffer_size=8 * 1024 * 1024)
        partial = False
        try:
            it = pd.read_csv(buf, chunksize=CHUNK_ROWS, low_memory=False, dtype=str,
                             keep_default_na=False, na_values=[""], encoding="utf-8",
                             encoding_errors="replace")
            for i, chunk in enumerate(it, 1):
                process_chunk(chunk, st, cfg_cols, all_cols, writers)
                el = time.time() - t0
                print(f"chunk {i:3d}  rows={st.rows:>10,}  bytes={reader.nbytes/1e6:8.1f} MB "
                      f"({reader.nbytes/EXPECTED_BYTES:5.1%})  {reader.nbytes/1e6/max(el,1e-9):5.1f} MB/s  "
                      f"elapsed={el:6.0f}s", flush=True)
                if max_chunks and i >= max_chunks:
                    partial = True
                    break
            if not partial:
                while buf.read(8 * 1024 * 1024):  # drain any trailing bytes so the hash covers the whole file
                    pass
        finally:
            for k in ("extract", "sample"):
                if k in writers and hasattr(writers[k], "close"):
                    writers[k].close()
    return st, reader, partial


# --------------------------------------------------------------------------- outputs
def write_outputs(st: State, reader: HashingReader, partial: bool, paths: dict, runtime_s: float, cfg_cols):
    prof = paths["profile_dir"]
    prof.mkdir(parents=True, exist_ok=True)
    actual_sha = reader.sha.hexdigest()
    sha_match = (actual_sha == EXPECTED_SHA256) and (reader.nbytes == EXPECTED_BYTES)

    ids = np.concatenate(st.ids)
    ids_s = pd.Series(ids)
    dup_id_rows = int(ids_s.duplicated().sum())
    dup_id_distinct = int(ids_s[ids_s.duplicated(keep=False)].nunique())
    hashes = np.concatenate(st.hashes)
    dup_rows = int(len(hashes) - len(np.unique(hashes)))

    # columns.csv
    cols = pd.DataFrame({
        "column": st.columns,
        "position": range(1, len(st.columns) + 1),
        "non_null": st.non_null.to_numpy(),
    })
    cols["missing_share"] = (1 - cols["non_null"] / st.rows).round(6)
    cols["numeric_share"] = (st.numeric_ok.to_numpy() / cols["non_null"].replace(0, np.nan)).round(6)
    cols["examples"] = [" | ".join(st.examples[c]) for c in st.columns]
    cols["kept_in_extract"] = cols["column"].isin(st.kept)
    cols.to_csv(prof / "columns.csv", index=False)

    # vintage tables
    vy = pd.concat(st.vint_year).groupby(["issue_year", "term", "loan_status"], as_index=False)["n"].sum()
    vy.to_csv(prof / "vintage_year_term_status.csv", index=False)
    vq = pd.concat(st.vint_qtr).groupby(["issue_year", "issue_quarter", "term", "loan_status"], as_index=False)["n"].sum()
    vq.to_csv(prof / "vintage_quarter_term_status.csv", index=False)

    # missingness by year: column x year non-null share
    share = st.nn_year.div(st.rows_year, axis=0)  # year x column
    mat = share.T.round(5)
    mat.columns = [int(c) for c in mat.columns]
    mat.index.name = "column"
    mat.to_csv(prof / "missing_by_year.csv")

    fmt = lambda t: None if t is None else t.strftime("%Y-%m")
    extract_size = paths["extract_path_final"].stat().st_size if paths["extract_path_final"].exists() else None
    profile = {
        "partial_run_for_testing": partial,
        "source_url": URL,
        "license": LICENSE,
        "sha256_expected": EXPECTED_SHA256,
        "sha256_actual": actual_sha,
        "sha256_match": bool(sha_match),
        "bytes_expected": EXPECTED_BYTES,
        "bytes_actual": reader.nbytes,
        "fetched_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": round(runtime_s, 1),
        "total_rows": st.rows,
        "valid_rows": st.valid_rows,
        "junk_rows": st.junk_rows,
        "junk_rows_bad_id": st.junk_bad_id,
        "junk_rows_null_issue_d": st.junk_null_issue_d,
        "junk_row_example_ids": st.junk_examples,
        "valid_rows_unparseable_issue_d": st.issue_d_unparseable,
        "n_columns": len(st.columns),
        "duplicate_id_rows": dup_id_rows,
        "distinct_ids_duplicated": dup_id_distinct,
        "duplicate_full_rows": dup_rows,
        "issue_d_min": fmt(st.issue_min),
        "issue_d_max": fmt(st.issue_max),
        "last_pymnt_d_max": fmt(st.last_pymnt_max),
        "last_credit_pull_d_max": fmt(st.last_credit_max),
        "config_columns_missing_from_file": st.missing_cfg,
        "leakage_demo_columns_missing_from_file": st.missing_extra,
        "kept_columns": st.kept,
        "numeric_coercion_lost_values": st.coerce_loss,
        "extract_file": str(paths["extract_path_final"].relative_to(paths["root"])).replace("\\", "/"),
        "extract_file_bytes": extract_size,
        "sample_file": str(paths["sample_path_final"].relative_to(paths["root"])).replace("\\", "/"),
        "sample_file_bytes": paths["sample_path_final"].stat().st_size if paths["sample_path_final"].exists() else None,
        "extract_type_rules": {
            "id": "int64", "string_columns": sorted(c for c in st.kept if c in STRING_COLS),
            "other_columns": "float64 via pd.to_numeric(errors='coerce')",
            "strings": "whitespace stripped (e.g. term ' 36 months' -> '36 months'); dates kept as original 'Mon-YYYY' text",
        },
    }
    (prof / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def source_md(p: dict) -> str:
    flag = "" if p["sha256_match"] else (
        "\n> **WARNING: the streamed SHA-256 / byte count did NOT match the expected values. "
        "Treat the extract as unverified.**\n")
    return f"""# Data source: Lending Club accepted loans 2007-2018Q4

- **Source URL:** {p['source_url']}
  (Hugging Face mirror `codesignal/lending-club-loan-accepted` of the Kaggle
  `wordsforthewise/lending-club` accepted-loans file.)
- **License:** {p['license']}
- **Fetched (UTC):** {p['fetched_at_utc']}
- **Bytes streamed:** {p['bytes_actual']:,} (expected {p['bytes_expected']:,})
- **SHA-256 (streamed):** `{p['sha256_actual']}`
- **SHA-256 (expected, from HF X-Linked-ETag):** `{p['sha256_expected']}` -> match: **{p['sha256_match']}**
{flag}
## Storage policy
The full 1.6 GB CSV is **never stored locally** (not even as a temporary file). It is streamed over
HTTP, hashed on the fly and parsed in 200,000-row chunks. Only a slim Parquet extract, a 5% all-column
sample and a small full-file profile are kept.

## What was parsed (full file, all columns)
- {p['total_rows']:,} data rows, {p['n_columns']} columns; {p['valid_rows']:,} valid rows,
  {p['junk_rows']:,} junk/footer rows (id not an integer or issue_d null; example ids: {p['junk_row_example_ids']}).
- issue_d range {p['issue_d_min']} to {p['issue_d_max']}; last_pymnt_d max {p['last_pymnt_d_max']};
  last_credit_pull_d max {p['last_credit_pull_d_max']}.
- Duplicate id rows: {p['duplicate_id_rows']:,}; duplicate full rows: {p['duplicate_full_rows']:,}.
- Full profile: `artifacts/data_profile/` (profile.json, columns.csv, vintage_*.csv, missing_by_year.csv).

## Extract rule
`{p['extract_file']}` ({(p['extract_file_bytes'] or 0)/1e6:.1f} MB): every valid row (all terms, all years).
Columns = `ID_COLS + CANDIDATE_FEATURES + BENCHMARK_COLS + LOSS_CALIBRATION_COLS + EXCLUDED_FAIR_LENDING`
from `src/scorecard/config.py`, plus post-origination fields kept ONLY for the leakage demonstration:
{', '.join(LEAKAGE_DEMO_COLS)}. Columns absent from the file are skipped and listed in profile.json
(config columns missing: {p['config_columns_missing_from_file']}).

Type conversion: `id` -> int64; text/date columns stay strings with whitespace stripped
(`term` becomes e.g. `36 months`; dates keep the original `Mon-YYYY` text, parsed later in the pipeline);
all other columns -> float64 via `pd.to_numeric(errors="coerce")`. Empty cells are null; other
placeholders (e.g. emp_length `n/a`) are kept verbatim. The file was read with every field as a string,
so nothing is altered by type inference.

`{p['sample_file']}` ({(p['sample_file_bytes'] or 0)/1e6:.1f} MB): deterministic 5% sample
(valid rows with `int(id) % {SAMPLE_MOD} == 0`), ALL {p['n_columns']} columns as original strings; used for
leakage analysis across every column.

## Reproduce
```
python scripts/fetch_data.py          # add --force to re-download and rebuild
```
"""


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if the extract exists")
    ap.add_argument("--max-chunks", type=int, default=None, help="testing: stop after N chunks")
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT, help="testing: write outputs under this root")
    args = ap.parse_args()

    root = args.out_root
    cfg = load_config(PROJECT_ROOT)
    cfg_cols, all_cols = wanted_columns(cfg)
    extract_final = root / "data" / "raw" / EXTRACT_NAME
    sample_final = root / "data" / "interim" / SAMPLE_NAME
    profile_dir = root / "artifacts" / "data_profile"
    if extract_final.exists() and (profile_dir / "profile.json").exists() and not args.force:
        print(f"{extract_final} exists; skipping (use --force to rebuild).")
        return
    for d in (extract_final.parent, sample_final.parent, profile_dir, root / "docs"):
        d.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": root, "profile_dir": profile_dir,
        "extract_path": str(extract_final) + ".partial", "sample_path": str(sample_final) + ".partial",
        "extract_path_final": extract_final, "sample_path_final": sample_final,
    }

    t0 = time.time()
    for attempt in range(1, ATTEMPTS + 1):
        print(f"=== attempt {attempt}/{ATTEMPTS} ===", flush=True)
        try:
            st, reader, partial = run_pass(paths, cfg_cols, all_cols, args.max_chunks)
        except (requests.RequestException, urllib3.exceptions.HTTPError, OSError, ConnectionError) as e:  # network trouble: restart whole run
            print(f"network error: {type(e).__name__}: {e}", flush=True)
            if attempt == ATTEMPTS:
                raise
            time.sleep(10)
            continue
        if not partial and reader.nbytes != EXPECTED_BYTES and attempt < ATTEMPTS:
            print(f"byte count {reader.nbytes} != expected {EXPECTED_BYTES}; retrying", flush=True)
            continue
        break

    Path(paths["extract_path"]).replace(extract_final)
    Path(paths["sample_path"]).replace(sample_final)
    profile = write_outputs(st, reader, partial, paths, time.time() - t0, cfg_cols)

    text = source_md(profile)
    (root / "data" / "raw" / "SOURCE.md").write_text(text, encoding="utf-8")
    (root / "docs" / "data_source.md").write_text(text, encoding="utf-8")

    print("\n=== DONE ===")
    print(f"sha256 match: {profile['sha256_match']}  bytes: {profile['bytes_actual']:,}")
    print(f"rows total/valid/junk: {profile['total_rows']:,} / {profile['valid_rows']:,} / {profile['junk_rows']:,}")
    print(f"config columns missing from file: {profile['config_columns_missing_from_file']}")
    if not profile["sha256_match"] and not partial:
        print("!!! WARNING: SHA-256 / size MISMATCH - outputs kept but flagged in profile.json !!!")
    if st.coerce_loss:
        print(f"note: numeric coercion dropped non-numeric values in: {st.coerce_loss}")


if __name__ == "__main__":
    sys.exit(main())
