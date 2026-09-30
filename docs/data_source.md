# Data source: Lending Club accepted loans 2007-2018Q4

- **Source URL:** https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv
  (Hugging Face mirror `codesignal/lending-club-loan-accepted` of the Kaggle
  `wordsforthewise/lending-club` accepted-loans file.)
- **License:** CC0-1.0 (public domain dedication)
- **Fetched (UTC):** 2026-09-30T20:13:22Z
- **Bytes streamed:** 1,675,133,810 (expected 1,675,133,810)
- **SHA-256 (streamed):** `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`
- **SHA-256 (expected, from HF X-Linked-ETag):** `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a` -> match: **True**

## Storage policy
The full 1.6 GB CSV is **never stored locally** (not even as a temporary file). It is streamed over
HTTP, hashed on the fly and parsed in 200,000-row chunks. Only a slim Parquet extract, a 5% all-column
sample and a small full-file profile are kept.

## What was parsed (full file, all columns)
- 2,260,701 data rows, 151 columns; 2,260,668 valid rows,
  33 junk/footer rows (id not an integer or issue_d null; example ids: ['Total amount funded in policy code 1: 6417608175', 'Total amount funded in policy code 2: 1944088810', 'Total amount funded in policy code 1: 1741781700', 'Total amount funded in policy code 2: 564202131', 'Total amount funded in policy code 1: 1791201400', 'Total amount funded in policy code 2: 651669342', 'Total amount funded in policy code 1: 1443412975', 'Total amount funded in policy code 2: 511988838', 'Total amount funded in policy code 1: 2063142975', 'Total amount funded in policy code 2: 823319310']).
- issue_d range 2007-06 to 2018-12; last_pymnt_d max 2019-03;
  last_credit_pull_d max 2019-04.
- Duplicate id rows: 0; duplicate full rows: 0.
- Full profile: `artifacts/data_profile/` (profile.json, columns.csv, vintage_*.csv, missing_by_year.csv).

## Extract rule
`data/raw/lc_accepted_2007_2018Q4_extract.parquet` (159.7 MB): every valid row (all terms, all years).
Columns = `ID_COLS + CANDIDATE_FEATURES + BENCHMARK_COLS + LOSS_CALIBRATION_COLS + EXCLUDED_FAIR_LENDING`
from `src/scorecard/config.py`, plus post-origination fields kept ONLY for the leakage demonstration:
total_pymnt, total_rec_int, total_rec_late_fee, last_pymnt_d, last_pymnt_amnt, last_fico_range_high, last_fico_range_low, out_prncp, last_credit_pull_d, debt_settlement_flag, hardship_flag, pymnt_plan, funded_amnt_inv, initial_list_status, policy_code, title, emp_title. Columns absent from the file are skipped and listed in profile.json
(config columns missing: []).

Type conversion: `id` -> int64; text/date columns stay strings with whitespace stripped
(`term` becomes e.g. `36 months`; dates keep the original `Mon-YYYY` text, parsed later in the pipeline);
all other columns -> float64 via `pd.to_numeric(errors="coerce")`. Empty cells are null; other
placeholders (e.g. emp_length `n/a`) are kept verbatim. The file was read with every field as a string,
so nothing is altered by type inference.

`data/interim/lc_all_columns_sample_5pct.parquet` (16.5 MB): deterministic 5% sample
(valid rows with `int(id) % 20 == 0`), ALL 151 columns as original strings; used for
leakage analysis across every column.

## Reproduce
```
python scripts/fetch_data.py          # add --force to re-download and rebuild
```
