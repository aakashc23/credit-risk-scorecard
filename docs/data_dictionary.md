# Data Dictionary

## Getting the data
Run `python scripts/fetch_data.py`. It streams the Lending Club accepted-loans CSV (same file
as Kaggle `wordsforthewise/lending-club`) and verifies its SHA-256. It writes a column-subset
parquet to `data/raw/lc_accepted_2007_2018Q4_extract.parquet` without storing the 1.68 GB CSV.
See [data_source.md](data_source.md). The pipeline also accepts the original `.csv`/`.csv.gz`.

## Columns used
| Column | Role | Notes |
|---|---|---|
| id | key | |
| issue_d | split only | "Mon-YYYY" → date; used for windows and credit-history length |
| term | filter | keep " 36 months" |
| loan_status | target source | see methodology §3 |
| loan_amnt | feature | requested amount; also EAD base |
| emp_length | feature | "< 1 year" … "10+ years" → years; "n/a" → missing |
| home_ownership | feature | ANY/NONE/OTHER pooled |
| annual_inc | feature | also used for ratio features |
| verification_status | feature | |
| purpose | feature | |
| application_type | feature | Individual / Joint |
| dti | feature | |
| delinq_2yrs, inq_last_6mths, open_acc, pub_rec, total_acc, mort_acc, pub_rec_bankruptcies, tax_liens, collections_12_mths_ex_med | feature | bureau counts at application |
| mths_since_last_delinq, mths_since_last_record, mths_since_recent_inq, mo_sin_rcnt_tl | feature | missing usually means "never", which gets its own WOE bin |
| earliest_cr_line | feature (engineered) | → credit_history_months |
| fico_range_low/high | feature (engineered) | → fico_mid (application FICO, **not** last_fico_*) |
| revol_bal, revol_util, bc_util, percent_bc_gt_75, total_rev_hi_lim, tot_cur_bal, avg_cur_bal, acc_open_past_24mths, num_actv_rev_tl, num_tl_op_past_12m | feature | bureau balances/utilisation at application |
| grade, sub_grade, int_rate, installment | benchmark only | LC's pricing model. Not a feature. |
| funded_amnt, total_rec_prncp, recoveries, collection_recovery_fee | LGD/EAD calibration & back-test only | **post-origination: never features** |

## Engineered features
| Feature | Formula |
|---|---|
| credit_history_months | months between earliest_cr_line and issue_d |
| fico_mid | (fico_range_low + fico_range_high) / 2 |
| emp_length_years | parsed emp_length (0 for "< 1 year", 10 for "10+ years") |
| loan_to_income | loan_amnt / annual_inc |
| revol_bal_to_income | revol_bal / annual_inc |
