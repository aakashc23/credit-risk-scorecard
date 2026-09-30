# Decision Log

| # | Date | Decision | Rationale |
|---|---|---|---|
| 1 | 2026-10-01 | Use Kaggle `accepted_2007_to_2018Q4.csv.gz` as the only data source; never edit it. | Canonical public Lending Club dump; reproducibility. |
| 2 | 2026-10-01 | Restrict to 36-month loans issued 2012–2015. | Loans are mature in a 2018Q4 snapshot and outcome window is consistent. |
| 3 | 2026-10-01 | Dev = 2012–2014 (70/30 train/test), OOT = 2015. | Out-of-time validation mimics real deployment. |
| 4 | 2026-10-01 | Target: Charged Off/Default = 1, Fully Paid = 0, others excluded. | Standard Lending Club default definition. |
| 5 | 2026-10-01 | Whitelist-based feature selection; explicit leakage blacklist + unit test. | Prevents post-origination leakage (recoveries, total_pymnt, last_pymnt_d, …). |
| 6 | 2026-10-01 | Exclude grade/sub_grade/int_rate/installment from features; use grade as benchmark. | They are outputs of LC's own risk model. Our score should be independent. |
| 7 | 2026-10-01 | Exclude zip_code/addr_state. | Fair-lending hygiene: geography can proxy for protected classes. |
| 8 | 2026-10-01 | Logistic regression on WOE features; coefficients must be negative. | Interpretable, industry-standard scorecard. |
| 9 | 2026-10-01 | Scaling: 650 @ 10:1 odds, PDO 50, clip 300–850. | FICO-like range; easy to explain. |
| 10 | 2026-10-01 | LGD & EAD ratio = pooled estimates from training-window defaults. | Keeps EL simple. Post-origination fields are used only to calibrate parameters, never as features. |
| 11 | 2026-10-01 | Cut-off analysis on OOT population. | Closest proxy for "next year's applicants". |
| 12 | 2026-10-01 | DuckDB for SQL profiling; Streamlit for the dashboard; dashboard reads small committed artifacts only. | Lightweight, free deployment (Streamlit Community Cloud), no raw data in repo. |
| 13 | 2026-10-01 | Raw input is the parquet extract `lc_accepted_2007_2018Q4_extract.parquet` (`RAW_FILENAME`); the loader also still reads CSV/CSV.gz. Added `CATEGORICAL_FEATURES` to config. | Extract is faster to read and smaller; loader reads parquet in row-group batches, filters term/window per batch. Categorical list makes WOE binning explicit. |
| 14 | 2026-10-01 | Missing-share filter (`MAX_MISSING_SHARE`) is applied to train as designed, so `mths_since_last_delinq` / `mths_since_last_record` are dropped if >30% missing; the data dictionary's "missing = never" bin idea is therefore not used for them. Revisit after the real run if IV is lost. | Follows the approved config; logged because it conflicts with the data-dictionary note. |
| 15 | 2026-10-01 | Dashboard scores single applicants from `woe_bins.json` + `model.json` (intercept, betas, scaling, LGD/EAD) with numpy only; sklearn, scipy and duckdb are not imported on the app path (scipy import made lazy in `woe.py`, sklearn type-only in `scoring.py`). | Streamlit Community Cloud installs only `requirements.txt`; keeps the deployment light and proves the scorecard is fully transparent. |
| 16 | 2026-10-01 | Risk bands (`RISK_BANDS`): >=740 Very Low, 680-739 Low, 620-679 Medium, 560-619 High, <560 Very High. Simulator default cut-off = highest cut-off keeping >=80% of applicants (`DEFAULT_APPROVAL_TARGET`). | Each band step roughly doubles the odds of default (PDO 50); the default cut-off is computed from the data, never hard-coded. |
| 17 | 2026-10-01 | Features with no bin for missing/unseen values get an explicit `Unbinned` row in the points table (neutral points, WOE 0). | The model scores such values at WOE 0, so the points table must give the same neutral points rather than 0. |
| 18 | 2026-10-01 | Power BI: pipeline exports three existing tables to `artifacts/powerbi/`; `docs/powerbi_guide.md` documents DAX and layout. No `.pbix` is produced. | Optional BI deliverable without new analytics or binary files. |
| 19 | 2026-10-01 | Risk bands moved to 50-point steps: >=700 Very Low, 650-699 Low, 600-649 Medium, 550-599 High, <550 Very High (replaces #16's bands). | With the old bands the median 2015 applicant (score ~626, PD ~13%, about the 14.9% portfolio average) was labelled "High" and 39% of applicants fell in "High". New bands: observed 2015 bad rates 3.3% / 8.0% / 14.2% / 23.4% / 34.1%, so "Medium" means average risk. |
