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
