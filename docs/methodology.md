# Methodology

Planned method. Sections are updated with actual results once the pipeline runs. Any
number here must come from `artifacts/`.

## 1. Data
- Source: Lending Club accepted loans 2007–2018Q4 (`accepted_2007_to_2018Q4.csv.gz`, Kaggle
  `wordsforthewise/lending-club`). The raw file is read-only.
- Only the columns we need are read, in chunks, so memory stays low.

## 2. Sample definition
| Choice | Value | Why |
|---|---|---|
| Term | 36 months only | One term keeps the outcome window consistent, and 36m loans from ≤2015 are mature in a 2018Q4 snapshot. |
| Development window | 2012-01 to 2014-12 issue dates | Recent enough that bureau fields are populated, and mature. |
| Out-of-time (OOT) window | 2015-01 to 2015-12 | Simulates scoring future applicants. |
| Train/test | 70/30 stratified random split inside the development window | In-time check for overfitting. |

## 3. Target
- `bad = 1`: Charged Off, Default (and the "does not meet credit policy" variant).
- `bad = 0`: Fully Paid (and its "does not meet credit policy" variant).
- Indeterminate (Current / Late / Grace) is excluded and its count is logged. After the
  maturity filter this group should be tiny.

## 4. Leakage control
Features come only from a whitelist (`CANDIDATE_FEATURES`) of application-time fields. A
test asserts the whitelist never overlaps `LEAKAGE_COLUMNS` (recoveries, total_pymnt,
last_pymnt_d, collection_recovery_fee, last_fico_*, out_prncp, hardship/settlement fields …).
LC's own risk outputs (grade, sub_grade, int_rate, installment) are excluded so the score is
independent. `grade` serves only as a benchmark.

## 5. Feature engineering
`credit_history_months` (issue_d − earliest_cr_line), `fico_mid`, `emp_length_years`,
`loan_to_income`, `revol_bal_to_income`, and cleaned percentage fields.

## 6. WOE / IV
- Numeric: up to 10 quantile fine bins (each ≥5% of rows), merged into ≤6 coarse bins with a
  monotonic bad rate. Missing values get their own bin.
- Categorical: categories under 1% share are pooled into "Other", then similar groups are merged.
- `WOE = ln(%good / %bad)` with +0.5 smoothing. `IV = Σ (%good − %bad) × WOE`.
- Selection: IV ≥ 0.02. IV > 0.5 is flagged for leakage review. If two WOE features have
  |corr| > 0.7, the one with the higher IV is kept. At most 15 features.

## 7. Model
Logistic regression on WOE-transformed features, fitted on the train split. Every coefficient
must be negative under this WOE convention; features with the wrong sign are dropped and the
model is refitted.

## 8. Score scaling
`score = offset + factor × ln(odds_good)`, `factor = PDO / ln 2`. The anchor is 650 points at
10:1 good:bad odds, with PDO = 50. Scores are clipped to 300–850. The result is also shown as a
points-per-attribute scorecard table.

## 9. Validation (train / test / OOT)
AUC, Gini (= 2·AUC − 1), KS, a decile table (count, bads, bad rate, cumulative capture),
calibration (mean predicted PD vs observed bad rate by decile), PSI of the score
(train → OOT), and a benchmark against LC `sub_grade`.

## 10. Expected loss
`EL = PD × LGD × EAD`
- **EAD** = `loan_amnt × EAD_ratio`. EAD_ratio = Σ(funded_amnt − total_rec_prncp) / Σ funded_amnt
  over **training-window defaults**.
- **LGD** = Σ(EAD_at_default − (recoveries − collection_recovery_fee)) / Σ EAD_at_default over
  the same defaults.
- Post-origination fields are used **only** for these two portfolio-level parameters and
  never as model inputs.
- Back-test: predicted EL vs realised loss on OOT.

## 11. Cut-off simulator
The analysis runs on the OOT population. For every cut-off (300–850, step 5) it reports:
approved / rejected counts, approval / rejection rate, expected bad rate (mean PD of approved
loans), observed bad rate, approved exposure, expected loss ($ and % of exposure), and
realised loss.

## 12. Limitations
- Accepted loans only, with no reject inference. The simulator re-scores LC's approved book.
- A single pooled LGD and EAD ratio, with no macro scenarios.
- 60-month loans are out of scope.
