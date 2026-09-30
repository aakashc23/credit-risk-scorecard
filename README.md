# Credit Risk Scorecard & Approval Cut-off Simulator

An interpretable credit-risk scorecard built on historical **Lending Club** loan data.
It estimates each applicant's **probability of default (PD)**, converts it to a
**300–850 credit score**, computes **Expected Loss = PD × LGD × EAD**, and lets a user
explore how moving the **approval cut-off** changes approval volume, bad rate and
expected loss.

> Status: 🚧 in progress. Metrics below are filled in automatically from
> `artifacts/metrics.json` after the pipeline runs — nothing here is hand-typed.

## Business question
*"If we approve every applicant scoring at or above X, how many loans do we book, what bad
rate should we expect, and how many dollars do we expect to lose?"*

## Pipeline
```
Lending Club accepted loans (raw, read-only)
 → clean & type-cast
 → keep only mature 36-month loans (outcome fully observed)
 → target: 1 = Charged Off / Default, 0 = Fully Paid
 → drop post-origination (leakage) fields — whitelist of application-time features only
 → feature engineering (credit-history length, FICO mid, loan-to-income, …)
 → fine/coarse binning → WOE / Information Value → feature selection
 → logistic regression on WOE features
 → PD → 300–850 score (points-to-double-odds scaling) → points-based scorecard table
 → validation: AUC, Gini, KS, decile table, calibration, PSI (train / test / out-of-time)
 → Expected Loss = PD × LGD × EAD
 → cut-off simulator (approval / rejection / bad rate / expected loss / counts)
```

## Results
| Metric | Train | Test | Out-of-time |
|---|---|---|---|
| AUC | TBD | TBD | TBD |
| Gini | TBD | TBD | TBD |
| KS | TBD | TBD | TBD |

_TBD — pending pipeline run._

## Quick start
```bash
pip install -e ".[dev]"
# put accepted_2007_to_2018Q4.csv.gz in data/raw/  (see docs/data_dictionary.md)
python scripts/run_pipeline.py
pytest -q
streamlit run app/streamlit_app.py
```

## Repo layout
See [CLAUDE.md](CLAUDE.md#layout). Methodology: [docs/methodology.md](docs/methodology.md).
Decision log: [docs/decisions.md](docs/decisions.md).

## Key design decisions
- **Leakage control by whitelist**: only fields known at application time can be features.
  `recoveries`, `total_pymnt`, `last_pymnt_d`, `collection_recovery_fee`, `last_fico_*` etc.
  are blocked, and a unit test enforces it.
- **Independent score**: LC's own `grade` / `int_rate` are excluded from the model and used
  only as a benchmark.
- **Out-of-time validation**: the model is developed on earlier vintages and validated on a
  later vintage it has never seen.
- **Honest limitations**: data contains *accepted* loans only (no reject inference), so
  cut-off results describe re-scoring LC's approved population.

## Data
Lending Club accepted loans 2007–2018Q4 (Kaggle: `wordsforthewise/lending-club`).
Raw data is not committed to this repository.

## Disclaimer
Educational / portfolio project. Not a lending decision system.
