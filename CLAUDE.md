# CLAUDE.md — Credit Risk Scorecard & Approval Cut-off Simulator

Guidance for AI assistants and contributors working in this repo.

## What this project is
A resume-scale, interpretable credit-risk scorecard built on historical Lending Club
accepted-loan data: clean → mature-loan filter → default target → leakage removal →
feature engineering → WOE/IV → logistic regression → PD → 300–850 score →
validation (AUC/KS/Gini/deciles/PSI/calibration) → Expected Loss (PD × LGD × EAD) →
approval cut-off simulator (Streamlit).

It is **not** an enterprise system. Keep it small, readable and finishable.

## Hard rules (do not break)
1. **Never modify raw data.** `data/raw/` is read-only input. All processing is reproducible
   from `python scripts/run_pipeline.py`.
2. **No post-origination leakage in model features.** Features come ONLY from the explicit
   whitelist in `src/scorecard/config.py` (`CANDIDATE_FEATURES`). Fields such as
   `recoveries`, `collection_recovery_fee`, `total_pymnt*`, `total_rec_*`, `last_pymnt_*`,
   `last_fico_*`, `out_prncp*`, `next_pymnt_d`, `last_credit_pull_d`, `hardship_*`,
   `settlement_*`, `debt_settlement_flag`, `pymnt_plan` are forbidden as features
   (`LEAKAGE_COLUMNS`). A test enforces this.
   - Exception: post-origination payment fields may be used **only** to estimate the
     portfolio-level LGD / EAD parameters on the *training window* defaults and to measure
     *realised* loss for back-testing. They never enter the model.
3. **Do not fabricate metrics.** Every number in README/docs/dashboard must come from a
   file written by the pipeline (`artifacts/metrics.json`, `artifacts/*.csv`). If data has
   not been run, write "TBD — pending pipeline run", never a guess.
4. **Main model is logistic regression on WOE features.** No XGBoost, deep learning,
   microservices, Kubernetes, auth, or extra APIs.
5. LC-assigned pricing outputs (`grade`, `sub_grade`, `int_rate`, `installment`) are **not**
   model features — they are LC's own risk model. `grade` is used only as a benchmark.
6. Geographic fields (`zip_code`, `addr_state`) are excluded (fair-lending hygiene).

## Layout
```
data/raw/            input CSV(.gz) — gitignored, never edited
data/interim/        cleaned/filtered parquet — gitignored
data/processed/      model-ready WOE tables — gitignored
src/scorecard/       python package (config, data, features, woe, model, scoring,
                     validation, loss, cutoff, plots, pipeline)
sql/                 DuckDB SQL for portfolio profiling & cut-off summaries
scripts/             CLI entry points (run_pipeline.py, make_sample.py)
artifacts/           SMALL committed outputs the app reads (scorecard, metrics, cutoff table,
                     scored OOT sample). Must stay < ~25 MB total.
reports/figures/     PNG charts for README/docs
app/streamlit_app.py dashboard (reads artifacts/ only — never raw data)
tests/               pytest; uses synthetic fixtures, not real data
docs/                methodology, decisions log, data dictionary, implementation plan
```

## Commands
```bash
pip install -e ".[dev]"             # install package + dev tools
python scripts/run_pipeline.py      # full pipeline: raw -> artifacts/ + reports/figures/
pytest -q                           # tests
streamlit run app/streamlit_app.py  # dashboard
```

## Conventions
- Python 3.11+, pandas, numpy, scikit-learn, duckdb, matplotlib, streamlit, plotly.
- All paths/constants/dates in `src/scorecard/config.py`; no magic numbers in modules.
- Random seed `RANDOM_STATE = 42` everywhere.
- WOE convention: `WOE = ln(%good / %bad)` → higher WOE = lower risk → LR coefficients on
  WOE features must be **negative** when target = 1 for bad.
- Log every important modelling decision in `docs/decisions.md` (date, decision, why).
- Keep functions small and pure where possible so they are unit-testable.
