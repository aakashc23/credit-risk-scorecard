# Implementation Plan

Planned with Opus; implemented phase by phase. Each phase follows the same loop:
inspect → plan → implement → run → test → fix → document.

**Status (2026-10-01): all phases complete.** Live app: https://credit-risk-scorecard-aakash.streamlit.app

| Phase | Scope | Deliverables | Needs real data? |
|---|---|---|---|
| 0 | Scaffold | structure, CLAUDE.md, README, config, docs, git | no ✅ |
| 1 | Data acquisition | `scripts/fetch_data.py` → `data/raw/lc_accepted_2007_2018Q4_extract.parquet` | done ✅ |
| 2 | Core library + tests | `src/scorecard/*` modules, synthetic-data pytest suite, pipeline smoke test | no |
| 3 | Real run: data prep + SQL | interim parquet, status mix, missingness by year, DuckDB profiling CSVs | yes |
| 4 | Real run: WOE/IV, model, scaling | iv_table.csv, woe_bins.json, scorecard_points.csv, model coefficients | yes |
| 5 | Real run: validation, EL, cut-off | metrics.json, deciles, calibration, PSI, cutoff_table.csv, scored_oot.parquet, figures | yes |
| 6 | Streamlit dashboard | `app/streamlit_app.py` reading artifacts/ only | artifacts |
| 7 | Docs finalisation | README results auto-filled from metrics.json, methodology results | artifacts |
| 8 | GitHub + deploy | public repo, Streamlit Community Cloud | **user login** |

## Module contract (`src/scorecard/`)
- `config.py`: constants (done)
- `data.py`: `load_raw(path, sample=None)` (chunked, usecols, filter term+window early),
  `parse_types(df)`, `make_target(df)` → (df, stats), `assign_split(df)` → train/test/oot column
- `features.py`: `engineer(df)`, `feature_columns(df)` (whitelist-derived; asserts no leakage)
- `woe.py`: `fit_bins(x, y, kind)`, `WoeBinner` (fit/transform over many columns), `iv_table()`
- `model.py`: `select_features(iv, woe_train)`, `fit_logit(X, y)` with sign-check loop
- `scoring.py`: `pd_to_score`, `score_to_pd`, `scorecard_points(binner, model)`
- `validation.py`: `auc`, `gini`, `ks`, `decile_table`, `calibration_table`, `psi`
- `loss.py`: `calibrate_lgd_ead(train_bads)`, `expected_loss(pd, loan_amnt, lgd, ead_ratio)`, `realised_loss(df)`
- `cutoff.py`: `cutoff_table(scored, step)`, `cutoff_summary(scored, cutoff)`
- `plots.py`: matplotlib figures → reports/figures/*.png
- `sql.py`: run `sql/*.sql` with DuckDB against interim/scored parquet → artifacts/*.csv
- `pipeline.py`: `run(raw_path, sample=None)` orchestration; `scripts/run_pipeline.py` CLI with `--sample N`

## Artifacts consumed by the app
`metrics.json`, `iv_table.csv`, `scorecard_points.csv`, `model_coefficients.csv`,
`decile_table_{split}.csv`, `calibration_oot.csv`, `cutoff_table.csv`, `scored_oot.parquet`
(score, pd, ead, el, bad, realised_loss, sub_grade), `loss_params.json`, `sql_*.csv`.
