"""End-to-end pipeline: raw Lending Club CSV -> model, metrics, artifacts, figures, SQL."""
from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard import plots
from scorecard.config import (
    ARTIFACTS,
    BASE_ODDS,
    BASE_SCORE,
    BENCHMARK_COLS,
    DATA_INTERIM,
    DATA_PROCESSED,
    FIGURES,
    ID_COLS,
    LOSS_CALIBRATION_COLS,
    PDO,
    RAW_PATH,
    SQL_DIR,
    TARGET,
)
from scorecard.cutoff import cutoff_table
from scorecard.data import assign_split, load_raw, make_target, parse_types
from scorecard.features import engineer, feature_columns, high_missing_features
from scorecard.loss import calibrate_lgd_ead, expected_loss, realised_loss
from scorecard.model import fit_logit, predict_pd, select_features
from scorecard.scoring import factor, offset, pd_to_score, points_score, scorecard_points
from scorecard.sql import run_sql
from scorecard.validation import (
    auc,
    calibration_table,
    decile_table,
    gini,
    grade_to_ordinal,
    ks,
    psi,
)
from scorecard.woe import WoeBinner

logger = logging.getLogger(__name__)

SPLITS = ["train", "test", "oot"]
SCORED_OOT_COLS = ["score", "pd", "ead", "el", "bad", "realised_loss", "loan_amnt", "sub_grade",
                   "issue_d"]


def _clean(obj):
    """Make nested data JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8")


def _split_metrics(d: pd.DataFrame) -> dict:
    return {"n": len(d), "bads": int(d[TARGET].sum()), "bad_rate": float(d[TARGET].mean()),
            "auc": auc(d[TARGET], d["pd"]), "gini": gini(d[TARGET], d["pd"]),
            "ks": ks(d[TARGET], d["pd"])}


def _benchmark_metrics(oot: pd.DataFrame) -> dict:
    """Discrimination of LC's own sub_grade (A1 best ... G5 worst) on the OOT loans."""
    risk = grade_to_ordinal(oot["sub_grade"])
    ok = risk.notna()
    y = oot.loc[ok, TARGET]
    return {"n": int(ok.sum()), "auc": auc(y, risk[ok]), "gini": gini(y, risk[ok]),
            "ks": ks(y, risk[ok])}


def _require_splits(df: pd.DataFrame) -> None:
    """Every split must hold both good and bad loans (else the sample is too small)."""
    classes = df.groupby("split")[TARGET].nunique().reindex(SPLITS).fillna(0)
    weak = classes[classes < 2].index.tolist()
    if weak:
        raise ValueError(f"split(s) {weak} are empty or hold a single class; the sample covers "
                         "too little of the development / out-of-time windows")


def _prepare(raw_path, sample, interim_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load -> parse -> target -> split -> engineer. Returns the frame and a row-count log."""
    raw = load_raw(raw_path, sample)
    counts = {"rows_read": int(raw.attrs.get("n_rows_read", len(raw))),
              "rows_36m_in_windows": len(raw)}
    df = parse_types(raw)
    df.to_parquet(interim_dir / "loans_filtered.parquet", index=False)
    df, target_stats = make_target(df)
    counts["rows_with_known_outcome"] = len(df)
    df = assign_split(df)
    _require_splits(df)
    counts["rows_per_split"] = {s: int((df["split"] == s).sum()) for s in SPLITS}
    return engineer(df), {"row_counts": counts, "target": target_stats}


def _score_frame(df: pd.DataFrame, lgd: float, ead_ratio: float) -> pd.DataFrame:
    df = df.copy()
    df["ead"] = df["loan_amnt"] * ead_ratio
    df["el"] = expected_loss(df["pd"], df["loan_amnt"], lgd, ead_ratio)
    df["realised_loss"] = realised_loss(df)
    return df


def _backtest(d: pd.DataFrame) -> dict:
    predicted, realised = float(d["el"].sum()), float(d["realised_loss"].sum())
    return {"expected_loss": predicted, "realised_loss": realised,
            "ratio_expected_to_realised": predicted / realised if realised else None}


def run(raw_path: Path | str = RAW_PATH, sample: int | None = None, *,
        artifacts_dir: Path = ARTIFACTS, interim_dir: Path = DATA_INTERIM,
        processed_dir: Path = DATA_PROCESSED, figures_dir: Path = FIGURES,
        sql_dir: Path = SQL_DIR) -> dict:
    """Run the whole pipeline and return the metrics dict (also written to metrics.json).

    All output locations are injectable so tests never touch the real ``artifacts/``.
    """
    for d in (artifacts_dir, interim_dir, processed_dir, figures_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    artifacts_dir, interim_dir = Path(artifacts_dir), Path(interim_dir)
    processed_dir, figures_dir = Path(processed_dir), Path(figures_dir)

    # ---- data -------------------------------------------------------------------------
    df, summary = _prepare(raw_path, sample, interim_dir)
    train = df[df["split"] == "train"]

    # ---- features ---------------------------------------------------------------------
    candidates = feature_columns(df)
    dropped_missing = high_missing_features(train, candidates)
    features = [c for c in candidates if c not in dropped_missing]
    logger.info("features: %d candidates, %d dropped for missing share: %s",
                len(candidates), len(dropped_missing), dropped_missing)
    keep = list(dict.fromkeys(ID_COLS + [TARGET, "split"] + features + BENCHMARK_COLS
                              + LOSS_CALIBRATION_COLS))
    df[keep].to_parquet(interim_dir / "modelling.parquet", index=False)

    # ---- WOE / selection / model -------------------------------------------------------
    binner = WoeBinner().fit(train, train[TARGET], features)
    iv = binner.iv_table()
    selected, selection = select_features(iv, binner.transform(train, features))
    model, final, coefs = fit_logit(binner.transform(train, selected), train[TARGET])
    dropped_sign = [f for f in selected if f not in final]
    coefs = coefs.merge(iv[["feature", "iv"]], on="feature", how="left")
    logger.info("model: %d features %s", len(final), final)

    woe = binner.transform(df, final)
    df["pd"] = predict_pd(model, woe)
    df["score"] = pd_to_score(df["pd"])
    for name in SPLITS:
        woe[df["split"] == name].astype("float32").assign(bad=df.loc[df["split"] == name, TARGET]) \
            .to_parquet(processed_dir / f"woe_{name}.parquet")

    # ---- scorecard points -------------------------------------------------------------
    points = scorecard_points(binner, model)
    oot_raw = df[df["split"] == "oot"]
    points_gap = float((points_score(binner, points, oot_raw)
                        - pd_to_score(oot_raw["pd"], clip=False)).abs().max())

    # ---- loss parameters and expected loss --------------------------------------------
    loss_params = calibrate_lgd_ead(train[train[TARGET] == 1])
    logger.info("LGD %.3f, EAD ratio %.3f", loss_params["lgd"], loss_params["ead_ratio"])
    df = _score_frame(df, loss_params["lgd"], loss_params["ead_ratio"])
    by_split = {s: df[df["split"] == s] for s in SPLITS}
    oot = by_split["oot"]

    # ---- validation -------------------------------------------------------------------
    deciles = {s: decile_table(d[TARGET], d["score"], d["pd"]) for s, d in by_split.items()}
    calib_oot = calibration_table(oot[TARGET], oot["pd"])
    cutoffs = cutoff_table(oot)

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sample": sample,
        "data_file": Path(raw_path).name,
        "splits": {s: _split_metrics(d) for s, d in by_split.items()},
        "benchmark_sub_grade_oot": _benchmark_metrics(oot),
        "psi_score_train_to_oot": psi(by_split["train"]["score"], oot["score"]),
        "features": final,
        "n_features": len(final),
        "score_scaling": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO,
                          "factor": factor(), "offset": offset()},
        "loss_params": loss_params,
        "el_backtest": {s: _backtest(d) for s, d in by_split.items()},
        "scorecard_points_max_abs_gap_oot": points_gap,
    }

    # ---- artifacts --------------------------------------------------------------------
    _write_json(metrics, artifacts_dir / "metrics.json")
    _write_json(loss_params, artifacts_dir / "loss_params.json")
    _write_json({**summary, "candidate_features": candidates,
                 "dropped_high_missing": dropped_missing, "selection": selection,
                 "dropped_wrong_sign": dropped_sign, "final_features": final,
                 "suspicious_iv": selection["suspicious_iv"]},
                artifacts_dir / "data_summary.json")
    binner.to_json(artifacts_dir / "woe_bins.json")
    iv.to_csv(artifacts_dir / "iv_table.csv", index=False)
    binner.bin_table().to_csv(artifacts_dir / "bin_table.csv", index=False)
    points.to_csv(artifacts_dir / "scorecard_points.csv", index=False)
    coefs.to_csv(artifacts_dir / "model_coefficients.csv", index=False)
    for s, table in deciles.items():
        table.to_csv(artifacts_dir / f"decile_table_{s}.csv", index=False)
    calib_oot.to_csv(artifacts_dir / "calibration_oot.csv", index=False)
    cutoffs.to_csv(artifacts_dir / "cutoff_table.csv", index=False)
    scored_oot = oot[SCORED_OOT_COLS].copy()
    float_cols = ["score", "pd", "ead", "el", "realised_loss", "loan_amnt"]
    scored_oot[float_cols] = scored_oot[float_cols].astype("float32")
    scored_oot["bad"] = scored_oot["bad"].astype("int8")
    scored_oot.to_parquet(artifacts_dir / "scored_oot.parquet", index=False)

    # ---- figures and SQL --------------------------------------------------------------
    plots.make_all(figures_dir, iv=iv, splits={s: d[["score", "pd", "bad", "sub_grade"]]
                                               for s, d in by_split.items()},
                   oot_deciles=deciles["oot"], oot_calibration=calib_oot, cutoffs=cutoffs)
    run_sql({"loans_filtered": interim_dir / "loans_filtered.parquet",
             "modelling": interim_dir / "modelling.parquet",
             "scored_oot": artifacts_dir / "scored_oot.parquet"},
            out_dir=artifacts_dir, sql_dir=sql_dir)
    logger.info("pipeline done: OOT AUC %.3f, KS %.3f", metrics["splits"]["oot"]["auc"],
                metrics["splits"]["oot"]["ks"])
    return metrics
