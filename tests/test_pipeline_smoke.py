"""End-to-end run on synthetic raw data, writing only into tmp dirs."""
import json

import pandas as pd
import pytest

from scorecard.config import LEAKAGE_COLUMNS
from scorecard.pipeline import run

METRIC_KEYS = {"generated_at", "sample", "data_file", "splits", "benchmark_sub_grade_oot",
               "psi_score_train_to_oot", "features", "score_scaling", "loss_params", "el_backtest"}
ARTIFACT_FILES = [
    "metrics.json", "iv_table.csv", "scorecard_points.csv", "model_coefficients.csv",
    "decile_table_train.csv", "decile_table_test.csv", "decile_table_oot.csv",
    "calibration_oot.csv", "cutoff_table.csv", "scored_oot.parquet", "loss_params.json",
    "woe_bins.json", "data_summary.json", "sql_status_mix.csv",
    "sql_portfolio_by_year_grade.csv", "sql_score_band_summary.csv",
]
FIGURE_FILES = ["iv_bar", "score_distribution", "roc", "ks_curve", "decile_bad_rate",
                "calibration", "cutoff_tradeoff", "score_vs_grade"]


@pytest.fixture(scope="module")
def outputs(raw_csv, tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    dirs = {k: root / k for k in ["artifacts", "interim", "processed", "figures"]}
    metrics = run(raw_csv, artifacts_dir=dirs["artifacts"], interim_dir=dirs["interim"],
                  processed_dir=dirs["processed"], figures_dir=dirs["figures"])
    return metrics, dirs


def test_artifacts_written(outputs):
    _, dirs = outputs
    for name in ARTIFACT_FILES:
        assert (dirs["artifacts"] / name).stat().st_size > 0, name
    for name in FIGURE_FILES:
        assert (dirs["figures"] / f"{name}.png").stat().st_size > 0, name
    assert (dirs["interim"] / "modelling.parquet").exists()


def test_metrics_keys_and_provenance(outputs):
    metrics, dirs = outputs
    on_disk = json.loads((dirs["artifacts"] / "metrics.json").read_text())
    assert METRIC_KEYS <= set(on_disk)
    assert on_disk["sample"] is None
    assert on_disk["data_file"].endswith(".csv.gz")
    assert set(on_disk["splits"]) == {"train", "test", "oot"}
    assert {"auc", "gini", "ks"} <= set(on_disk["splits"]["oot"])
    assert metrics["features"] == on_disk["features"]


def test_all_coefficients_negative(outputs):
    _, dirs = outputs
    coefs = pd.read_csv(dirs["artifacts"] / "model_coefficients.csv")
    slopes = coefs[coefs["feature"] != "intercept"]
    assert len(slopes) >= 3
    assert (slopes["coefficient"] < 0).all()


def test_model_has_signal_on_synthetic_data(outputs):
    metrics, _ = outputs
    for split in ("train", "test", "oot"):
        assert metrics["splits"][split]["auc"] > 0.6
    assert metrics["benchmark_sub_grade_oot"]["auc"] > 0.5


def test_no_leakage_in_model_features(outputs):
    metrics, dirs = outputs
    assert not set(metrics["features"]) & LEAKAGE_COLUMNS
    bins = json.loads((dirs["artifacts"] / "woe_bins.json").read_text())["bins"]
    assert not set(bins) & LEAKAGE_COLUMNS


def test_scored_oot_and_cutoff_table(outputs):
    _, dirs = outputs
    scored = pd.read_parquet(dirs["artifacts"] / "scored_oot.parquet")
    assert {"score", "pd", "ead", "el", "bad", "realised_loss", "loan_amnt", "sub_grade",
            "issue_d"} <= set(scored.columns)
    assert scored["score"].between(300, 850).all()
    cutoffs = pd.read_csv(dirs["artifacts"] / "cutoff_table.csv")
    assert cutoffs["n_total"].iloc[0] == len(scored)
    assert cutoffs["approval_rate"].iloc[0] == 1.0
    assert cutoffs["approval_rate"].is_monotonic_decreasing


def test_points_reproduce_score(outputs):
    metrics, _ = outputs
    assert metrics["scorecard_points_max_abs_gap_oot"] <= metrics["n_features"]


def test_sample_run_records_sample(raw_csv, tmp_path):
    metrics = run(raw_csv, sample=6000, artifacts_dir=tmp_path / "a", interim_dir=tmp_path / "i",
                  processed_dir=tmp_path / "p", figures_dir=tmp_path / "f")
    assert metrics["sample"] == 6000
    assert metrics["splits"]["oot"]["n"] > 0


def test_empty_split_gives_clear_error(raw_csv, tmp_path):
    with pytest.raises(ValueError, match="too little"):
        run(raw_csv, sample=40, artifacts_dir=tmp_path / "a", interim_dir=tmp_path / "i",
            processed_dir=tmp_path / "p", figures_dir=tmp_path / "f")
