"""Single-applicant scoring from saved artifacts (numpy only) must match the pipeline."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from scorecard.config import RISK_BANDS, SCORE_MAX, SCORE_MIN
from scorecard.cutoff import cutoff_for_approval_rate
from scorecard.predict import (
    ScoringModel,
    input_defaults,
    load_model,
    model_payload,
    required_inputs,
    score_applicant,
)
from scorecard.scoring import factor, offset, risk_band, score_to_pd
from scorecard.woe import fit_bins


@pytest.fixture(scope="module")
def model(synthetic_run):
    return load_model(synthetic_run[1]["artifacts"])


def _defaults(model):
    return dict(model.input_defaults)


def test_valid_applicant(model):
    out = score_applicant(_defaults(model), model)
    assert SCORE_MIN <= out["score"] <= SCORE_MAX
    assert 0 < out["pd"] < 1
    assert out["risk_band"] in [label for _, label in RISK_BANDS]
    assert set(out["woe"]) == set(out["points"]) == set(model.features)
    assert out["expected_loss"] == pytest.approx(out["pd"] * out["lgd"] * out["ead"])
    assert out["ead"] == pytest.approx(out["loan_amnt"] * model.ead_ratio)
    # unrounded points add up to the unclipped score
    assert sum(out["points"].values()) == pytest.approx(out["score_unclipped"])
    assert score_to_pd(out["score_unclipped"]) == pytest.approx(out["pd"])


def test_lgd_override_scales_expected_loss(model):
    base = score_applicant(_defaults(model), model)
    half = score_applicant(_defaults(model), model, lgd=base["lgd"] / 2)
    assert half["expected_loss"] == pytest.approx(base["expected_loss"] / 2)
    with pytest.raises(ValueError, match="LGD"):
        score_applicant(_defaults(model), model, lgd=1.5)


def test_better_fico_gives_higher_score(model):
    low = score_applicant({**_defaults(model), "fico": 620}, model)
    high = score_applicant({**_defaults(model), "fico": 820}, model)
    assert high["score"] > low["score"]
    assert high["pd"] < low["pd"]


@pytest.mark.parametrize("bad, message", [
    ({"annual_inc": -5}, "Annual income cannot be negative"),
    ({"dti": -1}, "Debt-to-income ratio must be between"),
    ({"dti": 150}, "Debt-to-income ratio must be between"),
    ({"fico": 250}, "FICO score at application must be between 300 and 850"),
    ({"fico": 900}, "FICO score at application must be between 300 and 850"),
    ({"loan_amnt": 0}, "Loan amount must be greater than 0"),
    ({"loan_amnt": -100}, "Loan amount must be greater than 0"),
    ({"loan_amnt": None}, "Loan amount is required"),
    ({"dti": "abc"}, "must be a number"),
    ({"dti": float("inf")}, "finite"),
    ({"revol_util": True}, "must be a number"),
    ({"not_a_field": 1}, "Unknown input"),
])
def test_invalid_inputs_raise_friendly_errors(model, bad, message):
    inputs = {**_defaults(model), "annual_inc": 60000.0, **bad}
    with pytest.raises(ValueError, match=message):
        score_applicant(inputs, model)


def test_missing_optional_inputs_use_missing_bin(model):
    out = score_applicant({"loan_amnt": 10000}, model)  # everything else blank
    assert all(math.isfinite(v) for v in out["woe"].values())
    assert SCORE_MIN <= out["score"] <= SCORE_MAX


def test_load_model_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        load_model(tmp_path)


def test_model_json_matches_metrics(synthetic_run, model):
    metrics, dirs = synthetic_run
    saved = json.loads((dirs["artifacts"] / "model.json").read_text())
    assert saved["features"] == metrics["features"] == model.features
    assert saved["loss_params"]["lgd"] == pytest.approx(metrics["loss_params"]["lgd"])
    assert all(b < 0 for b in saved["coefficients"].values())
    coefs = pd.read_csv(dirs["artifacts"] / "model_coefficients.csv").set_index("feature")
    assert saved["intercept"] == pytest.approx(coefs.loc["intercept", "coefficient"])
    assert set(required_inputs(model.features)) == set(model.inputs)
    assert set(saved["input_defaults"]) <= set(model.inputs)


def _row_inputs(row: pd.Series, inputs: list[str]) -> dict:
    source = {"fico": "fico_mid"}
    return {k: row[source.get(k, k)] for k in inputs}


def test_matches_pipeline_score_row_by_row(synthetic_run, model):
    """score_applicant on a row == the pipeline's score/PD/EL for that row."""
    _, dirs = synthetic_run
    scored = pd.read_parquet(dirs["artifacts"] / "scored_oot.parquet")
    modelling = pd.read_parquet(dirs["interim"] / "modelling.parquet")
    oot = modelling[modelling["split"] == "oot"].reset_index(drop=True)
    assert len(oot) == len(scored)  # same rows, same order
    rng = np.random.default_rng(0)
    for i in rng.choice(len(scored), size=60, replace=False):
        out = score_applicant(_row_inputs(oot.loc[i], model.inputs), model)
        assert out["pd"] == pytest.approx(scored.loc[i, "pd"], abs=1e-5)
        assert out["score"] == pytest.approx(scored.loc[i, "score"], abs=1e-2)
        assert out["expected_loss"] == pytest.approx(scored.loc[i, "el"], rel=1e-4, abs=1e-3)


@pytest.mark.parametrize("score, band", [
    (850, "Very Low"), (700, "Very Low"), (699, "Low"), (650, "Low"), (649, "Medium"),
    (600, "Medium"), (599, "High"), (550, "High"), (549, "Very High"), (300, "Very High"),
])
def test_risk_band_boundaries(score, band):
    assert risk_band(score) == band


def test_risk_bands_are_descending_and_cover_the_range():
    minimums = [m for m, _ in RISK_BANDS]
    assert minimums == sorted(minimums, reverse=True)
    assert minimums[-1] <= SCORE_MIN


def test_cutoff_for_approval_rate():
    table = pd.DataFrame({"cutoff": [300, 400, 500, 600], "approval_rate": [1.0, 0.9, 0.7, 0.2]})
    assert cutoff_for_approval_rate(table, 0.8) == 400
    assert cutoff_for_approval_rate(table, 1.1) == 300  # unreachable -> lowest cut-off


def test_derived_and_categorical_features():
    """Ratio features are computed from plain inputs; unseen categories fall into Other."""
    rng = np.random.default_rng(1)
    n = 4000
    income = rng.lognormal(11, 0.5, n)
    loan = rng.integers(10, 400, n) * 100.0
    purpose = rng.choice(["debt", "card", "moving", "rare"], n, p=[0.5, 0.3, 0.195, 0.005])
    p = 1 / (1 + np.exp(-(-2 + 3 * loan / income + 0.5 * (purpose == "card"))))
    y = (rng.random(n) < p).astype(float)
    df = pd.DataFrame({"loan_to_income": loan / income, "purpose": purpose})
    bins = {"loan_to_income": fit_bins(df["loan_to_income"], y, "numeric"),
            "purpose": fit_bins(df["purpose"], y, "categorical")}
    features = ["loan_to_income", "purpose"]
    scaling = {"factor": factor(), "offset": offset()}
    payload = model_payload(features, -1.5, [-1.0, -0.8], scaling,
                            {"lgd": 0.9, "ead_ratio": 0.6}, {})
    model = ScoringModel(features=features, intercept=payload["intercept"],
                         coefficients=payload["coefficients"], bins=bins, scaling=scaling,
                         lgd=0.9, ead_ratio=0.6)
    assert model.inputs == ["loan_amnt", "annual_inc", "purpose"]
    assert "Other" in model.categories("purpose")
    low = score_applicant({"loan_amnt": 2000, "annual_inc": 100000, "purpose": "debt"}, model)
    high = score_applicant({"loan_amnt": 30000, "annual_inc": 30000, "purpose": "debt"}, model)
    assert low["score"] > high["score"]
    unseen = score_applicant({"loan_amnt": 2000, "annual_inc": 100000, "purpose": "zzz"}, model)
    rare = score_applicant({"loan_amnt": 2000, "annual_inc": 100000, "purpose": "rare"}, model)
    assert unseen["bin"]["purpose"] == rare["bin"]["purpose"]
    # no income -> ratio missing -> still scores
    missing = score_applicant({"loan_amnt": 2000, "purpose": "card"}, model)
    assert sum(missing["points"].values()) == pytest.approx(missing["score_unclipped"])
    with pytest.raises(ValueError, match="cannot be negative"):
        score_applicant({"loan_amnt": 2000, "annual_inc": -1, "purpose": "card"}, model)


def test_input_defaults_uses_median_and_mode():
    train = pd.DataFrame({"fico_mid": [650.0, 700.0, 750.0], "loan_amnt": [1.0, 2.0, 9.0],
                          "annual_inc": [1, 2, 3], "purpose": ["a", "b", "b"]})
    got = input_defaults(train, ["fico_mid", "loan_to_income", "purpose"])
    assert got == {"loan_amnt": 2.0, "fico": 700.0, "annual_inc": 2.0, "purpose": "b"}
