"""Score one applicant from the saved artifacts, with numpy/pandas only (no scikit-learn).

The pipeline saves ``woe_bins.json`` (the fitted WOE bins) and ``model.json`` (intercept,
coefficients, score scaling, loss parameters). Scoring an applicant is then:

    WOE_j  = lookup of the applicant's value in feature j's bins          (woe.py)
    logit  = intercept + sum_j beta_j * WOE_j ;  PD = 1 / (1 + exp(-logit))
    score  = offset + factor * ln((1 - PD) / PD), clipped to 300..850     (scoring.py)
    EL     = PD * LGD * (loan_amnt * EAD_ratio)                           (loss.py)

``score_applicant`` takes friendly *inputs* (``fico``, ``loan_amnt``, ``annual_inc`` ...); the
ratio features the model uses (``loan_to_income`` ...) are derived here exactly as in
``features.engineer``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard.config import ARTIFACTS, CATEGORICAL_FEATURES, MODEL_FILENAME
from scorecard.loss import expected_loss
from scorecard.scoring import pd_to_score, risk_band
from scorecard.woe import MISSING, OTHER, label_feature, transform_feature

# Model feature -> the plain inputs it is computed from (others take the input of the same name).
FEATURE_INPUTS: dict[str, list[str]] = {
    "fico_mid": ["fico"],
    "loan_to_income": ["loan_amnt", "annual_inc"],
    "revol_bal_to_income": ["revol_bal", "annual_inc"],
}
REQUIRED_INPUTS = ["loan_amnt"]  # always needed: exposure (EAD) scales with the loan amount


def _spec(label: str, lo: float | None = 0, hi: float | None = None, unit: str = "",
          help: str = "", lo_exclusive: bool = False) -> dict:
    return {"label": label, "lo": lo, "hi": hi, "unit": unit, "help": help,
            "lo_exclusive": lo_exclusive}


# Friendly label and valid range for every input the scorecard can ever use.
INPUT_SPECS: dict[str, dict] = {
    "loan_amnt": _spec("Loan amount", 0, None, "$", "Requested amount.", lo_exclusive=True),
    "annual_inc": _spec("Annual income", 0, None, "$", "Self-reported gross annual income."),
    "fico": _spec("FICO score at application", 300, 850, "", "Credit bureau score, 300-850."),
    "dti": _spec("Debt-to-income ratio", 0, 100, "%", "Monthly debt payments / monthly income."),
    "emp_length_years": _spec("Employment length", 0, 10, "years", "10 means 10 or more years."),
    "credit_history_months": _spec("Credit history length", 0, None, "months",
                                   "Months since the first credit line was opened."),
    "revol_bal": _spec("Revolving balance", 0, None, "$", "Total balance on revolving accounts."),
    "revol_util": _spec("Revolving utilisation", 0, 200, "%", "Share of revolving credit used."),
    "bc_util": _spec("Bankcard utilisation", 0, 200, "%"),
    "percent_bc_gt_75": _spec("Bankcards above 75% of limit", 0, 100, "%"),
    "delinq_2yrs": _spec("Delinquencies in last 2 years", 0, None, "count"),
    "inq_last_6mths": _spec("Credit inquiries in last 6 months", 0, None, "count"),
    "mths_since_last_delinq": _spec("Months since last delinquency", 0, None, "months",
                                    "Leave empty if never."),
    "mths_since_last_record": _spec("Months since last public record", 0, None, "months",
                                    "Leave empty if never."),
    "mths_since_recent_inq": _spec("Months since most recent inquiry", 0, None, "months"),
    "open_acc": _spec("Open credit lines", 0, None, "count"),
    "total_acc": _spec("Total credit lines ever", 0, None, "count"),
    "pub_rec": _spec("Derogatory public records", 0, None, "count"),
    "mort_acc": _spec("Mortgage accounts", 0, None, "count"),
    "pub_rec_bankruptcies": _spec("Public-record bankruptcies", 0, None, "count"),
    "acc_open_past_24mths": _spec("Accounts opened in last 24 months", 0, None, "count"),
    "num_actv_rev_tl": _spec("Active revolving accounts", 0, None, "count"),
    "tot_cur_bal": _spec("Total current balance, all accounts", 0, None, "$"),
    "total_rev_hi_lim": _spec("Total revolving credit limit", 0, None, "$"),
    "avg_cur_bal": _spec("Average current balance per account", 0, None, "$"),
    "mo_sin_rcnt_tl": _spec("Months since most recent account opened", 0, None, "months"),
    "num_tl_op_past_12m": _spec("Accounts opened in last 12 months", 0, None, "count"),
    "tax_liens": _spec("Tax liens", 0, None, "count"),
    "collections_12_mths_ex_med": _spec("Collections in last 12 months", 0, None, "count"),
    "home_ownership": _spec("Home ownership", None, None),
    "verification_status": _spec("Income verification", None, None),
    "purpose": _spec("Loan purpose", None, None),
    "application_type": _spec("Application type", None, None),
}


def input_label(key: str) -> str:
    return INPUT_SPECS.get(key, {}).get("label", key.replace("_", " ").capitalize())


FEATURE_LABELS = {
    "fico_mid": "FICO score",
    "loan_to_income": "Loan amount / annual income",
    "revol_bal_to_income": "Revolving balance / annual income",
}


def feature_label(feature: str) -> str:
    """Friendly name of a model feature (derived ones differ from the input names)."""
    return FEATURE_LABELS.get(feature) or input_label(feature)


# --------------------------------------------------------------------------- model object
@dataclass
class ScoringModel:
    """Everything needed to score an applicant, loaded from JSON."""

    features: list[str]
    intercept: float
    coefficients: dict[str, float]
    bins: dict[str, dict]
    scaling: dict
    lgd: float
    ead_ratio: float
    input_defaults: dict = field(default_factory=dict)

    @property
    def inputs(self) -> list[str]:
        """Inputs the form must offer: loan amount first, then by model-feature order."""
        return required_inputs(self.features)

    def categories(self, feature: str) -> list[str]:
        """Selectable values of a categorical feature (as seen in training)."""
        spec = self.bins[feature]
        found = [c for b in spec["bins"] for c in b["categories"] if c != MISSING]
        return sorted(found, key=lambda c: (c == OTHER, c))


def required_inputs(features: list[str]) -> list[str]:
    """Input keys needed to compute ``features`` (loan amount first, no duplicates)."""
    keys = list(REQUIRED_INPUTS)
    for feat in features:
        keys += FEATURE_INPUTS.get(feat, [feat])
    return list(dict.fromkeys(keys))


def load_model(artifacts_dir: Path | str = ARTIFACTS) -> ScoringModel:
    """Read ``model.json`` and ``woe_bins.json``; raises FileNotFoundError if not yet built."""
    root = Path(artifacts_dir)
    model_path, bins_path = root / MODEL_FILENAME, root / "woe_bins.json"
    for path in (model_path, bins_path):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found in {root}. "
                                    "Run `python scripts/run_pipeline.py` to create it.")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    bins = json.loads(bins_path.read_text(encoding="utf-8"))["bins"]
    features = list(model["features"])
    return ScoringModel(
        features=features, intercept=float(model["intercept"]),
        coefficients={f: float(model["coefficients"][f]) for f in features},
        bins={f: bins[f] for f in features}, scaling=model["score_scaling"],
        lgd=float(model["loss_params"]["lgd"]), ead_ratio=float(model["loss_params"]["ead_ratio"]),
        input_defaults=model.get("input_defaults", {}))


@lru_cache(maxsize=4)
def _cached_model(path: str) -> ScoringModel:
    return load_model(path)


# --------------------------------------------------------------------------- building model.json
def input_defaults(train: pd.DataFrame, features: list[str]) -> dict:
    """Default form values: median (numeric) / most common value (categorical) in train."""
    source = {"fico": "fico_mid"}
    out: dict = {}
    for key in required_inputs(features):
        col = source.get(key, key)
        if col not in train.columns:
            continue
        values = train[col].dropna()
        if values.empty:
            continue
        if key in CATEGORICAL_FEATURES:
            out[key] = str(values.mode().iloc[0])
        else:
            out[key] = float(values.median())
    return out


def model_payload(features: list[str], intercept: float, coefficients: list[float],
                  scaling: dict, loss_params: dict, defaults: dict) -> dict:
    """The JSON document written to ``artifacts/model.json``."""
    return {
        "features": list(features), "intercept": float(intercept),
        "coefficients": dict(zip(features, map(float, coefficients), strict=True)),
        "score_scaling": scaling,
        "loss_params": {"lgd": float(loss_params["lgd"]),
                        "ead_ratio": float(loss_params["ead_ratio"])},
        "input_defaults": defaults,
    }


# --------------------------------------------------------------------------- validation
def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, float) and math.isnan(value))


def _number(key: str, value) -> float:
    """Validated float for a numeric input (NaN when blank). Raises ValueError if invalid."""
    spec, label = INPUT_SPECS[key], input_label(key)
    if _is_blank(value):
        return math.nan
    try:
        if isinstance(value, bool):
            raise TypeError(value)
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number (got {value!r}).") from None
    if not math.isfinite(x):
        raise ValueError(f"{label} must be a finite number.")
    lo, hi = spec["lo"], spec["hi"]
    unit = "%" if spec["unit"] == "%" else ""
    below = lo is not None and (x <= lo if spec["lo_exclusive"] else x < lo)
    if below and spec["lo_exclusive"]:
        raise ValueError(f"{label} must be greater than {lo:g}.")
    if below and hi is None and lo == 0:
        raise ValueError(f"{label} cannot be negative.")
    if below or (hi is not None and x > hi):
        if hi is None:
            raise ValueError(f"{label} must be at least {lo:g}.")
        raise ValueError(f"{label} must be between {lo:g}{unit} and {hi:g}{unit}.")
    return x


def _validate_inputs(inputs: dict) -> dict:
    """Clean copy of ``inputs``: numbers as floats (NaN = missing), categories as strings."""
    if "loan_amnt" not in inputs or _is_blank(inputs["loan_amnt"]):
        raise ValueError("Loan amount is required.")
    clean: dict = {}
    for key, value in inputs.items():
        if key not in INPUT_SPECS:
            raise ValueError(f"Unknown input {key!r}.")
        if key in CATEGORICAL_FEATURES:
            clean[key] = None if _is_blank(value) else str(value)
        else:
            clean[key] = _number(key, value)
    return clean


# --------------------------------------------------------------------------- scoring
def _feature_value(feature: str, clean: dict):
    """Model-feature value from the cleaned inputs (mirrors ``features.engineer``)."""
    get = clean.get
    if feature == "fico_mid":
        return get("fico", math.nan)
    if feature in ("loan_to_income", "revol_bal_to_income"):
        num = get("loan_amnt" if feature == "loan_to_income" else "revol_bal", math.nan)
        income = get("annual_inc", math.nan)
        return num / income if income and income > 0 else math.nan
    return get(feature, None if feature in CATEGORICAL_FEATURES else math.nan)


def _points(model: ScoringModel, feature: str, woe):
    """Continuous scorecard points of a WOE value: -(beta*woe + alpha/n)*factor + offset/n."""
    n = len(model.features)
    s = model.scaling
    return -(model.coefficients[feature] * woe + model.intercept / n) * s["factor"] \
        + s["offset"] / n


def _average_points(model: ScoringModel, feature: str) -> float:
    """Training-population average points for a feature (weights = bin sizes)."""
    bins = model.bins[feature]["bins"]
    weights = np.array([b["n"] for b in bins], float)
    woes = np.array([b["woe"] for b in bins], float)
    return float(np.average(_points(model, feature, woes), weights=weights))


def score_applicant(inputs: dict, model: ScoringModel | None = None,
                    lgd: float | None = None) -> dict:
    """Score one applicant. Blank optional inputs go to the feature's "Missing" bin.

    Raises ``ValueError`` with a plain-English message for invalid inputs. ``lgd`` overrides
    the calibrated LGD (what-if). Returns pd, score, risk_band, ead, expected_loss, lgd plus
    per-feature ``woe``, ``bin``, ``points`` and ``points_vs_avg`` dicts.
    """
    model = model or _cached_model(str(ARTIFACTS))
    clean = _validate_inputs(inputs)
    if lgd is not None and not 0 <= lgd <= 1:
        raise ValueError("LGD must be between 0 and 1.")
    lgd_used = model.lgd if lgd is None else float(lgd)

    woe, bin_label = {}, {}
    for feat in model.features:
        value = pd.Series([_feature_value(feat, clean)])
        woe[feat] = float(transform_feature(model.bins[feat], value)[0])
        bin_label[feat] = str(label_feature(model.bins[feat], value).iloc[0])

    logit = model.intercept + sum(model.coefficients[f] * woe[f] for f in model.features)
    p_bad = float(1 / (1 + math.exp(-logit)))
    score = float(pd_to_score(p_bad))
    points = {f: float(_points(model, f, woe[f])) for f in model.features}
    average = {f: _average_points(model, f) for f in model.features}
    loan = clean["loan_amnt"]
    return {
        "pd": p_bad, "score": score, "risk_band": risk_band(score),
        "score_unclipped": float(pd_to_score(p_bad, clip=False)),
        "woe": woe, "bin": bin_label, "points": points,
        "points_vs_avg": {f: points[f] - average[f] for f in model.features},
        "lgd": lgd_used, "ead_ratio": model.ead_ratio, "loan_amnt": loan,
        "ead": loan * model.ead_ratio,
        "expected_loss": float(expected_loss(p_bad, loan, lgd_used, model.ead_ratio)),
    }
