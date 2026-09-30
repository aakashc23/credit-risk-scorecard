"""Streamlit dashboard: every page renders, the simulator reacts, the applicant form validates.

Runs against the synthetic artifacts from the ``synthetic_run`` fixture (no real data).
"""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from scorecard import config
from scorecard.cutoff import cutoff_for_approval_rate, cutoff_summary

ROOT = Path(__file__).resolve().parents[1]
APP_DIR, SRC_DIR = ROOT / "app", ROOT / "src"
VIEWS = ["overview", "simulator", "applicant", "validation", "method"]


def _view_script(view: str, app_dir: str, src_dir: str):
    """Script body run by AppTest: render one page module."""
    import importlib
    import sys

    for path in (src_dir, app_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    importlib.import_module(f"scorecard_ui.{view}").render()


def open_view(view: str, **state) -> AppTest:
    at = AppTest.from_function(_view_script, args=(view, str(APP_DIR), str(SRC_DIR)),
                               default_timeout=60)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def metrics(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def texts(at: AppTest) -> str:
    return " ".join(e.value for e in [*at.markdown, *at.caption, *at.info, *at.warning,
                                      *at.error])


@pytest.fixture(autouse=True)
def artifacts_env(synthetic_run, monkeypatch):
    monkeypatch.setenv("SCORECARD_ARTIFACTS_DIR", str(synthetic_run[1]["artifacts"]))
    return synthetic_run[1]["artifacts"]


# --------------------------------------------------------------------------- pages render
def test_entry_point_runs():
    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert "AUC" in metrics(at)


@pytest.mark.parametrize("view", VIEWS)
def test_page_renders_without_exceptions(view):
    at = open_view(view)
    assert not at.exception, [e.value for e in at.exception]
    assert not any("Missing artifact" in w.value for w in at.warning)
    assert len(at.title) == 1


def test_overview_tiles_come_from_metrics_json(synthetic_run):
    tiles = metrics(open_view("overview"))
    oot = synthetic_run[0]["splits"]["oot"]
    assert tiles["AUC"] == f"{oot['auc']:.3f}"
    assert tiles["OOT loans"] == f"{oot['n']:,}"


@pytest.mark.parametrize("view, minimum", [("simulator", 3), ("validation", 6)])
def test_charts_carry_a_computed_takeaway(view, minimum):
    at = open_view(view)
    assert sum("Takeaway" in m.value for m in at.markdown) >= minimum


def test_method_page_shows_sql_tables_and_code():
    at = open_view("method")
    assert len(at.dataframe) >= 3
    assert any("SELECT" in c.value for c in at.code)


# --------------------------------------------------------------------------- simulator
def test_simulator_default_is_data_driven(artifacts_env):
    table = pd.read_csv(artifacts_env / "cutoff_table.csv")
    expected = int(cutoff_for_approval_rate(table, config.DEFAULT_APPROVAL_TARGET))
    at = open_view("simulator")
    assert at.slider(key="_cutoff_widget").value == expected
    rate = float(metrics(at)["Approval rate"].rstrip("%")) / 100
    assert rate >= config.DEFAULT_APPROVAL_TARGET - 0.005


def test_simulator_slider_updates_tiles(artifacts_env):
    scored = pd.read_parquet(artifacts_env / "scored_oot.parquet")
    at = open_view("simulator")
    before = metrics(at)
    at.slider(key="_cutoff_widget").set_value(700).run()
    assert not at.exception
    after = metrics(at)
    want = cutoff_summary(scored, 700)
    assert after["Approved loans"] == f"{want['n_approved']:,}"
    assert after["Rejected loans"] == f"{want['n_rejected']:,}"
    assert after["Approval rate"] == f"{want['approval_rate'] * 100:.1f}%"
    assert after["Approval rate"] != before["Approval rate"]
    assert after["Expected bad rate"] != before["Expected bad rate"]
    assert at.session_state["cutoff"] == 700
    assert any("At a cut-off of **700**" in i.value for i in at.info)


def test_simulator_lgd_whatif_changes_expected_but_not_realised_loss(synthetic_run):
    at = open_view("simulator")
    before = metrics(at)
    lgd = synthetic_run[0]["loss_params"]["lgd"]
    at.number_input(key="_lgd_widget").set_value(round(lgd / 2, 4)).run()
    after = metrics(at)
    assert after["Expected loss"] != before["Expected loss"]
    assert after["Realised loss"] == before["Realised loss"]


def test_simulator_extreme_cutoffs_do_not_crash():
    for cutoff in (config.SCORE_MIN, config.SCORE_MAX):
        at = open_view("simulator", _cutoff_widget=cutoff)
        assert not at.exception


# --------------------------------------------------------------------------- applicant
def test_applicant_valid_input_scores():
    at = open_view("applicant")
    assert not at.exception and not at.error
    tiles = metrics(at)
    assert config.SCORE_MIN <= int(tiles["Score"]) <= config.SCORE_MAX
    assert tiles["Risk band"] in [label for _, label in config.RISK_BANDS]
    assert any("Expected loss = PD x LGD x EAD" in m.value for m in at.markdown)


def test_applicant_better_fico_scores_higher():
    at = open_view("applicant")
    at.number_input(key="in_fico").set_value(620).run()
    low = int(metrics(at)["Score"])
    at.number_input(key="in_fico").set_value(820).run()
    assert int(metrics(at)["Score"]) > low


@pytest.mark.parametrize("key, value, message", [
    ("in_fico", 250, "FICO"),
    ("in_dti", -5, "Debt-to-income"),
    ("in_loan_amnt", 0, "Loan amount"),
])
def test_applicant_invalid_input_shows_friendly_error(key, value, message):
    at = open_view("applicant")
    at.number_input(key=key).set_value(value).run()
    assert not at.exception  # no traceback
    assert len(at.error) == 1 and message in at.error[0].value
    assert "Score" not in metrics(at)


def test_applicant_decision_follows_simulator_cutoff():
    assert "Approve" in texts(open_view("applicant", cutoff=config.SCORE_MIN))
    declined = texts(open_view("applicant", cutoff=config.SCORE_MAX + 5))
    assert "Decline" in declined


def test_applicant_unknown_fields_are_scored_as_missing():
    at = open_view("applicant")
    optional = [o for o in at.multiselect(key="in_unknown").options if o != "loan_amnt"]
    at.multiselect(key="in_unknown").set_value(optional).run()
    assert not at.exception and not at.error
    assert "Score" in metrics(at)


# --------------------------------------------------------------------------- missing data
@pytest.mark.parametrize("view", VIEWS)
def test_missing_artifacts_show_a_warning_not_a_crash(view, tmp_path, monkeypatch):
    monkeypatch.setenv("SCORECARD_ARTIFACTS_DIR", str(tmp_path))
    at = open_view(view)
    assert not at.exception, [e.value for e in at.exception]
    assert any("run_pipeline" in w.value for w in at.warning)
