"""Credit Risk Scorecard dashboard. Run: ``streamlit run app/streamlit_app.py``.

Reads only the small files in ``artifacts/`` (override with ``SCORECARD_ARTIFACTS_DIR``).
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for _path in (APP_DIR.parent / "src", APP_DIR):  # Streamlit Cloud installs requirements only
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st
from scorecard_ui import applicant, method, overview, simulator, validation

st.set_page_config(page_title="Credit Risk Scorecard", page_icon=":material/account_balance:",
                   layout="wide")

PAGES = [
    st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview",
            default=True),
    st.Page(simulator.render, title="Cut-off Simulator", icon=":material/tune:",
            url_path="cutoff-simulator"),
    st.Page(applicant.render, title="Score an Applicant", icon=":material/person_search:",
            url_path="score-applicant"),
    st.Page(validation.render, title="Model & Validation", icon=":material/verified:",
            url_path="validation"),
    st.Page(method.render, title="Data & Method", icon=":material/account_tree:",
            url_path="data-method"),
]
st.navigation(PAGES).run()
