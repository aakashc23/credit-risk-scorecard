"""Page 3: score a single applicant with the saved scorecard (numpy only, no scikit-learn)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scorecard import config
from scorecard.predict import INPUT_SPECS, ScoringModel, feature_label, input_label, score_applicant
from scorecard.scoring import score_to_pd
from scorecard_ui import common as c

COUNT_UNITS = {"count", "months", "years"}


def _field(model: ScoringModel, key: str):
    """One input widget with a sensible default (median / most common value in training)."""
    spec = INPUT_SPECS.get(key, {})
    default = model.input_defaults.get(key)
    label, help_ = input_label(key), spec.get("help") or None
    if key in config.CATEGORICAL_FEATURES:
        options = model.categories(key)
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, key=f"in_{key}", help=help_)
    unit = spec.get("unit", "")
    if unit:
        label = f"{label} ({unit})"
    if unit == "$":
        step, fmt = (500.0 if key == "loan_amnt" else 1000.0), "%.0f"
    elif unit == "%":
        step, fmt = 1.0, "%.1f"
    else:
        step, fmt = 1.0, "%.0f"
    return st.number_input(label, value=float(default if default is not None else 0.0),
                           step=step, format=fmt, key=f"in_{key}", help=help_)


def _points_chart(result: dict) -> None:
    rows = sorted(result["points_vs_avg"].items(), key=lambda kv: kv[1])
    labels = [feature_label(f) for f, _ in rows]
    values = [v for _, v in rows]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[c.BLUE if v >= 0 else c.VERMILLION for v in values],
        text=[f"{v:+.0f}" for v in values], textposition="outside", cliponaxis=False,
        customdata=[[result["bin"][f]] for f, _ in rows],
        hovertemplate="%{y}<br>Bin: %{customdata[0]}<br>%{x:+.1f} points<extra></extra>"))
    fig.update_layout(xaxis_title="Points vs a typical applicant", hovermode="closest")
    fig.add_vline(x=0, line_color="#1F2937", line_width=1)
    c.show(c.style(fig, height=max(240, 48 * len(rows) + 80), legend=False))
    best, worst = rows[-1], rows[0]
    parts = []
    if best[1] > 0:
        parts.append(f"{feature_label(best[0])} helps most ({best[1]:+.0f} points, "
                     f"{result['bin'][best[0]]})")
    if worst[1] < 0:
        parts.append(f"{feature_label(worst[0])} costs most ({worst[1]:+.0f} points, "
                     f"{result['bin'][worst[0]]})")
    reference = sum(result["points"].values()) - sum(result["points_vs_avg"].values())
    c.takeaway(("; ".join(parts) if parts else "This applicant is close to typical on every "
                "feature") + f". A typical applicant scores about {reference:.0f}.")


def render() -> None:
    c.header("Score an Applicant",
             "Enter application details to get a score, a probability of default and an "
             "expected loss. Scored live from the saved scorecard (bins + coefficients), "
             "with no model library and no data leaving the app.")
    if not c.need(config.MODEL_FILENAME, "woe_bins.json"):
        return
    model = c.get_model()

    form, out = st.columns([5, 6], gap="large")
    with form:
        st.subheader("Applicant")
        inputs = {key: _field(model, key) for key in model.inputs}
        optional = [k for k in model.inputs if k != "loan_amnt"]
        unknown = st.multiselect("Don't know these? Score them as missing", optional,
                                 format_func=input_label, key="in_unknown",
                                 help="Missing values fall in their own bin, learned from "
                                      "applicants who left the field empty.")
        inputs.update(dict.fromkeys(unknown))

    with out:
        st.subheader("Result")
        try:
            result = score_applicant(inputs, model)
        except ValueError as err:
            st.error(str(err), icon=":material/error:")
            return
        cutoff = c.current_cutoff() if c.load("cutoff_table.csv") is not None else None
        approve = cutoff is not None and result["score"] >= cutoff
        c.metric_row([
            ("Score", f"{result['score']:.0f}", f"range {config.SCORE_MIN}-{config.SCORE_MAX}"),
            ("Probability of default", c.pct(result["pd"]), "chance the loan goes bad"),
            ("Risk band", result["risk_band"]),
        ])
        if cutoff is not None:
            badge = (":green-badge[:material/check: Approve]" if approve
                     else ":red-badge[:material/close: Decline]")
            st.markdown(f"**Decision at the simulator cut-off ({cutoff}):** {badge}  \n"
                        f"Score {result['score']:.0f} is "
                        f"{'at or above' if approve else 'below'} the cut-off. Change it on the "
                        "Cut-off Simulator page.")

        st.markdown("**Expected loss = PD x LGD x EAD**")
        st.markdown(f"{c.pct(result['pd'], 2)} x {c.pct(result['lgd'])} x "
                    f"{c.usd(result['ead'])} = **{c.usd(result['expected_loss'])}**")
        st.caption(f"EAD = loan amount {c.usd(result['loan_amnt'])} x EAD ratio "
                   f"{c.pct(result['ead_ratio'])}. LGD and the EAD ratio are portfolio-wide "
                   "estimates from past defaults, not specific to this applicant.")

    st.subheader("What pushed the score up or down?")
    _points_chart(result)
    with st.expander("Scorecard detail for this applicant"):
        detail = pd.DataFrame({
            "Feature": [feature_label(f) for f in model.features],
            "Applicant falls in": [result["bin"][f] for f in model.features],
            "WOE": [result["woe"][f] for f in model.features],
            "Points": [result["points"][f] for f in model.features]})
        st.dataframe(detail, hide_index=True, column_config={
            "WOE": st.column_config.NumberColumn(format="%.3f"),
            "Points": st.column_config.NumberColumn(format="%.1f")})
        st.caption(f"Points add up to the score ({result['score_unclipped']:.1f} before "
                   f"clipping to {config.SCORE_MIN}-{config.SCORE_MAX}); the score maps back "
                   f"to PD = {c.pct(float(score_to_pd(result['score_unclipped'])), 2)}.")
    bands = ", ".join(f"{label} ({m}+)" for m, label in config.RISK_BANDS[:-1])
    bands += f", {config.RISK_BANDS[-1][1]} (below {config.RISK_BANDS[-2][0]})"
    st.caption(f"Risk bands by score: {bands}. Educational demo on historical data; not a "
               "lending decision.")
