"""Page 4: how well does the model work? Metrics, deciles, calibration, PSI, benchmark, IV."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scorecard import config
from scorecard.predict import feature_label
from scorecard_ui import common as c

SPLIT_NAMES = {"train": "Train", "test": "Test (in-time)", "oot": "Out-of-time"}


def _metrics_table(metrics: dict) -> None:
    splits = metrics["splits"]
    table = pd.DataFrame({
        "Sample": [SPLIT_NAMES.get(k, k) for k in splits],
        "Loans": [v["n"] for v in splits.values()],
        "Bad rate": [v["bad_rate"] * 100 for v in splits.values()],
        "AUC": [v["auc"] for v in splits.values()],
        "Gini": [v["gini"] for v in splits.values()],
        "KS": [v["ks"] for v in splits.values()]})
    st.dataframe(table, hide_index=True, column_config={
        "Loans": st.column_config.NumberColumn(format="%d"),
        "Bad rate": st.column_config.NumberColumn(format="%.1f%%"),
        "AUC": st.column_config.NumberColumn(format="%.3f"),
        "Gini": st.column_config.NumberColumn(format="%.3f"),
        "KS": st.column_config.NumberColumn(format="%.3f")})
    train, test, oot = splits["train"]["auc"], splits["test"]["auc"], splits["oot"]["auc"]
    gap = train - oot
    verdict = ("holds up well on later loans" if gap < 0.02 else
               "loses some power on later loans" if gap < 0.05 else
               "loses clear power on later loans")
    c.takeaway(f"AUC is {train:.3f} on training loans, {test:.3f} on held-out loans from the same "
               f"period and {oot:.3f} on the later, unseen out-of-time loans ({gap:+.3f} vs "
               f"training): the model {verdict}.")
    with st.expander("How to read this"):
        st.markdown(
            "- **AUC**: pick one loan that went bad and one that was repaid at random. AUC is "
            "how often the bad one has the lower score. 0.5 is guessing, 1.0 is perfect.\n"
            "- **Gini** = 2 x AUC - 1. The same information rescaled to 0-1.\n"
            "- **KS**: the widest gap between the cumulative share of bad loans and of good "
            "loans as you walk up the score scale. Bigger = better separation.\n"
            "- **Train / Test / Out-of-time**: train is what the model learned from; test is "
            "held-out loans from the same years; out-of-time is a later year, the closest "
            "thing to real deployment. Similar numbers across the three = no overfitting.")


def _decile_chart() -> None:
    split = st.radio("Sample", list(SPLIT_NAMES), index=2, horizontal=True,
                     format_func=SPLIT_NAMES.get, key="decile_split")
    deciles = c.load(f"decile_table_{split}.csv")
    if deciles is None:
        c.need(f"decile_table_{split}.csv")
        return
    labels = [f"{int(d)}" for d in deciles["decile"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=deciles["bad_rate"], name="Observed bad rate",
                marker_color=c.BLUE,
                customdata=deciles[["min_score", "max_score", "n"]],
                hovertemplate="Decile %{x}<br>Scores %{customdata[0]:.0f}-%{customdata[1]:.0f}"
                              "<br>%{customdata[2]:,} loans<br>Bad rate %{y:.1%}<extra></extra>")
    fig.add_scatter(x=labels, y=deciles["mean_pd"], name="Model's predicted PD", mode="lines+markers",
                    line={"color": c.ORANGE, "width": 3}, marker={"size": 8})
    fig.update_layout(xaxis_title="Score decile (1 = lowest scores, riskiest)",
                      yaxis_title="Bad rate", yaxis_tickformat=".0%", hovermode="x")
    c.show(c.style(fig, 340))
    worst, best = deciles["bad_rate"].iloc[0], deciles["bad_rate"].iloc[-1]
    k = min(3, len(deciles))
    capture = deciles["cum_bad_capture"].iloc[k - 1]
    share = k / len(deciles)
    ratio = f"{worst / best:.0f}x" if best > 0 else "many times"
    c.takeaway(f"The riskiest tenth of loans went bad {c.pct(worst)} of the time versus "
               f"{c.pct(best)} for the safest tenth ({ratio} higher), and the lowest-scoring "
               f"{c.pct(share, 0)} contain {c.pct(capture, 0)} of all bad loans.")


def _calibration_chart() -> None:
    calib = c.load("calibration_oot.csv")
    if calib is None:
        c.need("calibration_oot.csv")
        return
    top = max(calib["mean_pd"].max(), calib["observed_bad_rate"].max()) * 1.1
    fig = go.Figure()
    fig.add_scatter(x=[0, top], y=[0, top], name="Perfect calibration", mode="lines",
                    line={"color": c.GREY, "dash": "dash"})
    fig.add_scatter(x=calib["mean_pd"], y=calib["observed_bad_rate"], name="Risk decile (OOT)",
                    mode="lines+markers", line={"color": c.BLUE, "width": 3},
                    marker={"size": 9}, text=calib["decile"],
                    hovertemplate="Decile %{text}<br>Predicted %{x:.1%}<br>Observed %{y:.1%}"
                                  "<extra></extra>")
    fig.update_layout(xaxis_title="Predicted probability of default",
                      yaxis_title="Observed bad rate", xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", hovermode="closest")
    c.show(c.style(fig, 340))
    weights = calib["n"] / calib["n"].sum()
    predicted, observed = (calib["mean_pd"] * weights).sum(), (calib["observed_bad_rate"]
                                                               * weights).sum()
    direction = "over-predicts" if predicted > observed else "under-predicts"
    c.takeaway(f"Across all out-of-time loans the model predicts a {c.pct(predicted)} default "
               f"rate against {c.pct(observed)} observed, so it {direction} overall by "
               f"{abs(predicted - observed) * 100:.1f} percentage points. Points on the dashed "
               "line mean predicted and actual agree.")


def _psi_block(metrics: dict) -> None:
    psi = metrics["psi_score_train_to_oot"]
    c.metric_row([("Score PSI, train to out-of-time", f"{psi:.3f}", c.psi_reading(psi),
                   "Population Stability Index: how much the score distribution moved.")])
    c.takeaway(f"PSI of {psi:.3f} is '{c.psi_reading(psi).lower()}': below {config.PSI_STABLE} "
               f"is stable, {config.PSI_STABLE}-{config.PSI_SHIFT} is a moderate shift worth "
               f"watching, above {config.PSI_SHIFT} means the applicant mix has changed enough "
               "to consider rebuilding the model.")


def _benchmark_chart(metrics: dict) -> None:
    bench, ours = metrics["benchmark_sub_grade_oot"], metrics["splits"]["oot"]
    names = ["AUC", "Gini", "KS"]
    keys = ["auc", "gini", "ks"]
    fig = go.Figure()
    fig.add_bar(x=names, y=[ours[k] for k in keys], name="This scorecard", marker_color=c.BLUE,
                text=[f"{ours[k]:.3f}" for k in keys], textposition="outside")
    fig.add_bar(x=names, y=[bench[k] for k in keys], name="Lending Club sub-grade",
                marker_color=c.ORANGE, text=[f"{bench[k]:.3f}" for k in keys],
                textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="Out-of-time", hovermode="x")
    c.show(c.style(fig, 340))
    diff = ours["auc"] - bench["auc"]
    verdict = ("edges out" if diff > 0.005 else "trails" if diff < -0.005 else "roughly matches")
    c.takeaway(f"On out-of-time loans this scorecard (AUC {ours['auc']:.3f}) {verdict} Lending "
               f"Club's own sub-grade (AUC {bench['auc']:.3f}, {diff:+.3f}).")
    st.caption("Sub-grade is LC's own risk rating and was deliberately kept out of the model, so "
               "this is a fair, independent comparison. LC's rating may use inputs that are not in "
               "this public dataset.")


def _iv_block(metrics: dict) -> None:
    iv = c.load("iv_table.csv")
    if iv is None:
        c.need("iv_table.csv")
        return
    top = iv.head(20).iloc[::-1]
    final = set(metrics["features"])
    fig = go.Figure(go.Bar(
        x=top["iv"], y=[feature_label(f) for f in top["feature"]], orientation="h",
        marker_color=[c.BLUE if f in final else "#B6BDC8" for f in top["feature"]],
        customdata=[c.iv_strength(v) for v in top["iv"]],
        hovertemplate="%{y}<br>IV %{x:.3f} (%{customdata})<extra></extra>"))
    fig.update_layout(xaxis_title="Information value (blue = used in the model)",
                      hovermode="closest")
    c.show(c.style(fig, max(300, 26 * len(top) + 80), legend=False))
    best = iv.iloc[0]
    c.takeaway(f"{len(final)} of {len(iv)} candidate features made it into the model. The "
               f"strongest single predictor is {feature_label(best['feature'])} "
               f"(IV {best['iv']:.2f}, {c.iv_strength(best['iv']).lower()}).")
    with st.expander("What are WOE and IV?"):
        st.markdown(
            "Each feature is cut into a few **bins** (e.g. FICO 660-690, 690-720, ...). For "
            "every bin, **Weight of Evidence (WOE)** says how much better or worse than average "
            "its borrowers behave: WOE = ln(share of all good loans in the bin / share of all "
            "bad loans in the bin). Positive WOE = safer than average, negative = riskier.\n\n"
            "**Information Value (IV)** adds the bins up into one number per feature: how much "
            "the feature helps separate good from bad. Rules of thumb: under 0.02 not useful, "
            "0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, above 0.5 suspiciously strong "
            "(often a sign of data leakage, so it gets reviewed).\n\n"
            "The model then weighs each feature's WOE; the **points table** below is the same "
            "model rewritten as points per bin so anyone can add them up by hand.")


def _points_block(metrics: dict) -> None:
    points = c.load("scorecard_points.csv")
    if points is None:
        c.need("scorecard_points.csv")
        return
    features = [f for f in metrics["features"] if f in set(points["feature"])]
    feature = st.selectbox("Feature", features, format_func=feature_label, key="points_feature")
    p = points[points["feature"] == feature]
    fig = go.Figure(go.Bar(
        x=p["bin"], y=p["points"], marker_color=c.BLUE, text=[f"{v:.0f}" for v in p["points"]],
        textposition="outside", customdata=p["bad_rate"],
        hovertemplate="%{x}<br>%{y:.0f} points<br>Bad rate %{customdata:.1%}<extra></extra>"))
    fig.update_layout(xaxis_title=feature_label(feature), yaxis_title="Scorecard points",
                      hovermode="closest")
    c.show(c.style(fig, 320, legend=False))
    hi, lo = p.loc[p["points"].idxmax()], p.loc[p["points"].idxmin()]
    c.takeaway(f"For {feature_label(feature)}, '{hi['bin']}' earns {hi['points']:.0f} points "
               f"(historical bad rate {c.pct(hi['bad_rate'])}) while '{lo['bin']}' earns "
               f"{lo['points']:.0f} ({c.pct(lo['bad_rate'])}): a {hi['points'] - lo['points']:.0f}"
               "-point swing from this one feature.")
    with st.expander("Full scorecard points table"):
        shown = points.assign(feature=points["feature"].map(feature_label),
                              bad_rate=points["bad_rate"] * 100)
        st.dataframe(shown, hide_index=True, column_config={
            "n": st.column_config.NumberColumn("Loans", format="%d"),
            "bad_rate": st.column_config.NumberColumn("Bad rate", format="%.1f%%"),
            "woe": st.column_config.NumberColumn("WOE", format="%.3f"),
            "coefficient": st.column_config.NumberColumn("Coefficient", format="%.3f"),
            "points": st.column_config.NumberColumn("Points", format="%.0f")})


def render() -> None:
    c.header("Model & Validation",
             "Does the scorecard rank risk correctly, are its probabilities realistic, and is "
             "it stable over time?")
    if not c.need("metrics.json"):
        return
    metrics = c.load("metrics.json")
    perf, calib, bench, feats = st.tabs(
        ["Performance", "Calibration & stability", "Benchmark", "Features & points"])
    with perf:
        st.subheader("Discrimination: can it tell good from bad?")
        _metrics_table(metrics)
        st.subheader("Bad rate by score decile")
        _decile_chart()
    with calib:
        st.subheader("Calibration: are the probabilities realistic?")
        _calibration_chart()
        st.subheader("Stability: has the applicant mix drifted?")
        _psi_block(metrics)
    with bench:
        st.subheader("Versus Lending Club's own rating")
        _benchmark_chart(metrics)
    with feats:
        st.subheader("Which features matter? (Information Value)")
        _iv_block(metrics)
        st.subheader("Scorecard points")
        _points_block(metrics)
