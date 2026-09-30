"""Page 2: the approval cut-off simulator (the core business page)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scorecard import config
from scorecard.cutoff import cutoff_summary
from scorecard_ui import common as c

WIDGET_KEY = "_cutoff_widget"   # widget state is dropped when the page is left ...
SHARED_KEY = "cutoff"           # ... so the chosen value is mirrored here for other pages
HIST_STEP = 10


@st.cache_data(show_spinner=False)
def _summary(path: str, mtime: float, cutoff: float) -> dict:
    return cutoff_summary(c._read(path, mtime), cutoff)


@st.cache_data(show_spinner=False)
def _histograms(path: str, mtime: float) -> dict:
    """Score histograms (share of each group per bin) and group medians, computed once."""
    scored = c._read(path, mtime)
    edges = np.arange(config.SCORE_MIN, config.SCORE_MAX + HIST_STEP + 1, HIST_STEP)
    out = {"centres": (edges[:-1] + HIST_STEP / 2).tolist()}
    for name, flag in (("good", 0), ("bad", 1)):
        scores = scored.loc[scored["bad"] == flag, "score"].to_numpy(float)
        counts, _ = np.histogram(scores, edges)
        out[name] = (counts / max(len(scores), 1)).tolist()
        out[f"median_{name}"] = float(np.median(scores)) if len(scores) else float("nan")
    return out


def _scale_el(d: dict, factor: float) -> dict:
    """Expected loss is linear in LGD, so a what-if LGD just rescales the EL fields."""
    return {**d, "expected_loss": d["expected_loss"] * factor,
            "expected_loss_rate": d["expected_loss_rate"] * factor}


def _slider(default: int) -> int:
    if WIDGET_KEY not in st.session_state:
        st.session_state[WIDGET_KEY] = int(st.session_state.get(SHARED_KEY, default))
    cutoff = st.slider(
        "Approval cut-off: approve applicants scoring at or above", config.SCORE_MIN,
        config.SCORE_MAX, step=config.CUTOFF_STEP, key=WIDGET_KEY,
        help="Higher cut-off = stricter: fewer approvals, lower expected bad rate.")
    st.session_state[SHARED_KEY] = int(cutoff)
    return int(cutoff)


def _distribution_chart(cutoff: int, hist: dict, s: dict, total: dict) -> None:
    fig = go.Figure()
    fig.add_vrect(x0=config.SCORE_MIN, x1=cutoff, fillcolor=c.GREY, opacity=0.10, line_width=0,
                  annotation_text="rejected", annotation_position="top left")
    for name, label, colour in (("good", "Repaid (good)", c.BLUE),
                                ("bad", "Defaulted (bad)", c.VERMILLION)):
        fig.add_bar(x=hist["centres"], y=hist[name], name=label, marker_color=colour,
                    opacity=0.65, width=HIST_STEP * 0.95)
    fig.add_vline(x=cutoff, line_dash="dash", line_color="#1F2937")
    fig.update_layout(barmode="overlay", xaxis_title="Score", yaxis_title="Share of group",
                      yaxis_tickformat=".0%", xaxis_range=[config.SCORE_MIN, config.SCORE_MAX])
    c.show(c.style(fig))
    total_bad = total["observed_bad_rate"] * total["n_total"]
    bad_kept = s["observed_bad_rate"] * s["n_approved"] if s["n_approved"] else 0.0
    total_good = total["n_total"] - total_bad
    good_kept = s["n_approved"] - bad_kept
    bad_rej = 1 - bad_kept / total_bad if total_bad else float("nan")
    good_rej = 1 - good_kept / total_good if total_good else float("nan")
    c.takeaway(
        f"Loans that went bad scored lower (median {hist['median_bad']:.0f}) than loans that "
        f"were repaid (median {hist['median_good']:.0f}). This cut-off turns away "
        f"{c.pct(bad_rej, 0)} of the loans that later went bad, at the price of also "
        f"rejecting {c.pct(good_rej, 0)} of the loans that were repaid.")


def _tradeoff_charts(table, s: dict, total: dict, scale: float) -> None:
    t = table[table["approval_rate"] >= 0.02].copy()
    t["el"] = t["expected_loss"] * scale
    x = t["approval_rate"]
    left, right = st.columns(2)

    with left:
        fig = go.Figure()
        fig.add_scatter(x=x, y=t["expected_bad_rate"], name="Expected bad rate (model)",
                        line={"color": c.BLUE, "width": 3})
        fig.add_scatter(x=x, y=t["observed_bad_rate"], name=f"Observed bad rate ({c.OOT_YEAR})",
                        line={"color": c.VERMILLION, "width": 3, "dash": "dot"})
        fig.add_scatter(x=[s["approval_rate"]], y=[s["expected_bad_rate"]], name="Current cut-off",
                        mode="markers", marker={"size": 13, "color": "#1F2937", "symbol": "diamond"})
        fig.update_layout(xaxis_title="Approval rate", yaxis_title="Bad rate among approved",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%", hovermode="x")
        c.show(c.style(fig, 340))
        if s["rejection_rate"] == 0:
            c.takeaway("Everyone is approved at this cut-off; raise it to see the trade-off.")
        else:
            drop = total["expected_bad_rate"] - s["expected_bad_rate"]
            c.takeaway(f"Turning away {c.pct(s['rejection_rate'], 0)} of applicants lowers the "
                       f"expected bad rate by {drop * 100:.1f} percentage points, from "
                       f"{c.pct(total['expected_bad_rate'])} to {c.pct(s['expected_bad_rate'])}.")

    with right:
        fig = go.Figure()
        fig.add_scatter(x=x, y=t["el"], name="Expected loss (model)",
                        line={"color": c.BLUE, "width": 3})
        fig.add_scatter(x=x, y=t["realised_loss"], name=f"Realised loss ({c.OOT_YEAR})",
                        line={"color": c.VERMILLION, "width": 3, "dash": "dot"})
        fig.add_scatter(x=[s["approval_rate"]], y=[s["expected_loss"]], name="Current cut-off",
                        mode="markers", marker={"size": 13, "color": "#1F2937", "symbol": "diamond"})
        fig.update_layout(xaxis_title="Approval rate", yaxis_title="Loss ($)",
                          xaxis_tickformat=".0%", yaxis_tickformat="$~s", hovermode="x")
        c.show(c.style(fig, 340))
        saved = total["expected_loss"] - s["expected_loss"]
        if s["rejection_rate"] == 0:
            c.takeaway("Nothing is rejected at this cut-off, so there is no loss avoided yet.")
        else:
            c.takeaway(f"Compared with approving everyone ({c.usd(total['expected_loss'])}), "
                       f"this cut-off avoids {c.usd(saved)} of expected loss "
                       f"({c.pct(saved / total['expected_loss'], 0)}) while approving "
                       f"{c.pct(s['approval_rate'], 0)} of applicants.")


def render() -> None:
    c.header("Approval Cut-off Simulator",
             "Choose the lowest score you would approve and see the business consequences on "
             f"out-of-time loans (issued {c.OOT_YEAR}).")
    if not c.need("scored_oot.parquet", "cutoff_table.csv", "metrics.json"):
        return
    root = c.artifacts_dir() / "scored_oot.parquet"
    path, mtime = str(root), c._mtime(root)
    table = c.load("cutoff_table.csv")
    lgd_cal = c.load("metrics.json")["loss_params"]["lgd"]
    default = c.default_cutoff()

    cutoff = _slider(default)
    st.caption(f"Starting point: a cut-off of {default} approves about "
               f"{c.pct(config.DEFAULT_APPROVAL_TARGET, 0)} of applicants (picked from the data, "
               f"not hard-coded).")

    with st.expander("What-if: change the loss-given-default (LGD) assumption"):
        lgd = st.number_input("LGD (share of a defaulted balance that is lost)", 0.0, 1.0,
                              value=float(round(lgd_cal, 4)), step=0.01, format="%.4f",
                              key="_lgd_widget",
                              help="Default is the LGD calibrated on training-period defaults.")
        st.caption("Expected loss scales directly with LGD; realised loss is an actual "
                   f"{c.OOT_YEAR} figure and does not change.")
    scale = lgd / lgd_cal if lgd_cal else 1.0

    s = _scale_el(_summary(path, mtime, cutoff), scale)
    total = _scale_el(_summary(path, mtime, config.SCORE_MIN), scale)

    c.metric_row([
        ("Approval rate", c.pct(s["approval_rate"]), "share of applicants approved"),
        ("Rejection rate", c.pct(s["rejection_rate"]), "share turned away"),
        ("Approved loans", c.num(s["n_approved"]), f"of {c.num(s['n_total'])} applicants"),
        ("Rejected loans", c.num(s["n_rejected"])),
    ])
    c.metric_row([
        ("Expected bad rate", c.pct(s["expected_bad_rate"]), "model's average PD of approved"),
        ("Observed bad rate", c.pct(s["observed_bad_rate"]),
         f"{c.OOT_YEAR} actuals on approved loans"),
        ("Expected loss", c.usd(s["expected_loss"]),
         f"{c.pct(s['expected_loss_rate'])} of approved exposure",
         "Expected loss = PD x LGD x EAD, summed over approved loans."),
        ("Realised loss", c.usd(s["realised_loss"]),
         f"actual {c.OOT_YEAR} loss on approved loans"),
    ])

    if s["n_approved"] == 0:
        st.info("No applicants are approved at this cut-off. Lower it to see results.")
    else:
        st.info(
            f"At a cut-off of **{cutoff}** we approve **{c.num(s['n_approved'])} of "
            f"{c.num(s['n_total'])}** applicants (**{c.pct(s['approval_rate'])}**). The expected "
            f"bad rate is **{c.pct(s['expected_bad_rate'])}**, versus "
            f"**{c.pct(total['expected_bad_rate'])}** if we approved everyone. Expected loss is "
            f"**{c.usd(s['expected_loss'])}**, which is **{c.pct(s['expected_loss_rate'])}** of "
            f"approved exposure.", icon=":material/lightbulb:")

    st.subheader("Who gets approved?")
    _distribution_chart(cutoff, _histograms(path, mtime), s, total)
    st.subheader("The trade-off")
    _tradeoff_charts(table, s, total, scale)
    st.caption("Expected figures come from the model; observed and realised figures are what "
               f"actually happened to {c.OOT_YEAR} loans. Every loan here was really approved at "
               "the time, which is why actual outcomes exist even for those this cut-off rejects.")
