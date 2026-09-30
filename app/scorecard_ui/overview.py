"""Page 1: what the project does, headline results, pipeline and honest limitations."""
from __future__ import annotations

import streamlit as st

from scorecard import config
from scorecard_ui import common as c

PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        color="#9CA3AF", fillcolor="#F3F4F6", fontcolor="#1F2937", margin="0.12,0.06"];
  edge [color="#6B7280", arrowsize=0.7];
  raw [label="Lending Club\naccepted loans"];
  clean [label="Clean &\nmature 36m loans"];
  target [label="Target:\ndefault = 1"];
  leak [label="Drop post-origination\n(leakage) fields", fillcolor="#FDE7D6"];
  woe [label="Bin + WOE / IV\nfeature selection"];
  lr [label="Logistic\nregression", fillcolor="#D6E9F5"];
  score [label="PD -> 300-850\nscore", fillcolor="#D6E9F5"];
  val [label="Validate\nAUC KS PSI"];
  el [label="Expected loss\nPD x LGD x EAD"];
  cut [label="Cut-off\nsimulator", fillcolor="#D6E9F5"];
  raw -> clean -> target -> leak -> woe -> lr -> score -> val -> el -> cut;
}
"""


def render() -> None:
    c.header("Credit Risk Scorecard & Approval Cut-off Simulator",
             "From a loan application to a risk score, an expected loss and an approve/decline "
             "decision.")
    if not c.need("metrics.json"):
        return
    metrics = c.load("metrics.json")
    oot = metrics["splits"]["oot"]
    loss = metrics["loss_params"]

    st.markdown(
        f"- **Predicts default risk.** A transparent logistic-regression scorecard turns an "
        f"applicant's application-time facts into a probability of default (PD) and a "
        f"{config.SCORE_MIN}-{config.SCORE_MAX} score, like a credit score.\n"
        "- **Puts a price on it.** Expected loss = PD x LGD x EAD: how likely a loan is to go "
        "bad, how much of it is lost when it does, and how much is owed at that point.\n"
        "- **Lets you choose the trade-off.** Move the approval cut-off and see how many "
        "loans are approved, how many go bad and how many dollars are at risk.")

    st.subheader("Headline results")
    st.caption("Measured on the out-of-time (OOT) set: loans issued after the model's training "
               "period, which it has never seen.")
    c.metric_row([
        ("AUC", f"{oot['auc']:.3f}", "0.5 = coin flip, 1.0 = perfect",
         ("Chance that a randomly chosen bad loan gets a lower score than a randomly chosen "
          "good one.")),
        ("Gini", f"{oot['gini']:.3f}", "2 x AUC - 1", "Same information as AUC, rescaled to 0-1."),
        ("KS", f"{oot['ks']:.3f}", "max good/bad separation",
         "Largest gap between the cumulative share of bad and of good loans as the score rises."),
    ])
    c.metric_row([
        ("OOT loans", c.num(oot["n"]), "36-month loans, known outcome"),
        ("Observed bad rate", c.pct(oot["bad_rate"]), "share that charged off / defaulted"),
        ("LGD", c.pct(loss["lgd"]), "share of a defaulted balance lost",
         "Loss given default, calibrated on training-period defaults."),
        ("EAD ratio", c.pct(loss["ead_ratio"]), "owed at default / loan amount",
         "Exposure at default as a share of the loan amount, from training-period defaults."),
    ])
    if metrics.get("sample"):
        st.info(f"These artifacts come from a development sample (first {metrics['sample']:,} "
                "raw rows), not the full dataset.", icon=":material/science:")

    st.subheader("How it works")
    st.graphviz_chart(PIPELINE_DOT)
    st.caption("Blue = the model and its outputs; orange = the leakage guard that keeps "
               "information from after the loan was made out of the model.")

    st.warning(
        "**Read this first.** The data holds *accepted* loans only, so there is no information "
        "on applicants who were rejected; the simulator re-scores the approved book, it does "
        "not show what would have happened to declined applicants. This is a portfolio "
        "project on historical public data: it is not a lending system and must not be used "
        "to make credit decisions.", icon=":material/info:")
    st.caption(f"Artifacts generated {metrics.get('generated_at', 'n/a')} from "
               f"`{metrics.get('data_file', 'n/a')}`.")
