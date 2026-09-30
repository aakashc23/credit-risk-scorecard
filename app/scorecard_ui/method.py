"""Page 5: data funnel, target definition, leakage rules and the SQL profiling results."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scorecard import config
from scorecard_ui import common as c


def _funnel(summary: dict, metrics: dict | None) -> None:
    counts = summary["row_counts"]
    steps = [("Raw rows read", counts["rows_read"]),
             ("36-month loans in study window", counts["rows_36m_in_windows"]),
             ("Outcome known (good or bad)", counts["rows_with_known_outcome"])]
    fig = go.Figure(go.Bar(
        x=[v for _, v in steps][::-1], y=[k for k, _ in steps][::-1], orientation="h",
        marker_color=[c.BLUE, c.SKY, c.GREY], text=[c.num(v) for _, v in steps][::-1],
        textposition="outside", cliponaxis=False))
    fig.update_layout(xaxis_title="Rows", hovermode="closest")
    c.show(c.style(fig, 240, legend=False))
    read, window, known = (v for _, v in steps)
    c.takeaway(f"Of {c.num(read)} rows read, {c.pct(window / read if read else float('nan'))} are "
               f"{config.TERM_MONTHS}-month loans issued in the study window, and "
               f"{c.pct(known / window if window else float('nan'))} of those have a known "
               "good/bad outcome.")
    per_split = counts["rows_per_split"]
    rows = []
    for name, n in per_split.items():
        bad = metrics["splits"][name]["bad_rate"] if metrics else float("nan")
        rows.append({"Sample": name, "Loans": n, "Bad rate": bad * 100})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        "Loans": st.column_config.NumberColumn(format="%d"),
        "Bad rate": st.column_config.NumberColumn(format="%.1f%%")})


def _sample_and_target(summary: dict) -> None:
    st.markdown(
        f"- **Loans:** {config.TERM_MONTHS}-month terms only, so every loan has reached its "
        "contractual end and its outcome is known.\n"
        f"- **Development window:** loans issued {config.DEV_START} to {config.DEV_END}, split "
        f"{round((1 - config.TEST_SIZE) * 100)}/{round(config.TEST_SIZE * 100)} into train and "
        "in-time test.\n"
        f"- **Out-of-time window:** loans issued {config.OOT_START} to {config.OOT_END}, never "
        "used for fitting.")
    target = summary["target"]
    st.markdown(
        "**Target.** `bad = 1` when the final status is "
        + ", ".join(f"*{s}*" for s in sorted(config.BAD_STATUSES))
        + "; `bad = 0` when it is "
        + ", ".join(f"*{s}*" for s in sorted(config.GOOD_STATUSES))
        + f". Loans with any other status (current, late, in grace period) are excluded: "
        f"{c.num(target['n_indeterminate'])} here.")
    with st.expander("Loan status counts before exclusions"):
        counts = pd.DataFrame(sorted(target["status_counts"].items(), key=lambda kv: -kv[1]),
                              columns=["Loan status", "Loans"])
        st.dataframe(counts, hide_index=True)


def _leakage(summary: dict) -> None:
    st.markdown(
        "A model must only use what was known **when the application was made**. Anything that "
        "happens after the loan is issued (payments received, recoveries, hardship plans, "
        "settlements) would leak the answer, so features come from an explicit whitelist and a "
        "unit test fails if a forbidden field sneaks in.")
    cols = st.columns(3)
    cols[0].metric("Whitelisted raw fields", len(config.CANDIDATE_FEATURES), border=True)
    cols[1].metric("Blocked post-origination fields", len(config.LEAKAGE_COLUMNS), border=True)
    cols[2].metric("Features in final model", len(summary.get("final_features", [])),
                   border=True)
    with st.expander("Blocked post-origination (leakage) fields"):
        st.code(", ".join(sorted(config.LEAKAGE_COLUMNS)), language=None)
    with st.expander("Excluded on purpose: LC's own pricing outputs and geography"):
        st.markdown(
            f"- **Benchmark only, not features:** {', '.join(f'`{x}`' for x in config.BENCHMARK_COLS)}"
            " are Lending Club's own risk model output; using them would make this score a copy "
            "of theirs.\n"
            f"- **Fair-lending hygiene:** {', '.join(f'`{x}`' for x in config.EXCLUDED_FAIR_LENDING)}"
            " can act as proxies for protected classes.\n"
            f"- **Loss calibration only:** {', '.join(f'`{x}`' for x in config.LOSS_CALIBRATION_COLS)}"
            " are post-origination; they estimate portfolio LGD/EAD on training-period defaults "
            "and measure realised loss for back-testing. They never enter the model.")
    with st.expander("Feature selection log"):
        selection = summary.get("selection", {})
        st.markdown(
            f"- Candidates considered: {len(summary.get('candidate_features', []))}\n"
            f"- Dropped for >{config.MAX_MISSING_SHARE:.0%} missing: "
            f"{', '.join(summary.get('dropped_high_missing', [])) or 'none'}\n"
            f"- Dropped for IV below {config.MIN_IV}: "
            f"{', '.join(selection.get('dropped_low_iv', [])) or 'none'}\n"
            f"- Dropped for correlation with a stronger feature: "
            f"{', '.join(selection.get('dropped_correlated', {})) or 'none'}\n"
            f"- Dropped for a wrong-sign coefficient: "
            f"{', '.join(summary.get('dropped_wrong_sign', [])) or 'none'}\n"
            f"- Flagged for IV above {config.SUSPICIOUS_IV} (leakage review): "
            f"{', '.join(summary.get('suspicious_iv', [])) or 'none'}")


def _sql_results() -> None:
    st.markdown("Portfolio profiling runs as plain SQL (DuckDB) over the pipeline's parquet "
                "outputs. Each result below is the saved output of the query beneath it.")
    files = sorted(config.SQL_DIR.glob("*.sql"))
    if not files:
        st.caption("No SQL files found in the repository's `sql/` folder.")
    for sql_file in files:
        name = sql_file.stem.split("_", 1)[1]
        text = sql_file.read_text(encoding="utf-8")
        description = text.splitlines()[0].lstrip("- ").strip() if text.strip() else ""
        st.markdown(f"**{name.replace('_', ' ').capitalize()}**")
        if description:
            st.caption(description)
        result = c.load(f"sql_{name}.csv")
        if result is None:
            c.need(f"sql_{name}.csv")
        else:
            st.dataframe(result, hide_index=True)
        with st.expander(f"SQL: {sql_file.name}"):
            st.code(text, language="sql")


def render() -> None:
    c.header("Data & Method",
             "Where the numbers come from, how good/bad is defined and how leakage is kept out.")
    if not c.need("data_summary.json"):
        return
    summary = c.load("data_summary.json")
    metrics = c.load("metrics.json")
    st.subheader("From raw rows to a modelling sample")
    _funnel(summary, metrics)
    st.subheader("Sample and target definition")
    _sample_and_target(summary)
    st.subheader("Leakage control")
    _leakage(summary)
    st.subheader("SQL profiling")
    _sql_results()
