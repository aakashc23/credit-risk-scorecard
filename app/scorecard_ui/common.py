"""Shared helpers for the dashboard: artifact loading, formatting, chart styling.

The dashboard reads only the small files in ``artifacts/`` (or ``SCORECARD_ARTIFACTS_DIR``).
It never touches raw data and needs no scikit-learn.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scorecard import config
from scorecard.predict import ScoringModel, load_model

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, VERMILLION = "#0072B2", "#E69F00", "#D55E00"
GREEN, SKY, GREY = "#009E73", "#56B4E9", "#6B7280"
GRID = "#E5E7EB"

OOT_YEAR = config.OOT_START[:4]  # the out-of-time vintage, e.g. "2015"

RUN_HINT = "Run `python scripts/run_pipeline.py` to (re)build the artifacts, then reload this page."


# --------------------------------------------------------------------------- artifacts
def artifacts_dir() -> Path:
    """``$SCORECARD_ARTIFACTS_DIR`` if set (relative paths: from the cwd, then the repo root)."""
    override = os.environ.get("SCORECARD_ARTIFACTS_DIR", "").strip()
    if not override:
        return config.ARTIFACTS
    path = Path(override).expanduser()
    if not path.is_absolute() and not path.exists() and (config.ROOT / path).exists():
        path = config.ROOT / path
    return path


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float):
    """Read an artifact by extension. ``mtime`` only keys the cache so edits are picked up."""
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def load(name: str):
    """Artifact ``name`` (json -> dict, csv/parquet -> DataFrame), or None if it is missing."""
    path = artifacts_dir() / name
    return _read(str(path), _mtime(path)) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _model(path: str, mtime: float) -> ScoringModel:
    return load_model(path)


def get_model() -> ScoringModel | None:
    root = artifacts_dir()
    if not (root / config.MODEL_FILENAME).exists() or not (root / "woe_bins.json").exists():
        return None
    return _model(str(root), _mtime(root / config.MODEL_FILENAME))


def need(*names: str) -> bool:
    """True if every artifact exists; otherwise show a clear warning and return False."""
    missing = [n for n in names if not (artifacts_dir() / n).exists()]
    if missing:
        st.warning(f"Missing artifact file(s): {', '.join(f'`{m}`' for m in missing)} in "
                   f"`{artifacts_dir().name}/`. {RUN_HINT}", icon=":material/folder_off:")
    return not missing


def default_cutoff() -> int:
    """Data-driven starting cut-off: the highest one keeping DEFAULT_APPROVAL_TARGET approved."""
    from scorecard.cutoff import cutoff_for_approval_rate

    table = load("cutoff_table.csv")
    return int(cutoff_for_approval_rate(table, config.DEFAULT_APPROVAL_TARGET))


def current_cutoff() -> int:
    """The simulator's cut-off (shared through session state), else the default."""
    return int(st.session_state.get("cutoff", default_cutoff()))


# --------------------------------------------------------------------------- formatting
def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def usd(x: float) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e9:
        return f"{sign}${x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{sign}${x / 1e6:.1f}M"
    if x >= 1e4:
        return f"{sign}${x / 1e3:.0f}K"
    return f"{sign}${x:,.0f}"


def num(x: float, digits: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


# --------------------------------------------------------------------------- page furniture
def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def md(text: str) -> str:
    """Escape ``$`` so dollar amounts are not parsed as LaTeX by Streamlit's markdown."""
    return text.replace("$", r"\$")


def takeaway(text: str) -> None:
    """One plain-English line under a chart. Callers build ``text`` from the data."""
    st.markdown(f":blue[**Takeaway.**] {md(text)}")


def metric_row(tiles: list[tuple], columns: int | None = None) -> None:
    """Row of bordered KPI tiles: (label, value[, sub-text[, help]])."""
    cols = st.columns(columns or len(tiles))
    for col, tile in zip(cols, tiles, strict=False):
        label, value, *rest = tile
        sub = rest[0] if rest else None
        help_ = rest[1] if len(rest) > 1 else None
        with col.container(border=True):
            st.metric(label, value, help=help_)
            st.caption(sub or " ")


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    """Consistent, light chart styling."""
    fig.update_layout(
        template="plotly_white", height=height, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"size": 13, "color": "#1F2937"}, showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0, "title": None},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={"displayModeBar": False})


def iv_strength(iv: float) -> str:
    return next(label for bound, label in config.IV_BANDS if iv < bound)


def psi_reading(value: float) -> str:
    if value < config.PSI_STABLE:
        return "Stable"
    return "Moderate shift" if value < config.PSI_SHIFT else "Significant shift"
