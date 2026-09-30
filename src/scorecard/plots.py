"""Matplotlib figures (PNG) for the README and docs. Colour-blind-safe Okabe-Ito palette."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from scorecard.config import MIN_IV, SUSPICIOUS_IV
from scorecard.validation import auc, grade_to_ordinal

logger = logging.getLogger(__name__)

BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#6B6B6B")
SPLIT_COLORS = {"train": BLUE, "test": ORANGE, "oot": GREEN}
DPI = 140

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct_axis(ax: plt.Axes, axis: str = "y") -> None:
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def iv_bar(iv: pd.DataFrame, path: Path, top: int = 20) -> Path:
    """Information value per feature with the selection / leakage-review thresholds."""
    data = iv.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(data) + 1.6))
    ax.barh(data["feature"], data["iv"], color=BLUE)
    ax.axvline(MIN_IV, color=ORANGE, ls="--", label=f"Min IV ({MIN_IV})")
    ax.axvline(SUSPICIOUS_IV, color=RED, ls="--", label=f"Leakage review ({SUSPICIOUS_IV})")
    ax.set_xlabel("Information value (IV)")
    ax.set_title("Information value by feature (train)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def score_distribution(df: pd.DataFrame, path: Path, title: str = "Score distribution (OOT)") -> Path:
    """Histogram of scores for good vs bad loans."""
    bins = np.linspace(df["score"].min(), df["score"].max() + 1e-9, 41)
    fig, ax = plt.subplots(figsize=(7, 4))
    for value, label, color in [(0, "Good (fully paid)", BLUE), (1, "Bad (default)", RED)]:
        ax.hist(df.loc[df["bad"] == value, "score"], bins=bins, density=True, alpha=0.6,
                color=color, label=label)
    ax.set_xlabel("Score (points)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def roc(splits: dict[str, pd.DataFrame], path: Path) -> Path:
    """ROC curves for each split."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, df in splits.items():
        fpr, tpr, _ = roc_curve(df["bad"], df["pd"])
        ax.plot(fpr, tpr, color=SPLIT_COLORS.get(name, GREY), lw=2,
                label=f"{name} (AUC {auc(df['bad'], df['pd']):.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", label="Random")
    ax.set_xlabel("False positive rate (good loans rejected)")
    ax.set_ylabel("True positive rate (bad loans rejected)")
    ax.set_title("ROC curve by split")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return _save(fig, path)


def ks_curve(df: pd.DataFrame, path: Path) -> Path:
    """Cumulative good/bad distributions over score with the KS gap marked (OOT)."""
    d = df.sort_values("score")
    cum_bad = d["bad"].cumsum() / d["bad"].sum()
    cum_good = (1 - d["bad"]).cumsum() / (1 - d["bad"]).sum()
    gap = (cum_bad - cum_good).to_numpy()
    k = int(np.argmax(gap))
    x = d["score"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, cum_bad, color=RED, lw=2, label="Cumulative bad")
    ax.plot(x, cum_good, color=BLUE, lw=2, label="Cumulative good")
    ax.vlines(x[k], cum_good.iloc[k], cum_bad.iloc[k], color=GREY, lw=2)
    ax.annotate(f"KS = {gap[k]:.3f} at score {x[k]:.0f}", (x[k], (cum_bad.iloc[k] + cum_good.iloc[k]) / 2),
                xytext=(10, 0), textcoords="offset points")
    ax.set_xlabel("Score (points)")
    ax.set_ylabel("Cumulative share of loans")
    ax.set_title("KS curve (out-of-time)")
    _pct_axis(ax)
    ax.legend(loc="upper left")
    return _save(fig, path)


def decile_bad_rate(deciles: pd.DataFrame, path: Path) -> Path:
    """Observed bad rate by score decile (1 = worst) with mean predicted PD overlaid."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(deciles["decile"], deciles["bad_rate"], color=BLUE, label="Observed bad rate")
    ax.plot(deciles["decile"], deciles["mean_pd"], color=ORANGE, marker="o", lw=2,
            label="Mean predicted PD")
    ax.set_xticks(deciles["decile"])
    ax.set_xlabel("Score decile (1 = lowest scores / riskiest)")
    ax.set_ylabel("Bad rate")
    ax.set_title("Bad rate by score decile (out-of-time)")
    _pct_axis(ax)
    ax.legend()
    return _save(fig, path)


def calibration(cal: pd.DataFrame, path: Path) -> Path:
    """Mean predicted PD vs observed bad rate by PD decile (OOT)."""
    top = float(max(cal["mean_pd"].max(), cal["observed_bad_rate"].max())) * 1.1
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, top], [0, top], color=GREY, ls=":", label="Perfect calibration")
    ax.plot(cal["mean_pd"], cal["observed_bad_rate"], color=BLUE, marker="o", lw=2,
            label="PD deciles")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed bad rate")
    ax.set_title("Calibration (out-of-time)")
    _pct_axis(ax, "x")
    _pct_axis(ax, "y")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    return _save(fig, path)


def cutoff_tradeoff(table: pd.DataFrame, path: Path) -> Path:
    """Approval rate vs expected/observed bad rate, and vs expected loss rate."""
    t = table[table["n_approved"] > 0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(t["approval_rate"], t["expected_bad_rate"], color=ORANGE, lw=2,
              label="Expected bad rate (mean PD)")
    left.plot(t["approval_rate"], t["observed_bad_rate"], color=RED, lw=2, label="Observed bad rate")
    left.set_xlabel("Approval rate")
    left.set_ylabel("Bad rate among approved")
    left.set_title("Approval rate vs bad rate")
    _pct_axis(left, "x")
    _pct_axis(left, "y")
    left.legend()
    right.plot(t["approval_rate"], t["expected_loss_rate"], color=PURPLE, lw=2)
    right.set_xlabel("Approval rate")
    right.set_ylabel("Expected loss / exposure")
    right.set_title("Approval rate vs expected loss rate")
    _pct_axis(right, "x")
    _pct_axis(right, "y")
    return _save(fig, path)


def score_vs_grade(df: pd.DataFrame, path: Path, band: int = 50) -> Path:
    """Bad rate by LC sub-grade vs by 50-point score band (OOT benchmark view)."""
    grades = df.assign(ordinal=grade_to_ordinal(df["sub_grade"])).dropna(subset=["ordinal"])
    by_grade = grades.groupby("sub_grade")["bad"].mean().sort_index()
    bands = (df["score"] // band * band).astype(int)
    by_band = df.groupby(bands)["bad"].mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    left.bar(range(len(by_grade)), by_grade.to_numpy(), color=ORANGE)
    left.set_xticks(range(len(by_grade)), by_grade.index, rotation=90, fontsize=7)
    left.set_xlabel("LC sub-grade (A1 best, G5 worst)")
    left.set_ylabel("Observed bad rate")
    left.set_title("Bad rate by LC sub-grade")
    right.bar(range(len(by_band)), by_band.to_numpy(), color=BLUE)
    right.set_xticks(range(len(by_band)), [f"{b}-{b + band - 1}" for b in by_band.index],
                     rotation=45)
    right.set_xlabel("Scorecard score band (points)")
    right.set_title("Bad rate by scorecard band")
    _pct_axis(left)
    return _save(fig, path)


def make_all(figures_dir: Path, *, iv: pd.DataFrame, splits: dict[str, pd.DataFrame],
             oot_deciles: pd.DataFrame, oot_calibration: pd.DataFrame,
             cutoffs: pd.DataFrame) -> list[Path]:
    """Write every figure. ``splits`` maps train/test/oot to frames with score, pd, bad
    (the oot frame also needs sub_grade)."""
    oot = splits["oot"]
    figures_dir = Path(figures_dir)
    paths = [
        iv_bar(iv, figures_dir / "iv_bar.png"),
        score_distribution(oot, figures_dir / "score_distribution.png"),
        roc(splits, figures_dir / "roc.png"),
        ks_curve(oot, figures_dir / "ks_curve.png"),
        decile_bad_rate(oot_deciles, figures_dir / "decile_bad_rate.png"),
        calibration(oot_calibration, figures_dir / "calibration.png"),
        cutoff_tradeoff(cutoffs, figures_dir / "cutoff_tradeoff.png"),
        score_vs_grade(oot, figures_dir / "score_vs_grade.png"),
    ]
    logger.info("wrote %d figures to %s", len(paths), figures_dir)
    return paths
