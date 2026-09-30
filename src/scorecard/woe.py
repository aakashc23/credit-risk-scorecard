"""Weight-of-evidence binning and information value (own implementation).

Convention: ``WOE = ln(%good / %bad)`` so a higher WOE means lower risk. Good/bad counts
get ``WOE_SMOOTHING`` added per bin to avoid ``log(0)``; percentages are taken over the
smoothed totals, so they sum to one.

A fitted feature is a plain dict (JSON friendly)::

    {"kind": "numeric", "edges": [...], "bins": [{label, n, bads, bad_rate, woe, iv_contrib}],
     "iv": float}

Numeric bins are the intervals ``(-inf, e1], (e1, e2], ..., (ek, inf)`` followed by an
optional "Missing" bin. Categorical bins carry a ``categories`` list instead of ``edges``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard.config import (
    CATEGORICAL_FEATURES,
    MAX_COARSE_BINS,
    MAX_FINE_BINS,
    MIN_BIN_SHARE,
    MIN_CATEGORY_SHARE,
    SUSPICIOUS_IV,
    WOE_SMOOTHING,
)

OTHER = "Other"
MISSING = "Missing"


# --------------------------------------------------------------------------- core maths
def woe_iv(n: np.ndarray, bads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WOE and per-bin IV contribution from bin counts (smoothed)."""
    n, bads = np.asarray(n, float), np.asarray(bads, float)
    good_adj = n - bads + WOE_SMOOTHING
    bad_adj = bads + WOE_SMOOTHING
    pct_good, pct_bad = good_adj / good_adj.sum(), bad_adj / bad_adj.sum()
    woe = np.log(pct_good / pct_bad)
    return woe, (pct_good - pct_bad) * woe


def _bin_rows(labels: list[str], n: list[float], bads: list[float], extra: list[dict]) -> list[dict]:
    woe, iv = woe_iv(np.array(n), np.array(bads))
    rows = []
    for i, label in enumerate(labels):
        rows.append({"label": label, "n": int(n[i]), "bads": int(bads[i]),
                     "bad_rate": float(bads[i] / n[i]) if n[i] else float("nan"),
                     "woe": float(woe[i]), "iv_contrib": float(iv[i]), **extra[i]})
    return rows


def _merge_at(i: int, n: list, bads: list, keys: list) -> None:
    """Merge bin ``i`` into bin ``i + 1`` in place (``keys`` are edges or category groups)."""
    n[i + 1] += n[i]
    bads[i + 1] += bads[i]
    if keys and isinstance(keys[0], list):  # categorical: union the category groups
        keys[i + 1] = keys[i] + keys[i + 1]
    del n[i], bads[i], keys[i]  # numeric: the edge between bins i and i + 1 disappears


def _rates(n: list, bads: list) -> np.ndarray:
    return np.array(bads, float) / np.array(n, float)


def _merge_to_limit(n: list, bads: list, keys: list) -> None:
    """Merge the adjacent pair with the closest bad rates until <= MAX_COARSE_BINS remain."""
    while len(n) > MAX_COARSE_BINS:
        _merge_at(int(np.argmin(np.abs(np.diff(_rates(n, bads))))), n, bads, keys)


# --------------------------------------------------------------------------- numeric
def _interval_label(edges: list[float], i: int) -> str:
    if not edges:
        return "All"
    if i == 0:
        return f"<= {edges[0]:.6g}"
    if i == len(edges):
        return f"> {edges[-1]:.6g}"
    return f"({edges[i - 1]:.6g}, {edges[i]:.6g}]"


def _fine_edges(x: np.ndarray) -> list[float]:
    """Quantile cut points (actual data values) giving at most MAX_FINE_BINS bins."""
    qs = np.linspace(0, 1, MAX_FINE_BINS + 1)[1:-1]
    edges = np.unique(np.quantile(x, qs, method="lower"))
    return [float(e) for e in edges[edges < x.max()]]


def _merge_small_bins(n: list, bads: list, edges: list, min_n: float) -> None:
    """Merge every bin under ``min_n`` rows into its smaller neighbour."""
    while len(n) > 1 and min(n) < min_n:
        i = int(np.argmin(n))
        if i == 0 or (i < len(n) - 1 and n[i + 1] <= n[i - 1]):
            _merge_at(i, n, bads, edges)  # into the right neighbour
        else:
            _merge_at(i - 1, n, bads, edges)  # into the left neighbour


def _make_monotone(n: list, bads: list, edges: list, direction: int) -> None:
    """Merge adjacent bins until bad rate moves in ``direction`` (+1 up, -1 down) only."""
    while True:
        violations = np.flatnonzero(direction * np.diff(_rates(n, bads)) < 0)
        if len(violations) == 0:
            return
        _merge_at(int(violations[0]), n, bads, edges)


def fit_numeric(x: pd.Series, y: np.ndarray) -> dict:
    """Quantile fine bins -> min-size merge -> monotone merge -> <= MAX_COARSE_BINS."""
    from scipy.stats import spearmanr  # imported here so scoring (the app) does not need scipy

    values = x.to_numpy(float)
    miss = np.isnan(values)
    xv, yv = values[~miss], y[~miss]
    edges: list[float] = _fine_edges(xv) if len(xv) else []
    idx = np.searchsorted(edges, xv, side="left")
    k = len(edges) + 1
    n = np.bincount(idx, minlength=k).astype(float).tolist()
    bads = np.bincount(idx, weights=yv, minlength=k).tolist()

    _merge_small_bins(n, bads, edges, MIN_BIN_SHARE * len(xv))
    if len(n) > 1:
        constant_y = yv.min() == yv.max()
        corr = 0.0 if constant_y else spearmanr(np.searchsorted(edges, xv), yv)[0]
        direction = -1 if corr < 0 else 1
        _make_monotone(n, bads, edges, direction)
        _merge_to_limit(n, bads, edges)

    labels = [_interval_label(edges, i) for i in range(len(n))]
    extra = [{}] * len(n)
    if miss.any():
        labels.append(MISSING)
        n.append(float(miss.sum()))
        bads.append(float(y[miss].sum()))
        extra = [{}] * (len(n) - 1) + [{"is_missing": True}]
    rows = _bin_rows(labels, n, bads, extra)
    return {"kind": "numeric", "edges": edges, "bins": rows,
            "iv": float(sum(r["iv_contrib"] for r in rows))}


# --------------------------------------------------------------------------- categorical
def _as_labels(x: pd.Series) -> pd.Series:
    return x.astype(object).where(x.notna(), MISSING).astype(str)


def fit_categorical(x: pd.Series, y: np.ndarray) -> dict:
    """Pool rare categories into "Other", order by bad rate, merge to <= MAX_COARSE_BINS."""
    frame = pd.DataFrame({"cat": _as_labels(x).to_numpy(), "y": y})
    stats = frame.groupby("cat")["y"].agg(["size", "sum"])
    rare = stats["size"] < MIN_CATEGORY_SHARE * len(frame)
    groups: dict[str, list] = {c: [int(r["size"]), float(r["sum"])] for c, r in stats[~rare].iterrows()}
    if rare.any():
        pooled = groups.setdefault(OTHER, [0, 0.0])
        pooled[0] += int(stats.loc[rare, "size"].sum())
        pooled[1] += float(stats.loc[rare, "sum"].sum())

    order = sorted(groups, key=lambda c: groups[c][1] / groups[c][0])
    keys = [[c] for c in order]
    n = [float(groups[c][0]) for c in order]
    bads = [float(groups[c][1]) for c in order]
    _merge_to_limit(n, bads, keys)

    rows = _bin_rows([" | ".join(k) for k in keys], n, bads, [{"categories": k} for k in keys])
    return {"kind": "categorical", "bins": rows, "iv": float(sum(r["iv_contrib"] for r in rows))}


def fit_bins(x: pd.Series, y: np.ndarray, kind: str | None = None) -> dict:
    """Fit WOE bins for one feature. ``kind`` is inferred from dtype when omitted."""
    y = np.asarray(y, float)
    if kind is None:
        kind = "numeric" if pd.api.types.is_numeric_dtype(x) else "categorical"
    if kind == "numeric":
        return fit_numeric(x, y)
    if kind == "categorical":
        return fit_categorical(x, y)
    raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------------------- transform
def _numeric_index(spec: dict, x: pd.Series) -> np.ndarray:
    """Bin position per row; missing values get the Missing bin (or -1 if there is none)."""
    values = x.to_numpy(float)
    idx = np.searchsorted(spec["edges"], values, side="left")
    n_interval = len(spec["edges"]) + 1
    has_missing = len(spec["bins"]) > n_interval
    return np.where(np.isnan(values), n_interval if has_missing else -1, idx)


def _category_lookup(spec: dict, key: str) -> dict[str, float]:
    return {c: b[key] for b in spec["bins"] for c in b["categories"]}


def transform_feature(spec: dict, x: pd.Series) -> np.ndarray:
    """WOE per row. Unseen categories -> "Other" bin, else 0; missing with no bin -> 0."""
    if spec["kind"] == "numeric":
        woe = np.append([b["woe"] for b in spec["bins"]], 0.0)  # last slot = index -1
        return woe[_numeric_index(spec, x)]
    lookup = _category_lookup(spec, "woe")
    return _as_labels(x).map(lookup).fillna(lookup.get(OTHER, 0.0)).to_numpy(float)


def label_feature(spec: dict, x: pd.Series) -> pd.Series:
    """Bin label per row (same unseen-category rule as ``transform_feature``)."""
    if spec["kind"] == "numeric":
        labels = np.append([b["label"] for b in spec["bins"]], "Unbinned")
        return pd.Series(labels[_numeric_index(spec, x)], index=x.index)
    lookup = _category_lookup(spec, "label")
    fallback = lookup.get(OTHER, "Unbinned")
    return _as_labels(x).map(lookup).fillna(fallback)


# --------------------------------------------------------------------------- binner
class WoeBinner:
    """Fit WOE bins for many columns and transform data to WOE values."""

    def __init__(self, bins: dict[str, dict] | None = None) -> None:
        self.bins_: dict[str, dict] = bins or {}

    def fit(self, df: pd.DataFrame, y, columns: list[str]) -> WoeBinner:
        y = np.asarray(y, float)
        for col in columns:
            numeric = pd.api.types.is_numeric_dtype(df[col]) and col not in CATEGORICAL_FEATURES
            self.bins_[col] = fit_bins(df[col], y, "numeric" if numeric else "categorical")
        return self

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or list(self.bins_)
        data = {c: transform_feature(self.bins_[c], df[c]) for c in cols}
        return pd.DataFrame(data, index=df.index)

    def bin_labels(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or list(self.bins_)
        return pd.DataFrame({c: label_feature(self.bins_[c], df[c]) for c in cols})

    def bin_table(self) -> pd.DataFrame:
        """One row per bin: feature, bin, n, bads, bad_rate, woe, iv_contrib."""
        rows = [{"feature": f, "bin": b["label"], "n": b["n"], "bads": b["bads"],
                 "bad_rate": b["bad_rate"], "woe": b["woe"], "iv_contrib": b["iv_contrib"]}
                for f, spec in self.bins_.items() for b in spec["bins"]]
        return pd.DataFrame(rows, columns=["feature", "bin", "n", "bads", "bad_rate", "woe",
                                           "iv_contrib"])

    def iv_table(self) -> pd.DataFrame:
        """Information value per feature, strongest first."""
        rows = [{"feature": f, "kind": s["kind"], "n_bins": len(s["bins"]), "iv": s["iv"],
                 "suspicious": s["iv"] > SUSPICIOUS_IV} for f, s in self.bins_.items()]
        table = pd.DataFrame(rows, columns=["feature", "kind", "n_bins", "iv", "suspicious"])
        return table.sort_values("iv", ascending=False, ignore_index=True)

    def to_dict(self) -> dict:
        return {"bins": self.bins_}

    @classmethod
    def from_dict(cls, payload: dict) -> WoeBinner:
        return cls(payload["bins"])

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path | str) -> WoeBinner:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
