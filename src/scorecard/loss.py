"""Expected loss = PD x LGD x EAD.

The post-origination columns used here (funded_amnt, total_rec_prncp, recoveries,
collection_recovery_fee) are for PARAMETER CALIBRATION and BACK-TESTING ONLY. They are
never model inputs; LGD/EAD are estimated on training-window defaults only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ead_at_default(df: pd.DataFrame) -> pd.Series:
    """Outstanding principal at default: funded_amnt - total_rec_prncp (>= 0)."""
    return (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)


def net_recovery(df: pd.DataFrame) -> pd.Series:
    """Recoveries net of the collection fee (>= 0)."""
    return (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0)


def calibrate_lgd_ead(train_bads: pd.DataFrame) -> dict:
    """Pooled LGD and EAD ratio from TRAIN-split defaults only (calibration use only).

    EAD_ratio = sum(ead_at_default) / sum(funded_amnt);
    LGD = sum(ead_at_default - net_recovery) / sum(ead_at_default), clipped to [0, 1].
    """
    if len(train_bads) == 0:
        raise ValueError("no defaults to calibrate LGD/EAD on")
    ead = ead_at_default(train_bads)
    funded = float(train_bads["funded_amnt"].sum())
    total_ead = float(ead.sum())
    if funded <= 0 or total_ead <= 0:
        raise ValueError("funded amount or exposure at default is zero")
    lgd = float(np.clip((ead - net_recovery(train_bads)).sum() / total_ead, 0.0, 1.0))
    return {"lgd": lgd, "ead_ratio": total_ead / funded, "n_defaults": len(train_bads),
            "total_funded": funded, "total_ead_at_default": total_ead}


def expected_loss(pd_, loan_amnt, lgd: float, ead_ratio: float):
    """EL per loan = PD * LGD * (loan_amnt * EAD_ratio)."""
    return pd_ * lgd * loan_amnt * ead_ratio


def realised_loss(df: pd.DataFrame) -> pd.Series:
    """Realised loss per loan (back-test only): bad * max(ead_at_default - net_recovery, 0)."""
    return df["bad"] * (ead_at_default(df) - net_recovery(df)).clip(lower=0)
