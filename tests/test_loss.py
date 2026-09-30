import pandas as pd
import pytest

from scorecard.loss import calibrate_lgd_ead, expected_loss, realised_loss


def _defaults() -> pd.DataFrame:
    return pd.DataFrame({
        "funded_amnt": [10000.0, 20000.0],
        "total_rec_prncp": [4000.0, 12000.0],
        "recoveries": [600.0, 2000.0],
        "collection_recovery_fee": [100.0, 400.0],
        "bad": [1, 1],
    })


def test_lgd_and_ead_hand_computed():
    params = calibrate_lgd_ead(_defaults())
    # EAD at default: 6000 + 8000 = 14000; net recoveries: 500 + 1600 = 2100
    assert params["ead_ratio"] == pytest.approx(14000 / 30000)
    assert params["lgd"] == pytest.approx((14000 - 2100) / 14000)
    assert params["n_defaults"] == 2


def test_negative_components_are_clipped():
    df = _defaults()
    df.loc[0, "total_rec_prncp"] = 11000.0  # overpaid -> EAD clipped to 0
    df.loc[1, "collection_recovery_fee"] = 5000.0  # fee > recoveries -> net recovery 0
    params = calibrate_lgd_ead(df)
    assert params["total_ead_at_default"] == pytest.approx(8000.0)
    assert params["lgd"] == pytest.approx((8000 + 0 - 500) / 8000)  # portfolio-level sums


def test_lgd_clipped_to_unit_interval():
    df = _defaults()
    df["recoveries"] = 1e6
    assert calibrate_lgd_ead(df)["lgd"] == 0.0


def test_no_defaults_raises():
    with pytest.raises(ValueError):
        calibrate_lgd_ead(_defaults().iloc[0:0])


def test_expected_loss_hand_computed():
    el = expected_loss(pd.Series([0.1, 0.2]), pd.Series([10000.0, 5000.0]), lgd=0.8, ead_ratio=0.5)
    assert el.tolist() == pytest.approx([0.1 * 0.8 * 10000 * 0.5, 0.2 * 0.8 * 5000 * 0.5])


def test_realised_loss():
    df = pd.concat([_defaults(), _defaults().assign(bad=0)], ignore_index=True)
    loss = realised_loss(df)
    assert loss.iloc[:2].tolist() == pytest.approx([6000 - 500, 8000 - 1600])
    assert (loss.iloc[2:] == 0).all()  # good loans have no realised loss
