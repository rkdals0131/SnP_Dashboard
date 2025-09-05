import numpy as np
import pandas as pd

from src.targets import compute_forward_returns, parse_horizon


def test_parse_horizon_monthly():
    idx = pd.date_range("2020-01-31", periods=24, freq="M")
    assert parse_horizon("1M", idx) == 1
    assert parse_horizon("3M", idx) == 3
    assert parse_horizon("12M", idx) == 12


def test_compute_forward_returns_log_monthly():
    idx = pd.date_range("2020-01-31", periods=6, freq="M")
    spx = pd.Series([100, 105, 110, 100, 120, 144], index=idx, name="SP500")
    df = pd.DataFrame({"SP500": spx})

    out = compute_forward_returns(df, "SP500", ["1M", "3M"], method="log")
    # r_{Jan->Feb} = log(105/100)
    assert np.isclose(out.loc[idx[0], "return_1M"], np.log(105/100))
    # 3M from Jan -> Apr
    assert np.isclose(out.loc[idx[0], "return_3M"], np.log(100/100))
    # last rows should be NaN for horizons going beyond end
    assert np.isnan(out.loc[idx[-1], "return_1M"])  # last month has no future

