import numpy as np
import pandas as pd

from src.market_data import normalize_yfinance_frame
from src.portfolio import (
    BollingerConfig,
    classify_core_signal,
    classify_upro_tactical_signal,
    compute_bollinger_features,
    compute_rebalance_orders,
    compute_target_weights,
)


def _ohlcv(close):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    close = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )


def test_compute_bollinger_features_adds_expected_columns():
    frame = _ohlcv(np.linspace(100, 130, 40))
    out = compute_bollinger_features(frame, BollingerConfig(window=20, num_std=2.0))

    assert {"middle", "upper", "lower", "pct_b", "bandwidth", "ema_5", "adx"}.issubset(out.columns)
    latest = out.dropna(subset=["middle", "upper", "lower", "pct_b"]).iloc[-1]
    assert latest["upper"] > latest["middle"] > latest["lower"]
    assert np.isfinite(latest["pct_b"])


def test_classify_core_signal_boundaries():
    assert classify_core_signal(pd.Series({"pct_b": -0.1}))["action"] == "BUY"
    assert classify_core_signal(pd.Series({"pct_b": 0.2}))["action"] == "WATCH_BUY"
    assert classify_core_signal(pd.Series({"pct_b": 0.5}))["action"] == "HOLD"
    assert classify_core_signal(pd.Series({"pct_b": 0.8}))["action"] == "WATCH_SELL"
    assert classify_core_signal(pd.Series({"pct_b": 1.1}))["action"] == "SELL"


def test_target_weights_default_core_and_inactive_tactical():
    weights = compute_target_weights({"GLD": 2, "SIVR": 1, "UPRO": 1}, 0.20, False)

    assert weights["GLD"] == 0.40
    assert weights["SIVR"] == 0.20
    assert weights["UPRO"] == 0.20
    assert weights["CASH"] == 0.20


def test_target_weights_moves_tactical_sleeve_to_upro_when_active():
    weights = compute_target_weights({"GLD": 2, "SIVR": 1, "UPRO": 1}, 0.20, True)

    assert weights["GLD"] == 0.40
    assert weights["SIVR"] == 0.20
    assert weights["UPRO"] == 0.40
    assert weights["CASH"] == 0.0


def test_compute_rebalance_orders_uses_integer_shares():
    orders = compute_rebalance_orders(
        quantities={"GLD": 1, "SIVR": 0, "UPRO": 0},
        cash=900,
        prices={"GLD": 100, "SIVR": 50, "UPRO": 40},
        core_ratio={"GLD": 2, "SIVR": 1, "UPRO": 1},
        tactical_weight=0.20,
        tactical_active=False,
        min_trade_value=0,
        drift_tolerance=0,
    )

    gld = orders.set_index("ticker").loc["GLD"]
    sivr = orders.set_index("ticker").loc["SIVR"]
    cash = orders.set_index("ticker").loc["CASH"]
    assert gld["shares"] == 3
    assert sivr["shares"] == 4
    assert cash["target_weight"] == 0.20


def test_upro_tactical_entry_signal_with_confirmations():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    upro = pd.DataFrame(
        {
            "close": [101, 110, 112],
            "upper": [105, 108, 109],
            "lower": [95, 96, 97],
            "middle": [100, 101, 102],
            "pct_b": [0.6, 1.1, 1.2],
            "ema_5": [100, 105, 108],
            "adx": [25, 26, 27],
        },
        index=dates,
    )
    sp500 = pd.DataFrame(
        {
            "close": [100, 102],
            "middle": [99, 100],
            "upper": [105, 106],
            "lower": [95, 96],
            "pct_b": [0.5, 0.6],
        },
        index=dates[-2:],
    )
    nasdaq = sp500.copy()
    vix = pd.DataFrame(
        {
            "close": [18, 17],
            "middle": [20, 20],
            "upper": [25, 25],
            "lower": [15, 15],
            "pct_b": [0.3, 0.2],
        },
        index=dates[-2:],
    )

    signal = classify_upro_tactical_signal(upro, sp500, nasdaq, vix, sleeve_weight=0.20)

    assert signal["action"] == "ENTER"
    assert signal["target_exposure"] == 0.20


def test_normalize_yfinance_frame_handles_title_case_columns():
    idx = pd.date_range("2024-01-01", periods=2)
    raw = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Adj Close": [1.4, 2.4],
            "Volume": [100, 200],
        },
        index=idx,
    )

    out = normalize_yfinance_frame(raw)

    assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert out.loc[idx[-1], "close"] == 2.5
