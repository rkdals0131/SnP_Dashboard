import numpy as np
import pandas as pd

from src.backtesting import (
    BacktestConfig,
    close_panel,
    performance_metrics,
    run_backtest_suite,
    run_strategy_backtest,
    signal_quality_report,
)


def _market_data(n=320):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = np.linspace(100, 140, n) + np.sin(np.linspace(0, 20, n)) * 4
    data = {}
    for ticker, scale in {
        "GLD": 1.0,
        "SIVR": 0.7,
        "UPRO": 1.4,
        "SPY": 1.1,
        "^GSPC": 1.1,
        "^IXIC": 1.2,
        "^VIX": -0.08,
    }.items():
        close = 25 + base * scale if ticker != "^VIX" else 25 + np.cos(np.linspace(0, 12, n)) * 5
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000,
            },
            index=idx,
        )
        data[ticker] = frame
    return data


def test_close_panel_aligns_available_prices():
    panel = close_panel(_market_data(30))

    assert {"GLD", "SIVR", "UPRO", "SPY"}.issubset(panel.columns)
    assert len(panel) == 30


def test_run_static_strategy_backtest_returns_equity_and_trades():
    panel = close_panel(_market_data())
    result = run_strategy_backtest(panel, "static_monthly", BacktestConfig())

    assert not result["equity"].empty
    assert result["equity"].iloc[-1] > 0
    assert not result["trades"].empty


def test_backtest_suite_outputs_metrics_and_equity_columns():
    panel = close_panel(_market_data())
    metrics, equity, _ = run_backtest_suite(panel, BacktestConfig())

    assert "2:1:1 월간 리밸런싱" in set(metrics["strategy"])
    assert "SPY 보유" in set(metrics["strategy"])
    assert not equity.empty


def test_performance_metrics_has_drawdown_fields():
    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    equity = pd.Series(np.linspace(100, 130, len(idx)), index=idx)

    metrics = performance_metrics(equity)

    assert metrics["CAGR"] > 0
    assert metrics["MDD"] <= 0
    assert "Worst 1M" in metrics


def test_signal_quality_report_contains_event_rows():
    panel = close_panel(_market_data())
    report = signal_quality_report(panel, BacktestConfig())

    assert {"ticker", "signal", "events", "20D_avg"}.issubset(report.columns)
    assert set(report["ticker"]).issubset({"GLD", "SIVR", "UPRO"})
