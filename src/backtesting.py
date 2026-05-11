"""Backtests and signal diagnostics for the GLD/SIVR/UPRO dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .portfolio import BollingerConfig, compute_bollinger_features

RebalanceFrequency = Literal["monthly", "quarterly"]
StrategyName = Literal[
    "static_monthly",
    "static_quarterly",
    "mean_reversion",
    "trend_following",
    "mean_reversion_trend_filter",
    "spy_buy_hold",
    "unlevered_monthly",
]


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    core_weights: tuple[float, float, float] = (0.50, 0.25, 0.25)
    rebalance_band: float = 0.05
    transaction_cost_bps: float = 5.0
    satellite_weight: float = 0.20
    satellite_unit_weight: float = 0.05
    max_upro_weight: float = 0.35
    min_cash_weight: float = 0.05
    bb_window: int = 20
    bb_std: float = 2.0


def close_panel(market_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an aligned close-price panel from normalized market data frames."""

    cols = {}
    for ticker, frame in market_data.items():
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        series = pd.to_numeric(frame["close"], errors="coerce").rename(ticker)
        cols[ticker] = series
    if not cols:
        return pd.DataFrame()
    panel = pd.concat(cols.values(), axis=1).sort_index().ffill()
    return panel.dropna(how="all")


def _strategy_specs() -> dict[StrategyName, dict[str, object]]:
    return {
        "static_monthly": {
            "label": "2:1:1 월간 리밸런싱",
            "kind": "core",
            "frequency": "monthly",
        },
        "static_quarterly": {
            "label": "2:1:1 분기 리밸런싱",
            "kind": "core",
            "frequency": "quarterly",
        },
        "mean_reversion": {
            "label": "볼린저 평균회귀",
            "kind": "mean_reversion",
            "frequency": "monthly",
        },
        "trend_following": {
            "label": "UPRO 추세추종 위성",
            "kind": "trend_following",
            "frequency": "monthly",
        },
        "mean_reversion_trend_filter": {
            "label": "평균회귀 + 200D 필터",
            "kind": "mean_reversion_trend_filter",
            "frequency": "monthly",
        },
        "spy_buy_hold": {
            "label": "SPY 보유",
            "kind": "buy_hold",
            "frequency": "monthly",
        },
        "unlevered_monthly": {
            "label": "GLD/SPY/SIVR 월간",
            "kind": "unlevered",
            "frequency": "monthly",
        },
    }


def _rebalance_mask(index: pd.DatetimeIndex, frequency: RebalanceFrequency) -> pd.Series:
    periods = index.to_period("Q" if frequency == "quarterly" else "M")
    mask = pd.Series(periods != pd.Series(periods, index=index).shift(1).to_numpy(), index=index)
    if not mask.empty:
        mask.iloc[0] = True
    return mask


def _trade_to_weights(
    holdings: pd.Series,
    cash: float,
    prices: pd.Series,
    target_weights: pd.Series,
    total_value: float,
    transaction_cost_bps: float,
) -> tuple[pd.Series, float, float]:
    target_values = target_weights.reindex(prices.index).fillna(0.0) * total_value
    current_values = holdings.reindex(prices.index).fillna(0.0) * prices
    trade_values = target_values - current_values
    cost = float(trade_values.abs().sum() * transaction_cost_bps / 10_000.0)
    adjusted_total = max(total_value - cost, 0.0)
    target_values = target_weights.reindex(prices.index).fillna(0.0) * adjusted_total
    new_holdings = target_values / prices.replace(0, np.nan)
    new_holdings = new_holdings.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    new_cash = adjusted_total * max(1.0 - float(target_weights.sum()), 0.0)
    turnover = float(trade_values.abs().sum() / total_value) if total_value > 0 else 0.0
    return new_holdings, new_cash, turnover


def _core_target_weights(config: BacktestConfig) -> pd.Series:
    return pd.Series(
        {
            "GLD": config.core_weights[0],
            "UPRO": config.core_weights[1],
            "SIVR": config.core_weights[2],
        },
        dtype=float,
    )


def _apply_mean_reversion_tilts(
    base: pd.Series,
    features: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    config: BacktestConfig,
    use_trend_filter: bool,
) -> pd.Series:
    target = base.copy()
    unit = config.satellite_unit_weight
    for ticker in ["GLD", "SIVR", "UPRO"]:
        frame = features.get(ticker, pd.DataFrame())
        if frame.empty or date not in frame.index:
            continue
        row = frame.loc[date]
        if pd.isna(row.get("pct_b", np.nan)):
            continue
        pct_b = float(row["pct_b"])
        sma200 = float(frame["close"].rolling(200, min_periods=120).mean().loc[date])
        trend_ok = not use_trend_filter or ticker != "UPRO" or float(row["close"]) > sma200
        if pct_b <= 0.0 and trend_ok:
            target[ticker] = target.get(ticker, 0.0) + unit
        elif pct_b >= 1.0:
            target[ticker] = max(target.get(ticker, 0.0) - unit, 0.0)

    risky_sum = float(target.sum())
    max_risky = 1.0 - config.min_cash_weight
    if risky_sum > max_risky:
        target = target * (max_risky / risky_sum)
    return target


def _trend_following_target(
    base: pd.Series,
    features: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    config: BacktestConfig,
) -> pd.Series:
    target = base * (1.0 - config.satellite_weight)
    upro = features.get("UPRO", pd.DataFrame())
    sp500 = features.get("^GSPC", features.get("SPY", pd.DataFrame()))
    nasdaq = features.get("^IXIC", pd.DataFrame())
    vix = features.get("^VIX", pd.DataFrame())
    if any(frame.empty or date not in frame.index for frame in [upro, sp500, nasdaq, vix]):
        return target

    upro_row = upro.loc[date]
    prior = upro.loc[:date].tail(2)
    sma60 = float(upro["close"].rolling(60, min_periods=40).mean().loc[date])
    high20 = float(upro["close"].rolling(20, min_periods=20).max().shift(1).loc[date])
    trend_entry = bool(
        len(prior) == 2
        and (prior["close"] > prior["upper"]).all()
        and float(upro_row.get("adx", 0.0)) >= 20.0
        and float(upro_row["close"]) > sma60
        and float(upro_row["close"]) >= high20
        and float(sp500.loc[date, "close"]) > float(sp500.loc[date, "middle"])
        and float(nasdaq.loc[date, "close"]) > float(nasdaq.loc[date, "middle"])
        and float(vix.loc[date, "close"]) <= float(vix.loc[date, "middle"])
    )
    if trend_entry:
        target["UPRO"] = min(
            target.get("UPRO", 0.0) + config.satellite_weight,
            config.max_upro_weight,
        )
    return target


def run_strategy_backtest(
    prices: pd.DataFrame,
    strategy: StrategyName,
    config: BacktestConfig | None = None,
) -> dict[str, object]:
    """Run one deterministic daily close-to-close backtest."""

    config = config or BacktestConfig()
    specs = _strategy_specs()
    spec = specs[strategy]
    required = ["GLD", "UPRO", "SIVR"]
    if strategy == "spy_buy_hold":
        required = ["SPY" if "SPY" in prices.columns else "^GSPC"]
    elif strategy == "unlevered_monthly":
        required = ["GLD", "SPY" if "SPY" in prices.columns else "^GSPC", "SIVR"]

    panel = prices.loc[:, [col for col in required if col in prices.columns]].dropna()
    if len(panel) < max(config.bb_window + 5, 40):
        return {"label": spec["label"], "equity": pd.Series(dtype=float), "trades": pd.DataFrame()}

    bb_config = BollingerConfig(window=config.bb_window, num_std=config.bb_std)
    features = {
        ticker: compute_bollinger_features(
            pd.DataFrame({"high": prices[ticker], "low": prices[ticker], "close": prices[ticker]}),
            bb_config,
        )
        for ticker in prices.columns
        if ticker in {"GLD", "UPRO", "SIVR", "SPY", "^GSPC", "^IXIC", "^VIX"}
    }

    holdings = pd.Series(0.0, index=panel.columns)
    cash = float(config.initial_capital)
    equity_rows = []
    trade_rows = []
    rebalance_days = _rebalance_mask(panel.index, spec["frequency"])

    for idx, (day, row) in enumerate(panel.iterrows()):
        if idx > 0:
            total_value = float((holdings.reindex(row.index).fillna(0.0) * row).sum() + cash)
        else:
            total_value = cash

        target = pd.Series(0.0, index=panel.columns)
        kind = spec["kind"]
        if kind == "buy_hold":
            target.iloc[0] = 1.0
        elif kind == "unlevered":
            target.iloc[0] = 0.50
            target.iloc[1] = 0.25
            target.iloc[2] = 0.25
        else:
            target = _core_target_weights(config).reindex(panel.columns).fillna(0.0)
            if kind == "mean_reversion":
                target = _apply_mean_reversion_tilts(target, features, day, config, False)
            elif kind == "mean_reversion_trend_filter":
                target = _apply_mean_reversion_tilts(target, features, day, config, True)
            elif kind == "trend_following":
                target = _trend_following_target(target, features, day, config)

        current_weights = (
            holdings.reindex(row.index).fillna(0.0) * row / total_value
            if total_value > 0
            else pd.Series(0.0, index=row.index)
        )
        drift = float((current_weights.reindex(target.index).fillna(0.0) - target).abs().max())
        should_rebalance = bool(rebalance_days.loc[day] or drift >= config.rebalance_band)
        if should_rebalance:
            holdings, cash, turnover = _trade_to_weights(
                holdings.reindex(row.index).fillna(0.0),
                cash,
                row,
                target,
                total_value,
                config.transaction_cost_bps,
            )
            total_value = float((holdings.reindex(row.index).fillna(0.0) * row).sum() + cash)
            trade_rows.append({"date": day, "turnover": turnover, "cash": cash})

        equity_rows.append(
            {
                "date": day,
                "equity": total_value,
                "cash": cash,
                **{
                    f"{ticker}_weight": (
                        float(holdings.get(ticker, 0.0) * row[ticker] / total_value)
                        if total_value > 0 and ticker in row.index
                        else 0.0
                    )
                    for ticker in row.index
                },
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("date")
    trades = pd.DataFrame(trade_rows)
    return {"label": spec["label"], "equity": equity["equity"], "details": equity, "trades": trades}


def performance_metrics(equity: pd.Series) -> dict[str, float]:
    clean = equity.dropna()
    if len(clean) < 2:
        return {
            "CAGR": np.nan,
            "MDD": np.nan,
            "Vol": np.nan,
            "Sharpe": np.nan,
            "Worst 1M": np.nan,
            "Worst 3M": np.nan,
            "Worst 1Y": np.nan,
            "Recovery Days": np.nan,
        }
    returns = clean.pct_change().dropna()
    years = max((clean.index[-1] - clean.index[0]).days / 365.25, 1e-9)
    cagr = (clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0
    drawdown = clean / clean.cummax() - 1.0
    mdd = float(drawdown.min())
    vol = float(returns.std() * np.sqrt(252))
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else np.nan
    underwater = drawdown < 0
    recovery_days = 0
    if underwater.any():
        recovery_days = int(underwater.groupby((~underwater).cumsum()).sum().max())
    return {
        "CAGR": float(cagr),
        "MDD": mdd,
        "Vol": vol,
        "Sharpe": sharpe,
        "Worst 1M": float(clean.pct_change(21).min()),
        "Worst 3M": float(clean.pct_change(63).min()),
        "Worst 1Y": float(clean.pct_change(252).min()),
        "Recovery Days": float(recovery_days),
    }


def run_backtest_suite(
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    """Run the default comparison suite and return metrics, equity, raw results."""

    config = config or BacktestConfig()
    results = {
        strategy: run_strategy_backtest(prices, strategy, config)
        for strategy in _strategy_specs().keys()
    }
    metrics_rows = []
    equity_cols = {}
    for _strategy, result in results.items():
        equity = result["equity"]
        if not isinstance(equity, pd.Series) or equity.empty:
            continue
        row = {"strategy": str(result["label"]), **performance_metrics(equity)}
        trades = result.get("trades", pd.DataFrame())
        row["Trades"] = float(len(trades)) if isinstance(trades, pd.DataFrame) else 0.0
        row["Avg Turnover"] = (
            float(trades["turnover"].mean())
            if isinstance(trades, pd.DataFrame) and not trades.empty
            else 0.0
        )
        metrics_rows.append(row)
        equity_cols[str(result["label"])] = equity / equity.iloc[0]

    metrics = pd.DataFrame(metrics_rows)
    equity = pd.DataFrame(equity_cols).dropna(how="all")
    return metrics, equity, results


def signal_quality_report(
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
    horizons: tuple[int, ...] = (5, 10, 20, 40),
) -> pd.DataFrame:
    """Evaluate post-signal forward returns for Bollinger/trend conditions."""

    config = config or BacktestConfig()
    bb_config = BollingerConfig(window=config.bb_window, num_std=config.bb_std)
    rows = []
    for ticker in ["GLD", "SIVR", "UPRO"]:
        if ticker not in prices.columns:
            continue
        close = prices[ticker].dropna()
        frame = compute_bollinger_features(
            pd.DataFrame({"high": close, "low": close, "close": close}),
            bb_config,
        )
        if len(frame) < 220:
            continue
        sma20_slope = frame["middle"].diff()
        sma60 = frame["close"].rolling(60, min_periods=40).mean()
        sma200 = frame["close"].rolling(200, min_periods=120).mean()
        high20 = frame["close"].rolling(20, min_periods=20).max().shift(1)
        upper_breakout = (frame["close"] > frame["upper"]) & (
            frame["close"].shift(1) <= frame["upper"].shift(1)
        )
        lower_breakdown = (frame["close"] < frame["lower"]) & (
            frame["close"].shift(1) >= frame["lower"].shift(1)
        )
        lower_reentry = (frame["close"] >= frame["lower"]) & (
            frame["close"].shift(1) < frame["lower"].shift(1)
        )
        signals = {
            "상단 돌파": upper_breakout,
            "하단 이탈": lower_breakdown,
            "하단 재진입": lower_reentry,
            "추세추종 후보": (
                (frame["close"] > frame["upper"])
                & (sma20_slope > 0)
                & (frame["close"] > sma60)
                & (frame["close"] >= high20)
            ),
            "200D 위 하단 이탈": (frame["close"] < frame["lower"]) & (frame["close"] > sma200),
            "200D 아래 하단 이탈": (frame["close"] < frame["lower"]) & (frame["close"] <= sma200),
        }
        for name, mask in signals.items():
            event_dates = frame.index[mask.fillna(False)]
            row = {"ticker": ticker, "signal": name, "events": float(len(event_dates))}
            for horizon in horizons:
                forward = frame["close"].shift(-horizon) / frame["close"] - 1.0
                values = forward.reindex(event_dates).dropna()
                row[f"{horizon}D_avg"] = float(values.mean()) if not values.empty else np.nan
                row[f"{horizon}D_win_rate"] = (
                    float((values > 0).mean()) if not values.empty else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)
