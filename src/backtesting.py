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

STRESS_PERIODS = {
    "2011 유럽 재정위기": ("2011-07-01", "2011-10-31"),
    "2015-2016 원자재/위안화": ("2015-08-01", "2016-02-29"),
    "2018 Q4 급락": ("2018-10-01", "2018-12-31"),
    "2020 코로나 폭락": ("2020-02-15", "2020-04-30"),
    "2022 금리 급등": ("2022-01-01", "2022-12-31"),
    "2023 은행 위기": ("2023-03-01", "2023-05-31"),
    "2024-2026 최근 구간": ("2024-01-01", "2026-12-31"),
}


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
    max_satellite_hold_days: int = 7


def close_panel(market_data: dict[str, pd.DataFrame], adjusted: bool = True) -> pd.DataFrame:
    """Build an aligned close-price panel from normalized market data frames."""

    cols = {}
    for ticker, frame in market_data.items():
        if frame is None or frame.empty:
            continue
        price_col = "adj_close" if adjusted and "adj_close" in frame.columns else "close"
        if price_col not in frame.columns:
            continue
        series = pd.to_numeric(frame[price_col], errors="coerce").rename(ticker)
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


def _trend_following_entry_exit(
    features: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> tuple[bool, bool]:
    upro = features.get("UPRO", pd.DataFrame())
    sp500 = features.get("^GSPC", features.get("SPY", pd.DataFrame()))
    nasdaq = features.get("^IXIC", pd.DataFrame())
    vix = features.get("^VIX", pd.DataFrame())
    if any(frame.empty or date not in frame.index for frame in [upro, sp500, nasdaq, vix]):
        return False, False

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
    vix_clean = vix.loc[:date].dropna(subset=["close"])
    previous_vix_close = float(vix_clean.iloc[-2]["close"]) if len(vix_clean) >= 2 else np.nan
    vix_change = (
        (float(vix.loc[date, "close"]) / previous_vix_close) - 1.0
        if np.isfinite(previous_vix_close) and previous_vix_close > 0
        else 0.0
    )
    trend_exit = bool(
        float(upro_row["close"]) < float(upro_row.get("ema_5", np.nan))
        or float(upro_row["close"]) < float(upro_row["middle"])
        or float(vix.loc[date, "close"]) > float(vix.loc[date, "upper"])
        or vix_change >= 0.10
    )
    return trend_entry, trend_exit


def _trend_following_target(
    base: pd.Series,
    config: BacktestConfig,
    in_tactical: bool,
) -> pd.Series:
    target = base * (1.0 - config.satellite_weight)
    if in_tactical:
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
    in_tactical = False
    days_in_tactical = 0

    for idx, (day, row) in enumerate(panel.iterrows()):
        if idx > 0:
            total_value = float((holdings.reindex(row.index).fillna(0.0) * row).sum() + cash)
        else:
            total_value = cash

        signal_day = panel.index[idx - 1] if idx > 0 else None
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
            if kind == "mean_reversion" and signal_day is not None:
                target = _apply_mean_reversion_tilts(target, features, signal_day, config, False)
            elif kind == "mean_reversion_trend_filter" and signal_day is not None:
                target = _apply_mean_reversion_tilts(target, features, signal_day, config, True)
            elif kind == "trend_following":
                if signal_day is not None:
                    entry, exit_ = _trend_following_entry_exit(features, signal_day)
                    if not in_tactical and entry:
                        in_tactical = True
                        days_in_tactical = 0
                    elif in_tactical:
                        days_in_tactical += 1
                        if exit_ or days_in_tactical >= config.max_satellite_hold_days:
                            in_tactical = False
                            days_in_tactical = 0
                target = _trend_following_target(target, config, in_tactical)

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
                "tactical_active": float(in_tactical),
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


def out_of_sample_report(
    prices: pd.DataFrame,
    split_date: str = "2021-01-01",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Compare fixed-rule strategy metrics before and after a split date."""

    config = config or BacktestConfig()
    split = pd.Timestamp(split_date)
    rows = []
    for name, result in run_backtest_suite(prices, config)[2].items():
        equity = result["equity"]
        if not isinstance(equity, pd.Series) or equity.empty:
            continue
        for segment, segment_equity in {
            f"before {split.date()}": equity[equity.index < split],
            f"after {split.date()}": equity[equity.index >= split],
        }.items():
            if len(segment_equity) < 40:
                continue
            rows.append(
                {
                    "strategy_key": name,
                    "strategy": str(result["label"]),
                    "segment": segment,
                    **performance_metrics(segment_equity),
                }
            )
    return pd.DataFrame(rows)


def walk_forward_report(
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
    train_years: int = 5,
    test_years: int = 1,
    anchored: bool = False,
) -> pd.DataFrame:
    """Run fixed-rule walk-forward windows and report next-window metrics."""

    config = config or BacktestConfig()
    if prices.empty:
        return pd.DataFrame()

    start = prices.dropna(how="all").index.min()
    end = prices.dropna(how="all").index.max()
    rows = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end + pd.Timedelta(days=1):
            break
        window_start = start if anchored else train_start
        window_prices = prices.loc[window_start:test_end]
        _, _, results = run_backtest_suite(window_prices, config)
        for result in results.values():
            equity = result["equity"]
            if not isinstance(equity, pd.Series) or equity.empty:
                continue
            test_equity = equity[(equity.index > train_end) & (equity.index <= test_end)]
            if len(test_equity) < 40:
                continue
            rows.append(
                {
                    "mode": "anchored" if anchored else "rolling",
                    "train_start": window_start.date(),
                    "train_end": train_end.date(),
                    "test_start": (train_end + pd.Timedelta(days=1)).date(),
                    "test_end": test_end.date(),
                    "strategy": str(result["label"]),
                    **performance_metrics(test_equity),
                }
            )
        train_start = train_start + pd.DateOffset(years=test_years)
    return pd.DataFrame(rows)


def parameter_sensitivity_report(
    prices: pd.DataFrame,
    strategy: StrategyName = "trend_following",
    windows: tuple[int, ...] = (10, 15, 20, 30, 40, 60),
    stds: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    hold_days: tuple[int, ...] = (5, 10, 20, 40),
    base_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run a coarse robustness grid without selecting a single winner."""

    base_config = base_config or BacktestConfig()
    rows = []
    for window in windows:
        for std in stds:
            for hold in hold_days:
                config = BacktestConfig(
                    initial_capital=base_config.initial_capital,
                    core_weights=base_config.core_weights,
                    rebalance_band=base_config.rebalance_band,
                    transaction_cost_bps=base_config.transaction_cost_bps,
                    satellite_weight=base_config.satellite_weight,
                    satellite_unit_weight=base_config.satellite_unit_weight,
                    max_upro_weight=base_config.max_upro_weight,
                    min_cash_weight=base_config.min_cash_weight,
                    bb_window=window,
                    bb_std=std,
                    max_satellite_hold_days=hold,
                )
                result = run_strategy_backtest(prices, strategy, config)
                equity = result["equity"]
                if not isinstance(equity, pd.Series) or equity.empty:
                    continue
                rows.append(
                    {
                        "strategy": str(result["label"]),
                        "bb_window": window,
                        "bb_std": std,
                        "max_hold_days": hold,
                        **performance_metrics(equity),
                    }
                )
    return pd.DataFrame(rows)


def cost_sensitivity_report(
    prices: pd.DataFrame,
    costs_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 50.0),
    base_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Show how strategy metrics change as trading costs/slippage rise."""

    base_config = base_config or BacktestConfig()
    rows = []
    for cost in costs_bps:
        config = BacktestConfig(
            initial_capital=base_config.initial_capital,
            core_weights=base_config.core_weights,
            rebalance_band=base_config.rebalance_band,
            transaction_cost_bps=cost,
            satellite_weight=base_config.satellite_weight,
            satellite_unit_weight=base_config.satellite_unit_weight,
            max_upro_weight=base_config.max_upro_weight,
            min_cash_weight=base_config.min_cash_weight,
            bb_window=base_config.bb_window,
            bb_std=base_config.bb_std,
            max_satellite_hold_days=base_config.max_satellite_hold_days,
        )
        metrics, _, _ = run_backtest_suite(prices, config)
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics.insert(0, "cost_bps", cost)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def regime_report(
    prices: pd.DataFrame,
    strategy: StrategyName = "trend_following",
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Summarize daily strategy returns under simple SPY/VIX/GLD regimes."""

    config = config or BacktestConfig()
    result = run_strategy_backtest(prices, strategy, config)
    equity = result["equity"]
    if not isinstance(equity, pd.Series) or equity.empty:
        return pd.DataFrame()
    returns = equity.pct_change().dropna()
    spy_col = "SPY" if "SPY" in prices.columns else "^GSPC"
    masks = {}
    if spy_col in prices.columns:
        spy = prices[spy_col].dropna()
        spy_sma200 = spy.rolling(200, min_periods=120).mean()
        masks["SPY > 200D"] = spy > spy_sma200
        masks["SPY <= 200D"] = spy <= spy_sma200
    if "^VIX" in prices.columns:
        vix = prices["^VIX"].dropna()
        masks["VIX > 20"] = vix > 20
        masks["VIX <= 20"] = vix <= 20
    if "GLD" in prices.columns:
        gld = prices["GLD"].dropna()
        gld_sma200 = gld.rolling(200, min_periods=120).mean()
        masks["GLD > 200D"] = gld > gld_sma200
        masks["GLD <= 200D"] = gld <= gld_sma200

    rows = []
    for name, mask in masks.items():
        aligned_mask = mask.reindex(returns.index).fillna(False)
        sample = returns[aligned_mask]
        if sample.empty:
            continue
        rows.append(
            {
                "strategy": str(result["label"]),
                "regime": name,
                "days": float(len(sample)),
                "ann_return": float(sample.mean() * 252),
                "ann_vol": float(sample.std() * np.sqrt(252)),
                "hit_rate": float((sample > 0).mean()),
                "worst_day": float(sample.min()),
            }
        )
    return pd.DataFrame(rows)


def stress_period_report(
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
    periods: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Run the default suite on named historical stress windows."""

    config = config or BacktestConfig()
    periods = periods or STRESS_PERIODS
    rows = []
    for name, (start, end) in periods.items():
        period_prices = prices.loc[start:end]
        if len(period_prices.dropna(how="all")) < 20:
            continue
        metrics, _, _ = run_backtest_suite(period_prices, config)
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics.insert(0, "period", name)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
