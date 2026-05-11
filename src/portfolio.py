"""Portfolio signal and rebalancing logic for GLD/SIVR/UPRO dashboard."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

CORE_TICKERS = ("GLD", "SIVR", "UPRO")
DEFAULT_CORE_RATIO = {"GLD": 2.0, "SIVR": 1.0, "UPRO": 1.0}


@dataclass(frozen=True)
class BollingerConfig:
    """Daily Bollinger/ADX settings."""

    window: int = 20
    num_std: float = 2.0
    adx_window: int = 14
    bandwidth_lookback: int = 125


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    title_name = name.title()
    if title_name in frame.columns:
        return pd.to_numeric(frame[title_name], errors="coerce")
    raise KeyError(f"Required column missing: {name}")


def compute_bollinger_features(
    ohlcv: pd.DataFrame,
    config: BollingerConfig | None = None,
) -> pd.DataFrame:
    """Compute Bollinger Band, %B, BandWidth, EMA, and ADX columns.

    The input must contain at least close/high/low columns, case-insensitive for
    common yfinance-style title-cased names.
    """

    if ohlcv.empty:
        return pd.DataFrame()

    config = config or BollingerConfig()
    close = _column(ohlcv, "close")
    high = _column(ohlcv, "high")
    low = _column(ohlcv, "low")
    result = pd.DataFrame(index=pd.to_datetime(ohlcv.index).tz_localize(None))
    result["close"] = close.to_numpy(dtype=float)
    result["high"] = high.to_numpy(dtype=float)
    result["low"] = low.to_numpy(dtype=float)

    middle = result["close"].rolling(config.window, min_periods=config.window).mean()
    std = result["close"].rolling(config.window, min_periods=config.window).std()
    result["middle"] = middle
    result["upper"] = middle + config.num_std * std
    result["lower"] = middle - config.num_std * std
    band_range = (result["upper"] - result["lower"]).replace(0, np.nan)
    result["pct_b"] = (result["close"] - result["lower"]) / band_range
    result["bandwidth"] = band_range / result["middle"].replace(0, np.nan)
    result["bandwidth_pct"] = result["bandwidth"].rolling(
        config.bandwidth_lookback,
        min_periods=max(config.window, min(config.bandwidth_lookback, 30)),
    ).rank(pct=True)
    result["ema_5"] = result["close"].ewm(span=5, adjust=False, min_periods=5).mean()

    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = result["high"].diff()
    down_move = -result["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=result.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=result.index,
    )
    atr = true_range.rolling(config.adx_window, min_periods=config.adx_window).mean()
    plus_di = (
        100.0
        * plus_dm.rolling(config.adx_window, min_periods=config.adx_window).mean()
        / atr
    )
    minus_di = (
        100.0
        * minus_dm.rolling(config.adx_window, min_periods=config.adx_window).mean()
        / atr
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    result["adx"] = dx.rolling(config.adx_window, min_periods=config.adx_window).mean()

    return result.dropna(subset=["close"])


def latest_signal_row(features: pd.DataFrame) -> pd.Series | None:
    clean = features.dropna(subset=["close", "middle", "upper", "lower", "pct_b"])
    if clean.empty:
        return None
    return clean.iloc[-1]


def classify_core_signal(row: pd.Series | None) -> dict[str, object]:
    """Classify mean-reversion Bollinger signal for core allocation."""

    if row is None or pd.isna(row.get("pct_b", np.nan)):
        return {
            "action": "NO_DATA",
            "label": "데이터 부족",
            "score": 0,
            "tone": "neutral",
            "detail": "볼린저 밴드 계산에 필요한 일봉 데이터가 부족합니다.",
        }

    pct_b = float(row["pct_b"])
    if pct_b <= 0.0:
        return {
            "action": "BUY",
            "label": "매수",
            "score": 2,
            "tone": "positive",
            "detail": "하단 밴드 이탈. 평균회귀 관점에서 분할매수 후보입니다.",
        }
    if pct_b <= 0.2:
        return {
            "action": "WATCH_BUY",
            "label": "매수 관심",
            "score": 1,
            "tone": "positive",
            "detail": "하단 밴드 근처. 리밸런싱 매수 우선순위를 높입니다.",
        }
    if pct_b >= 1.0:
        return {
            "action": "SELL",
            "label": "축소/매도",
            "score": -2,
            "tone": "negative",
            "detail": "상단 밴드 돌파. 평균회귀 관점에서 과열 축소 후보입니다.",
        }
    if pct_b >= 0.8:
        return {
            "action": "WATCH_SELL",
            "label": "익절 관심",
            "score": -1,
            "tone": "warning",
            "detail": "상단 밴드 근처. 신규 매수보다 비중 점검이 우선입니다.",
        }
    return {
        "action": "HOLD",
        "label": "보유",
        "score": 0,
        "tone": "neutral",
        "detail": "밴드 중앙권. 목표비중 중심으로 관리합니다.",
    }


def classify_upro_tactical_signal(
    upro: pd.DataFrame,
    sp500: pd.DataFrame,
    nasdaq: pd.DataFrame,
    vix: pd.DataFrame,
    sleeve_weight: float = 0.20,
    adx_threshold: float = 20.0,
) -> dict[str, object]:
    """Classify 3-7 trading day trend-following tactical signal for UPRO."""

    frames = {"UPRO": upro, "S&P500": sp500, "NASDAQ": nasdaq, "VIX": vix}
    if any(frame is None or len(frame.dropna(subset=["close"])) < 2 for frame in frames.values()):
        return {
            "action": "NO_DATA",
            "label": "데이터 부족",
            "target_exposure": 0.0,
            "detail": "UPRO 단타 슬리브 판단에 필요한 데이터가 부족합니다.",
        }

    latest_upro = latest_signal_row(upro)
    latest_sp500 = latest_signal_row(sp500)
    latest_nasdaq = latest_signal_row(nasdaq)
    latest_vix = latest_signal_row(vix)
    if any(row is None for row in [latest_upro, latest_sp500, latest_nasdaq, latest_vix]):
        return {
            "action": "NO_DATA",
            "label": "데이터 부족",
            "target_exposure": 0.0,
            "detail": "20일 밴드 계산이 끝난 뒤부터 단타 신호를 제공합니다.",
        }

    upro_clean = upro.dropna(subset=["close", "upper", "middle", "ema_5", "adx"])
    if len(upro_clean) < 2:
        return {
            "action": "NO_DATA",
            "label": "데이터 부족",
            "target_exposure": 0.0,
            "detail": "UPRO ADX/밴드 계산에 필요한 일봉 데이터가 부족합니다.",
        }

    last_two = upro_clean.iloc[-2:]
    two_day_breakout = bool((last_two["close"] > last_two["upper"]).all())
    trend_strength = float(latest_upro.get("adx", 0.0)) >= adx_threshold
    index_confirmation = bool(
        latest_sp500["close"] > latest_sp500["middle"]
        and latest_nasdaq["close"] > latest_nasdaq["middle"]
    )
    volatility_ok = bool(latest_vix["close"] <= latest_vix["middle"])

    previous_vix_close = float(vix.dropna(subset=["close"]).iloc[-2]["close"])
    vix_change = (
        (float(latest_vix["close"]) / previous_vix_close) - 1.0
        if previous_vix_close > 0
        else 0.0
    )
    exit_trigger = bool(
        latest_upro["close"] < latest_upro["ema_5"]
        or latest_upro["close"] < latest_upro["middle"]
        or latest_vix["close"] > latest_vix["upper"]
        or vix_change >= 0.10
    )

    if exit_trigger:
        return {
            "action": "EXIT",
            "label": "단타 청산",
            "target_exposure": 0.0,
            "detail": "UPRO 단기 추세 훼손 또는 VIX 급등 조건이 발생했습니다.",
        }
    if two_day_breakout and trend_strength and index_confirmation and volatility_ok:
        return {
            "action": "ENTER",
            "label": "단타 진입",
            "target_exposure": float(sleeve_weight),
            "detail": "UPRO 상단 밴드 돌파가 지수 확인과 낮은 VIX 조건을 동반했습니다.",
        }
    return {
        "action": "WAIT",
        "label": "대기",
        "target_exposure": 0.0,
        "detail": "추세추종 진입 조건이 아직 모두 충족되지 않았습니다.",
    }


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {ticker: max(float(value), 0.0) for ticker, value in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        return {ticker: 1.0 / len(CORE_TICKERS) for ticker in CORE_TICKERS}
    return {ticker: clean.get(ticker, 0.0) / total for ticker in CORE_TICKERS}


def compute_target_weights(
    core_ratio: Mapping[str, float] = DEFAULT_CORE_RATIO,
    tactical_weight: float = 0.20,
    tactical_active: bool = False,
) -> dict[str, float]:
    """Return total-portfolio target weights for GLD/SIVR/UPRO and CASH."""

    tactical = min(max(float(tactical_weight), 0.0), 1.0)
    core_pool = 1.0 - tactical
    core_weights = normalize_weights(core_ratio)
    targets = {ticker: core_pool * core_weights[ticker] for ticker in CORE_TICKERS}
    targets["CASH"] = tactical
    if tactical_active:
        targets["UPRO"] += tactical
        targets["CASH"] = 0.0
    return targets


def compute_rebalance_orders(
    quantities: Mapping[str, float],
    cash: float,
    prices: Mapping[str, float],
    core_ratio: Mapping[str, float] = DEFAULT_CORE_RATIO,
    tactical_weight: float = 0.20,
    tactical_active: bool = False,
    min_trade_value: float = 0.0,
    drift_tolerance: float = 0.0,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Compute target values and integer ETF order suggestions."""

    qty_clean = {ticker: float(quantities.get(ticker, 0.0)) for ticker in CORE_TICKERS}
    prices_clean: dict[str, float] = {}
    missing_held = []
    for ticker in CORE_TICKERS:
        price = float(prices.get(ticker, np.nan))
        if not np.isfinite(price) or price <= 0:
            if qty_clean[ticker] > 0:
                missing_held.append(ticker)
            price = np.nan
        prices_clean[ticker] = price
    if missing_held:
        missing_text = ", ".join(missing_held)
        raise ValueError(f"Missing valid prices for held positions: {missing_text}")

    current_values = {
        ticker: qty_clean[ticker] * prices_clean[ticker]
        for ticker in CORE_TICKERS
        if np.isfinite(prices_clean[ticker])
    }
    total_value = float(cash) + sum(current_values.values())
    targets = compute_target_weights(core_ratio, tactical_weight, tactical_active)

    cost_rate = max(float(transaction_cost_bps), 0.0) / 10_000.0
    rows = []
    for ticker in CORE_TICKERS:
        price = prices_clean[ticker]
        current_value = current_values.get(ticker, 0.0)
        target_weight = targets[ticker]
        target_value = total_value * target_weight
        drift_value = target_value - current_value
        drift_pct = drift_value / total_value if total_value > 0 else 0.0

        should_trade = (
            np.isfinite(price)
            and price > 0
            and abs(drift_value) >= float(min_trade_value)
            and abs(drift_pct) >= float(drift_tolerance)
        )
        if should_trade:
            shares_abs = np.floor(abs(drift_value) / price)
            shares = int(np.sign(drift_value) * shares_abs)
        else:
            shares = 0
        trade_value = shares * price if np.isfinite(price) else 0.0

        rows.append(
            {
                "ticker": ticker,
                "quantity": qty_clean[ticker],
                "price": price,
                "current_value": current_value,
                "target_weight": target_weight,
                "target_value": target_value,
                "drift_value": drift_value,
                "drift_pct": drift_pct,
                "shares": shares,
                "trade_value": trade_value,
                "action": "BUY" if shares > 0 else "SELL" if shares < 0 else "HOLD",
            }
        )

    sell_proceeds = sum(abs(row["trade_value"]) for row in rows if row["shares"] < 0)
    sell_fee = sell_proceeds * cost_rate
    available_cash = max(float(cash) + sell_proceeds - sell_fee, 0.0)
    buy_rows = sorted(
        [row for row in rows if row["shares"] > 0],
        key=lambda row: row["drift_value"],
        reverse=True,
    )
    for row in buy_rows:
        price = row["price"]
        if not np.isfinite(price) or price <= 0:
            row["shares"] = 0
            row["trade_value"] = 0.0
            row["action"] = "HOLD"
            continue
        max_affordable = int(np.floor(available_cash / (price * (1.0 + cost_rate))))
        if row["shares"] > max_affordable:
            row["shares"] = max_affordable
            row["trade_value"] = row["shares"] * price
            row["action"] = "BUY" if row["shares"] > 0 else "HOLD"
        available_cash -= max(row["trade_value"], 0.0) * (1.0 + cost_rate)

    cash_from_sells = sum(abs(row["trade_value"]) for row in rows if row["shares"] < 0)
    cash_used_for_buys = sum(row["trade_value"] for row in rows if row["shares"] > 0)
    estimated_fee = (cash_from_sells + cash_used_for_buys) * cost_rate
    post_cash = float(cash) + cash_from_sells - cash_used_for_buys - estimated_fee
    post_values = {}
    for row in rows:
        price = row["price"]
        post_qty = row["quantity"] + row["shares"]
        post_value = post_qty * price if np.isfinite(price) else 0.0
        post_values[row["ticker"]] = post_value
        row["post_qty"] = post_qty
        row["post_value"] = post_value
    post_total = post_cash + sum(post_values.values())
    for row in rows:
        row["post_weight"] = row["post_value"] / post_total if post_total > 0 else 0.0
        row["residual_drift"] = row["target_value"] - row["post_value"]

    rows.append(
        {
            "ticker": "CASH",
            "quantity": np.nan,
            "price": 1.0,
            "current_value": float(cash),
            "target_weight": targets["CASH"],
            "target_value": total_value * targets["CASH"],
            "drift_value": total_value * targets["CASH"] - float(cash),
            "drift_pct": (
                (total_value * targets["CASH"] - float(cash)) / total_value
                if total_value > 0
                else 0.0
            ),
            "shares": 0,
            "trade_value": 0.0,
            "post_qty": np.nan,
            "post_value": post_cash,
            "post_weight": post_cash / post_total if post_total > 0 else 0.0,
            "residual_drift": total_value * targets["CASH"] - post_cash,
            "cash_from_sells": cash_from_sells,
            "cash_used_for_buys": cash_used_for_buys,
            "estimated_fee": estimated_fee,
            "post_trade_cash": post_cash,
            "action": "RESERVE",
        }
    )
    return pd.DataFrame(rows)
