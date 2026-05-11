"""Market data helpers for the portfolio dashboard."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from structlog import get_logger

from .data_sources import FearGreedSource

logger = get_logger()


DEFAULT_PRICE_TICKERS = ["GLD", "SIVR", "UPRO", "SPY", "^GSPC", "^IXIC", "^VIX"]
TICKER_LABELS = {
    "GLD": "GLD",
    "SIVR": "SIVR",
    "UPRO": "UPRO",
    "SPY": "SPY",
    "^GSPC": "S&P500",
    "^IXIC": "NASDAQ",
    "^VIX": "VIX",
}


def normalize_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance OHLCV data to lowercase columns."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).strip() for col in df.columns]

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    ohlcv_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    keep = [col for col in ohlcv_cols if col in df.columns]
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.replace('^', '')}.parquet"


def _load_cached_price(cache_dir: Path, ticker: str) -> pd.DataFrame:
    path = _cache_path(cache_dir, ticker)
    if not path.exists():
        return pd.DataFrame()
    try:
        return normalize_yfinance_frame(pd.read_parquet(path))
    except Exception as exc:
        logger.warning("price_cache_load_failed", ticker=ticker, error=str(exc))
        return pd.DataFrame()


def _save_cached_price(cache_dir: Path, ticker: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, ticker)
    try:
        existing = _load_cached_price(cache_dir, ticker)
        combined = pd.concat([existing, frame]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(path, compression="snappy")
    except Exception as exc:
        logger.warning("price_cache_save_failed", ticker=ticker, error=str(exc))


def fetch_yfinance_prices(
    tickers: Iterable[str] = DEFAULT_PRICE_TICKERS,
    start: date | None = None,
    end: date | None = None,
    cache_dir: Path = Path("data/raw/yfinance"),
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV data with per-ticker Parquet caching."""

    import yfinance as yf

    end_date = end or date.today()
    start_date = start or (end_date - timedelta(days=365 * 3))
    frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        cached = _load_cached_price(cache_dir, ticker) if use_cache else pd.DataFrame()
        stale = True
        if not cached.empty:
            latest = cached.index.max().date()
            earliest = cached.index.min().date()
            stale = latest < end_date - timedelta(days=3) or earliest > start_date
            frames[ticker] = cached.loc[
                (cached.index.date >= start_date) & (cached.index.date <= end_date)
            ]

        if use_cache and not stale and not frames[ticker].empty:
            continue

        try:
            downloaded = yf.download(
                ticker,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            normalized = normalize_yfinance_frame(downloaded)
            _save_cached_price(cache_dir, ticker, normalized)
            if not normalized.empty:
                frames[ticker] = normalized.loc[
                    (normalized.index.date >= start_date) & (normalized.index.date <= end_date)
                ]
        except Exception as exc:
            logger.warning("yfinance_fetch_failed", ticker=ticker, error=str(exc))
            frames.setdefault(ticker, cached)

    return frames


def fetch_cnn_fear_greed() -> dict[str, object]:
    """Fetch latest CNN Fear & Greed value through the optional package."""

    try:
        import fear_greed

        payload = fear_greed.get()
        if isinstance(payload, dict):
            score = payload.get("score") or payload.get("value") or payload.get("fear_and_greed")
            rating = payload.get("rating") or payload.get("status") or payload.get("classification")
        else:
            score = fear_greed.get_score()
            rating = fear_greed.get_rating()
        return {
            "score": float(score) if score is not None and not pd.isna(score) else np.nan,
            "rating": str(rating) if rating is not None else "",
            "source": "CNN Fear & Greed",
            "ok": True,
        }
    except Exception as exc:
        logger.warning("cnn_fear_greed_fetch_failed", error=str(exc))
        return {"score": np.nan, "rating": "", "source": "CNN Fear & Greed", "ok": False}


def fetch_crypto_fear_greed(
    start: date | None = None,
    end: date | None = None,
) -> dict[str, object]:
    """Fetch latest Alternative.me Crypto Fear & Greed value."""

    end_date = end or date.today()
    start_date = start or (end_date - timedelta(days=14))
    try:
        source = FearGreedSource(provider="alternative")
        df = source.fetch(["CRYPTO_FNG"], start_date, end_date)
        if df.empty:
            raise RuntimeError("empty crypto fear-greed response")
        latest = df.sort_index().iloc[-1]
        return {
            "score": float(latest["value"]),
            "rating": str(latest.get("value_classification", "")),
            "source": "Alternative.me Crypto Fear & Greed",
            "ok": True,
        }
    except Exception as exc:
        logger.warning("crypto_fear_greed_fetch_failed", error=str(exc))
        return {
            "score": np.nan,
            "rating": "",
            "source": "Alternative.me Crypto Fear & Greed",
            "ok": False,
        }
