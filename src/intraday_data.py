"""S&P 500 실시간 인트라데이 데이터 수집 모듈

yfinance를 사용하여 1분, 5분, 15분, 30분, 1시간, 1일 단위의 실시간 데이터를 제공합니다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Literal

import pandas as pd
import yfinance as yf
from structlog import get_logger

logger = get_logger()

IntervalType = Literal["1m", "5m", "15m", "30m", "1h", "1d"]


def fetch_intraday_data(
    symbol: str = "^GSPC",  # S&P 500 ticker
    interval: IntervalType = "1h",
    period: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """실시간 인트라데이  데이터 가져오기

    Args:
        symbol: 티커 심볼 (기본: ^GSPC = S&P 500)
        interval: 데이터 간격 (1m, 5m, 15m, 30m, 1h, 1d)
        period: 조회 기간 (예: "1d", "5d", "1mo", "3mo", "1y", "2y", "5y", "max")
        start: 시작 날짜/시간 (period 대신 사용 가능)
        end: 종료 날짜/시간 (period 대신 사용 가능)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume

    Notes:
        - 1분 데이터는 최근 7일간만 사용 가능
        - 5분~1시간 데이터는 최근 60일간 사용 가능
        - 1일 데이터는 제한 없음
    """
    # 기본 period 설정
    if period is None and start is None:
        if interval == "1m":
            period = "1d"
        elif interval in ["5m", "15m", "30m"]:
            period = "5d"
        elif interval == "1h":
            period = "1mo"
        else:  # 1d
            period = "6mo"

    try:
        ticker = yf.Ticker(symbol)

        if start is not None:
            df = ticker.history(interval=interval, start=start, end=end, auto_adjust=False)
        else:
            df = ticker.history(period=period, interval=interval, auto_adjust=False)

        if df.empty:
            logger.warning("데이터가 비어있습니다", symbol=symbol, interval=interval)
            return pd.DataFrame()

        # 컬럼 정리 (yfinance는 Open, High, Low, Close, Volume, Dividends, Stock Splits 반환)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # 인덱스 이름 설정
        df.index.name = "Datetime"

        logger.info(
            "인트라데이 데이터 가져오기 성공",
            symbol=symbol,
            interval=interval,
            rows=len(df),
            start=df.index.min(),
            end=df.index.max(),
        )

        return df

    except Exception as e:
        logger.error("인트라데이 데이터 가져오기 실패", symbol=symbol, interval=interval, error=str(e))
        return pd.DataFrame()


def get_current_price(symbol: str = "^GSPC") -> Optional[float]:
    """현재 가격 가져오기

    Args:
        symbol: 티커 심볼

    Returns:
        현재 가격 (실패 시 None)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # 가능한 가격 필드들 시도
        for key in ["regularMarketPrice", "currentPrice", "price"]:
            if key in info and info[key] is not None:
                return float(info[key])

        # 최신 데이터로부터 가격 가져오기
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])

        return None

    except Exception as e:
        logger.error("현재 가격 가져오기 실패", symbol=symbol, error=str(e))
        return None


def calculate_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산

    Args:
        df: OHLCV 데이터프레임

    Returns:
        기술적 지표가 추가된 데이터프레임
    """
    if df.empty:
        return df

    df = df.copy()

    # 이동평균선
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # 볼린저 밴드
    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = sma_20 + (std_20 * 2)
    df["BB_Lower"] = sma_20 - (std_20 * 2)

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    return df
