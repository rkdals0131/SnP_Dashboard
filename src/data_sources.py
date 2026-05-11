"""데이터 소스 어댑터 모듈

FRED, BLS, BEA 등 공공 API를 통한 데이터 수집을 담당하는 어댑터 구현.
재시도/백오프 로직과 Parquet 저장 기능을 포함.

Example:
    >>> from data_sources import FREDSource
    >>> fred = FREDSource(api_key="your_key")
    >>> df = fred.fetch(["DGS10", "VIXCLS"], start=date(2020,1,1), end=date(2023,12,31))
"""

import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Any, List, Tuple

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from structlog import get_logger

# 구조화 로깅
logger = get_logger()


def _cache_covers_range(
    df: pd.DataFrame,
    start: date,
    end: date,
    max_stale_days: int = 3,
) -> bool:
    """Return True when cached data broadly covers the requested date range."""

    if df.empty:
        return False
    idx = pd.to_datetime(df.index)
    min_date = idx.min().date()
    max_date = idx.max().date()
    stale_cutoff = end - pd.Timedelta(days=max_stale_days).to_pytimedelta()
    return min_date <= start and max_date >= stale_cutoff


class DataSource(Protocol):
    """데이터 소스 인터페이스 프로토콜"""
    name: str
    
    def fetch(self, series: Iterable[str], start: date, end: date, **kw: Any) -> pd.DataFrame:
        """시계열 데이터 가져오기
        
        Args:
            series: 시리즈 ID 목록
            start: 시작일
            end: 종료일
            **kw: 추가 파라미터
            
        Returns:
            DataFrame with columns: series_id, value, realtime_start, realtime_end
            Index: date
        """
        ...


class FREDSource:
    """FRED(Federal Reserve Economic Data) API 어댑터
    
    FRED API를 통해 경제 시계열 데이터를 수집하고 Parquet 형식으로 저장.
    재시도 로직과 백오프를 포함하여 안정적인 데이터 수집 보장.
    """
    
    name = "fred"
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0
    ):
        """
        Args:
            api_key: FRED API 키 (환경변수 FRED_API_KEY 사용 가능)
            cache_dir: 캐시 디렉토리 경로
            max_retries: 최대 재시도 횟수
            backoff_factor: 재시도 간격 증가율
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY env var or pass api_key")
        
        self.cache_dir = cache_dir or Path("data/raw/fred")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        logger.info("FRED 소스 초기화", cache_dir=str(self.cache_dir))
    
    def fetch(
        self, 
        series: Iterable[str], 
        start: date, 
        end: date, 
        use_cache: bool = True,
        **kw: Any
    ) -> pd.DataFrame:
        """FRED에서 시계열 데이터 가져오기
        
        Args:
            series: FRED 시리즈 ID 목록 (예: ["DGS10", "VIXCLS"])
            start: 시작일
            end: 종료일
            use_cache: 캐시 사용 여부
            
        Returns:
            통합된 DataFrame (인덱스: date, 컬럼: series_id, value, realtime_start, realtime_end)
        """
        all_data = []
        
        for series_id in series:
            logger.info("시리즈 수집 시작", series_id=series_id, start=str(start), end=str(end))
            
            # 캐시 확인
            if use_cache:
                cached_df = self._load_cache(series_id, start, end)
                if cached_df is not None:
                    all_data.append(cached_df)
                    continue
            
            # API 호출
            df = self._fetch_series(series_id, start, end)
            
            # 캐시 저장
            if df is not None and not df.empty:
                self._save_cache(series_id, df)
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        # 모든 시리즈 통합
        result = pd.concat(all_data, ignore_index=False)
        logger.info("데이터 수집 완료", total_rows=len(result))
        
        return result
    
    def _fetch_series(self, series_id: str, start: date, end: date) -> Optional[pd.DataFrame]:
        """단일 시리즈 API 호출"""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
            "file_type": "json",
            "sort_order": "asc"
        }
        
        url = f"{self.BASE_URL}/series/observations"
        
        # 재시도 로직
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # 관측치가 없는 경우
                if "observations" not in data or not data["observations"]:
                    logger.warning("관측치 없음", series_id=series_id)
                    return None
                
                # DataFrame 변환
                df = pd.DataFrame(data["observations"])
                
                # 날짜 파싱 및 정리
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                
                # 값 변환 (결측치 처리)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                
                # 필요 컬럼 선택
                df["series_id"] = series_id
                df = df[["series_id", "value", "realtime_start", "realtime_end"]]
                
                logger.info("시리즈 수집 성공", series_id=series_id, rows=len(df))
                return df
                
            except requests.exceptions.RequestException as e:
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(
                    "API 호출 실패, 재시도 대기", 
                    series_id=series_id,
                    attempt=attempt + 1,
                    wait_time=wait_time,
                    error=str(e)
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error("최대 재시도 횟수 초과", series_id=series_id)
                    return None
    
    def _load_cache(self, series_id: str, start: date, end: date) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 로드"""
        cache_file = self.cache_dir / f"{series_id}.parquet"
        
        if not cache_file.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            
            # 요청 기간 필터링
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            df_filtered = df[mask]
            
            if len(df_filtered) > 0 and _cache_covers_range(df_filtered, start, end):
                logger.info("캐시에서 로드", series_id=series_id, rows=len(df_filtered))
                return df_filtered
                
        except Exception as e:
            logger.warning("캐시 로드 실패", series_id=series_id, error=str(e))
        
        return None
    
    def _save_cache(self, series_id: str, df: pd.DataFrame) -> None:
        """캐시에 데이터 저장"""
        cache_file = self.cache_dir / f"{series_id}.parquet"
        
        try:
            # 기존 캐시와 병합
            if cache_file.exists():
                existing_df = pd.read_parquet(cache_file)
                # 중복 제거하여 병합
                df = pd.concat([existing_df, df]).drop_duplicates()
                df = df.sort_index()
            
            # Parquet 저장
            df.to_parquet(cache_file, compression="snappy")
            logger.info("캐시 저장 완료", series_id=series_id, file=str(cache_file))
            
        except Exception as e:
            logger.error("캐시 저장 실패", series_id=series_id, error=str(e))


class FearGreedSource:
    """공포-탐욕 지수 API 어댑터

    기본 제공자는 Alternative.me(/fng/)이며, 일별 지수(0~100)를 수집합니다.
    """

    name = "fear_greed"
    ALTERNATIVE_URL = "https://api.alternative.me/fng/"

    def __init__(
        self,
        provider: str = "alternative",
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        self.provider = provider.lower()
        self.cache_dir = cache_dir or Path("data/raw/fear_greed")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        logger.info("FearGreed 소스 초기화", provider=self.provider, cache_dir=str(self.cache_dir))

    def fetch(
        self,
        series: Iterable[str],
        start: date,
        end: date,
        use_cache: bool = True,
        **kw: Any,
    ) -> pd.DataFrame:
        """공포-탐욕 지수 시계열 수집

        Args:
            series: 시리즈 ID 목록(예: ["CRYPTO_FNG"])
            start: 시작일
            end: 종료일
            use_cache: 캐시 사용 여부

        Returns:
            통합된 DataFrame (index=date, columns=[series_id,value,value_classification,realtime_start,realtime_end,source])
        """
        all_data: List[pd.DataFrame] = []

        for series_id in series:
            if use_cache:
                cached_df = self._load_cache(series_id, start, end)
                if cached_df is not None:
                    all_data.append(cached_df)
                    continue

            if self.provider != "alternative":
                raise ValueError(f"지원하지 않는 provider: {self.provider}")

            raw_df = self._fetch_alternative(series_id)
            if raw_df is None or raw_df.empty:
                continue

            # 전체 이력을 캐시에 저장하고 요청 구간만 반환
            self._save_cache(series_id, raw_df)
            mask = (raw_df.index >= pd.Timestamp(start)) & (raw_df.index <= pd.Timestamp(end))
            all_data.append(raw_df.loc[mask])

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=False).sort_index()
        logger.info("FearGreed 데이터 수집 완료", total_rows=len(result))
        return result

    def _fetch_alternative(self, series_id: str) -> Optional[pd.DataFrame]:
        params = {
            "limit": 0,      # 전체 이력
            "format": "json"
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.ALTERNATIVE_URL, params=params, timeout=20)
                response.raise_for_status()
                payload = response.json()
                df = self._parse_alternative_payload(payload, series_id)
                logger.info("Alternative.me 수집 성공", series_id=series_id, rows=len(df))
                return df
            except requests.exceptions.RequestException as e:
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(
                    "Alternative.me 호출 실패, 재시도 대기",
                    attempt=attempt + 1,
                    wait_time=wait_time,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error("Alternative.me 최대 재시도 초과", series_id=series_id)
                    return None
            except Exception as e:
                logger.error("Alternative.me 파싱 실패", series_id=series_id, error=str(e))
                return None

    @staticmethod
    def _parse_alternative_payload(payload: Dict[str, Any], series_id: str) -> pd.DataFrame:
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if metadata.get("error"):
            raise ValueError(f"Alternative.me API error: {metadata.get('error')}")

        rows: List[Dict[str, Any]] = []
        for rec in payload.get("data", []):
            ts_raw = rec.get("timestamp")
            if ts_raw is None:
                continue
            ts = pd.to_datetime(int(ts_raw), unit="s")
            day = pd.Timestamp(ts.date())

            rows.append(
                {
                    "date": day,
                    "series_id": series_id,
                    "value": float(rec.get("value")) if rec.get("value") is not None else np.nan,
                    "value_classification": rec.get("value_classification"),
                    "realtime_start": day.strftime("%Y-%m-%d"),
                    "realtime_end": day.strftime("%Y-%m-%d"),
                    "source": "alternative.me",
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "series_id",
                    "value",
                    "value_classification",
                    "realtime_start",
                    "realtime_end",
                    "source",
                ]
            )

        df = pd.DataFrame.from_records(rows).set_index("date").sort_index()
        # 동일 날짜 중복 시 최신 레코드 유지
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _load_cache(self, series_id: str, start: date, end: date) -> Optional[pd.DataFrame]:
        cache_file = self.cache_dir / f"{series_id}.parquet"
        if not cache_file.exists():
            return None

        try:
            df = pd.read_parquet(cache_file)
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            df_filtered = df[mask]
            if len(df_filtered) > 0 and _cache_covers_range(
                df_filtered,
                start,
                end,
                max_stale_days=1,
            ):
                logger.info("FearGreed 캐시에서 로드", series_id=series_id, rows=len(df_filtered))
                return df_filtered
        except Exception as e:
            logger.warning("FearGreed 캐시 로드 실패", series_id=series_id, error=str(e))

        return None

    def _save_cache(self, series_id: str, df: pd.DataFrame) -> None:
        cache_file = self.cache_dir / f"{series_id}.parquet"
        try:
            if cache_file.exists():
                existing_df = pd.read_parquet(cache_file)
                df = pd.concat([existing_df, df]).drop_duplicates()
                df = df.sort_index()
            df.to_parquet(cache_file, compression="snappy")
            logger.info("FearGreed 캐시 저장 완료", series_id=series_id, file=str(cache_file))
        except Exception as e:
            logger.error("FearGreed 캐시 저장 실패", series_id=series_id, error=str(e))


def persist_parquet(df: pd.DataFrame, path: Path) -> None:
    """DataFrame을 Parquet 형식으로 저장
    
    Args:
        df: 저장할 DataFrame
        path: 저장 경로
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy")
    logger.info("Parquet 저장", path=str(path), rows=len(df))


def load_parquet(path: Path) -> pd.DataFrame:
    """Parquet 파일에서 DataFrame 로드
    
    Args:
        path: 파일 경로
        
    Returns:
        로드된 DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없음: {path}")
    
    df = pd.read_parquet(path)
    logger.info("Parquet 로드", path=str(path), rows=len(df))
    return df


# 추가 데이터 소스를 위한 스텁
class BLSSource:
    """BLS(Bureau of Labor Statistics) API 어댑터 스텁"""
    name = "bls"
    
    def fetch(self, series: Iterable[str], start: date, end: date, **kw: Any) -> pd.DataFrame:
        """향후 구현 예정"""
        raise NotImplementedError("BLS 소스는 향후 구현 예정")


class BEASource:
    """BEA(Bureau of Economic Analysis) API 어댑터 스텁"""
    name = "bea"
    
    def fetch(self, series: Iterable[str], start: date, end: date, **kw: Any) -> pd.DataFrame:
        """향후 구현 예정"""
        raise NotImplementedError("BEA 소스는 향후 구현 예정")


class FXSource:
    """무료 환율 API 어댑터 (exchangerate.host)

    - 키 불필요, 무료로 일별 시계열(timeseries) 제공
    - 지원 형식: 'USDKRW', 'USDJPY', 'USDEUR' 등 기본통화USD 페어
    - 스키마는 FRED과 동일하게 통일(index=date, columns=[series_id,value,realtime_start,realtime_end])
    """

    name = "fx"
    BASE_URL = "https://api.exchangerate.host/timeseries"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        self.cache_dir = cache_dir or Path("data/raw/fx")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        logger.info("FX 소스 초기화", cache_dir=str(self.cache_dir))

    def fetch(
        self,
        series: Iterable[str],
        start: date,
        end: date,
        use_cache: bool = True,
        base: str = "USD",
        **kw: Any,
    ) -> pd.DataFrame:
        all_data: List[pd.DataFrame] = []

        for sid in series:
            base_ccy, quote_ccy = self._parse_pair(sid, default_base=base)
            cache_df = self._load_cache(sid, start, end) if use_cache else None
            if cache_df is not None:
                all_data.append(cache_df)
                continue

            df = self._fetch_pair_series(sid, base_ccy, quote_ccy, start, end)
            if df is not None and not df.empty:
                self._save_cache(sid, df)
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=False)
        logger.info("FX 데이터 수집 완료", total_rows=len(result))
        return result

    def _parse_pair(self, pair: str, default_base: str = "USD") -> Tuple[str, str]:
        pair = pair.upper().replace("/", "")
        if len(pair) == 6:
            return pair[:3], pair[3:]
        if pair.startswith(default_base) and len(pair) > 3:
            return default_base, pair[3:]
        # 기본: USD{XXX}
        return default_base, pair[-3:]

    def _fetch_pair_series(
        self,
        series_id: str,
        base_ccy: str,
        quote_ccy: str,
        start: date,
        end: date,
    ) -> Optional[pd.DataFrame]:
        params = {
            "start_date": str(start),
            "end_date": str(end),
            "base": base_ccy,
            "symbols": quote_ccy,
        }

        for attempt in range(self.max_retries):
            try:
                r = requests.get(self.BASE_URL, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                if not data.get("rates"):
                    logger.warning("FX 관측치 없음", series_id=series_id)
                    return None

                records = []
                for d, rates in data["rates"].items():
                    val = rates.get(quote_ccy)
                    if val is None:
                        continue
                    records.append(
                        {
                            "date": pd.to_datetime(d),
                            "series_id": series_id.upper(),
                            "value": float(val),
                            "realtime_start": d,
                            "realtime_end": d,
                        }
                    )

                df = pd.DataFrame.from_records(records).set_index("date").sort_index()
                logger.info("FX 시리즈 수집 성공", series_id=series_id, rows=len(df))
                return df
            except requests.exceptions.RequestException as e:
                wait_time = self.backoff_factor * (2 ** attempt)
                logger.warning(
                    "FX API 호출 실패, 재시도 대기",
                    series_id=series_id,
                    attempt=attempt + 1,
                    wait_time=wait_time,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error("FX 최대 재시도 초과", series_id=series_id)
                    return None

    def _load_cache(self, series_id: str, start: date, end: date) -> Optional[pd.DataFrame]:
        cache_file = self.cache_dir / f"{series_id}.parquet"
        if not cache_file.exists():
            return None
        try:
            df = pd.read_parquet(cache_file)
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            df_filtered = df[mask]
            if len(df_filtered) > 0:
                logger.info("FX 캐시에서 로드", series_id=series_id, rows=len(df_filtered))
                return df_filtered
        except Exception as e:
            logger.warning("FX 캐시 로드 실패", series_id=series_id, error=str(e))
        return None

    def _save_cache(self, series_id: str, df: pd.DataFrame) -> None:
        cache_file = self.cache_dir / f"{series_id}.parquet"
        try:
            if cache_file.exists():
                existing_df = pd.read_parquet(cache_file)
                df = pd.concat([existing_df, df]).drop_duplicates()
                df = df.sort_index()
            df.to_parquet(cache_file, compression="snappy")
            logger.info("FX 캐시 저장 완료", series_id=series_id, file=str(cache_file))
        except Exception as e:
            logger.error("FX 캐시 저장 실패", series_id=series_id, error=str(e))
