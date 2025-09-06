"""데이터 정합 및 빈티지 처리 모듈

원천 시계열 데이터를 공통 캘린더(월말)로 정렬하고, 
발표 지연 규칙을 적용하여 선견편향을 방지하는 asof 조인 수행.

Example:
    >>> from align_vintage import build_master_panel, apply_publication_delays
    >>> panel = build_master_panel(raw_paths, calendar)
    >>> vintage_panel = apply_publication_delays(panel, PUBLICATION_RULES)
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd
from structlog import get_logger

logger = get_logger()


# 발표 지연 규칙 정의 (시리즈별)
PUBLICATION_RULES = {
    # 일간 지표: T+1 영업일
    "DGS10": "1BD",  # 10년 국채 수익률
    "DGS3MO": "1BD",  # 3개월 국채 수익률
    "DGS2": "1BD",  # 2년 국채 수익률
    "VIXCLS": "1BD",  # VIX 종가
    "SP500": "1BD",  # S&P 500 지수
    "DCOILWTICO": "1BD",  # WTI 유가
    "GOLDAMGBD228NLBM": "1BD",  # 금 가격 (LBMA AM, USD)
    # 환율(일별)
    "USDKRW": "1BD",
    "USDJPY": "1BD",
    "USDEUR": "1BD",
    
    # 주간 지표: 공표 주의 금요일 T+1 영업일
    "ICSA": "6D",  # 신규 실업수당 청구
    
    # 월간 지표: 다음 달 3영업일
    "CPIAUCSL": "1M+3BD",  # CPI
    "CPILFESL": "1M+3BD",  # Core CPI
    "PCEPI": "1M+3BD",  # PCE
    "UNRATE": "1M+3BD",  # 실업률
    "BAA": "1M+3BD",  # Moody's BAA 스프레드
    "BAMLH0A0HYM2": "1M+3BD",  # High Yield OAS
    "FEDFUNDS": "1M+3BD",  # 연방기금 금리(월)
    "PAYEMS": "1M+3BD",  # 비농업 신규고용
    "CIVPART": "1M+3BD",  # 경제활동참가율
    "JTSJOL": "1M+3BD",  # 구인건수(JOLTS)
    "DTWEXBGS": "1M+3BD",  # Broad 달러지수
    
    # 분기 지표: 다음 분기 20영업일
    "GDP": "1Q+20BD",  # GDP
    "GDPC1": "1Q+20BD",  # Real GDP
    
    # 연준/기관 지표
    "NFCI": "1W",  # Chicago Fed NFCI (주간)
    "ANFCI": "1W",  # Adjusted NFCI
}


def create_month_end_calendar(
    start_date: Union[str, date],
    end_date: Union[str, date],
    freq: str = "ME",
) -> pd.DatetimeIndex:
    """월말 기준 캘린더 생성
    
    Args:
        start_date: 시작일
        end_date: 종료일
        freq: 빈도 (기본값: "M" = 월말)
        
    Returns:
        월말 날짜 인덱스
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # 월말 날짜 생성 (ME = month end)
    calendar = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    logger.info(
        "월말 캘린더 생성", 
        start=str(calendar[0].date()) if len(calendar) > 0 else None,
        end=str(calendar[-1].date()) if len(calendar) > 0 else None,
        periods=len(calendar)
    )
    
    return calendar


def parse_delay_rule(rule: str) -> timedelta:
    """발표 지연 규칙 문자열을 timedelta로 변환
    
    Args:
        rule: 지연 규칙 문자열 (예: "1BD", "1M+3BD", "1Q+20BD")
        
    Returns:
        timedelta 객체
    """
    if rule == "1BD":
        return BDay(1)
    elif rule == "6D":
        return timedelta(days=6)
    elif rule == "1W":
        return timedelta(weeks=1)
    elif rule.startswith("1M"):
        # 1개월 + 추가 영업일
        base_delay = MonthEnd(1)
        if "+" in rule:
            extra_days = int(rule.split("+")[1].replace("BD", ""))
            return base_delay + BDay(extra_days)
        return base_delay
    elif rule.startswith("1Q"):
        # 1분기 + 추가 영업일
        base_delay = MonthEnd(3)  # 3개월 = 1분기
        if "+" in rule:
            extra_days = int(rule.split("+")[1].replace("BD", ""))
            return base_delay + BDay(extra_days)
        return base_delay
    else:
        raise ValueError(f"알 수 없는 지연 규칙: {rule}")


def build_master_panel(
    raw_paths: Dict[str, Path],
    calendar: pd.DatetimeIndex,
    forward_fill: bool = False
) -> pd.DataFrame:
    """원천 시계열을 공통 캘린더로 정렬하여 마스터 패널 구축
    
    Args:
        raw_paths: 시리즈별 원천 데이터 경로 매핑
        calendar: 공통 캘린더 (월말 기준)
        forward_fill: Forward-fill 허용 여부 (기본값: False)
        
    Returns:
        정렬된 마스터 패널 DataFrame (인덱스: date, 컬럼: series_id별)
    """
    all_series = {}
    
    for series_id, path in raw_paths.items():
        logger.info("시리즈 로드", series_id=series_id, path=str(path))
        
        try:
            # Parquet 파일 로드
            df = pd.read_parquet(path)
            
            # series_id로 필터링 (여러 시리즈가 함께 저장된 경우)
            if "series_id" in df.columns:
                df = df[df["series_id"] == series_id]
            
            # value 컬럼만 추출
            if "value" in df.columns:
                series = df["value"]
            else:
                series = df.iloc[:, 0]  # 첫 번째 컬럼 사용
            
            # 시리즈 이름 설정
            series.name = series_id
            
            # 날짜 인덱스 확인
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            
            all_series[series_id] = series
            
        except Exception as e:
            logger.error("시리즈 로드 실패", series_id=series_id, error=str(e))
            continue
    
    if not all_series:
        raise ValueError("로드된 시리즈가 없음")
    
    # 모든 시리즈를 DataFrame으로 결합
    panel = pd.DataFrame(all_series)
    
    # 캘린더에 맞춰 리샘플링 (asof join)
    aligned_data = {}
    
    for col in panel.columns:
        # 각 캘린더 날짜에 대해 가장 최근 값 찾기 (asof)
        aligned_series = []
        
        for cal_date in calendar:
            # cal_date 이전의 가장 최근 값
            mask = panel.index <= cal_date
            if mask.any():
                last_idx = panel.index[mask].max()
                value = panel.loc[last_idx, col]
            else:
                value = np.nan
            
            aligned_series.append(value)
        
        aligned_data[col] = aligned_series
    
    # 정렬된 패널 생성
    master_panel = pd.DataFrame(aligned_data, index=calendar)
    
    # Forward-fill 옵션 (기본적으로 사용하지 않음)
    if forward_fill:
        master_panel = master_panel.fillna(method="ffill")
    
    # 가용성 플래그 추가
    for col in master_panel.columns:
        avail_col = f"{col}_is_avail"
        master_panel[avail_col] = ~master_panel[col].isna()
    
    logger.info(
        "마스터 패널 구축 완료",
        shape=master_panel.shape,
        columns=list(master_panel.columns),
        date_range=(str(master_panel.index[0].date()), str(master_panel.index[-1].date()))
    )
    
    return master_panel


def apply_publication_delays(
    panel: pd.DataFrame,
    rules: Dict[str, str],
    as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    """발표 지연 규칙을 적용하여 빈티지 데이터 시뮬레이션
    
    Args:
        panel: 마스터 패널 DataFrame
        rules: 시리즈별 발표 지연 규칙
        as_of_date: 기준 날짜 (None이면 전체 기간에 적용)
        
    Returns:
        발표 지연이 적용된 DataFrame
    """
    vintage_panel = panel.copy()

    # 월말 캘린더 전제에서의 Lean 지연 시뮬레이션: 월간/분기 지표는 기간 수 만큼 시프트
    for series_id in panel.columns:
        if series_id.endswith("_is_avail"):
            continue

        rule = rules.get(series_id)
        if not rule:
            logger.warning("발표 지연 규칙 없음", series_id=series_id)
            # 가용성은 원래 값 기준
            avail_col = f"{series_id}_is_avail"
            if avail_col in vintage_panel.columns:
                vintage_panel[avail_col] = vintage_panel[series_id].notna()
            continue

        # 시프트 기간 결정 (월말 캘린더 기준 근사)
        shift_periods = 0
        if rule.startswith("1M"):
            shift_periods = 1
        elif rule.startswith("1Q"):
            shift_periods = 3
        else:
            shift_periods = 0  # 일간/주간은 월말 시점에서는 가용하다고 간주

        if shift_periods > 0:
            vintage_panel[series_id] = panel[series_id].shift(shift_periods)

        # as_of_date 이후 차단
        if as_of_date is not None:
            vintage_panel.loc[vintage_panel.index > pd.Timestamp(as_of_date), series_id] = np.nan

        # 가용성 플래그 갱신
        avail_col = f"{series_id}_is_avail"
        if avail_col in vintage_panel.columns:
            vintage_panel[avail_col] = vintage_panel[series_id].notna()
        else:
            vintage_panel[avail_col] = vintage_panel[series_id].notna()
    
    # 통계 로깅
    total_values = len(vintage_panel) * len([c for c in vintage_panel.columns if not c.endswith("_is_avail")])
    available_values = int(sum(vintage_panel[c].notna().sum() for c in vintage_panel.columns if not c.endswith("_is_avail")))
    
    logger.info(
        "빈티지 데이터 생성 완료",
        total_cells=total_values,
        available_cells=available_values,
        availability_ratio=round(available_values / total_values, 3) if total_values > 0 else 0
    )
    
    return vintage_panel


def get_available_data(
    vintage_panel: pd.DataFrame,
    as_of_date: Union[str, date],
    series_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """특정 시점에서 사용 가능한 데이터만 추출
    
    Args:
        vintage_panel: 빈티지 처리된 패널
        as_of_date: 기준 날짜
        series_list: 추출할 시리즈 목록 (None이면 전체)
        
    Returns:
        기준 날짜 시점에서 사용 가능한 데이터
    """
    if isinstance(as_of_date, str):
        as_of_date = pd.to_datetime(as_of_date)
    
    # 기준 날짜까지의 데이터만 선택
    mask = vintage_panel.index <= as_of_date
    available_data = vintage_panel.loc[mask].copy()
    
    # 특정 시리즈만 선택
    if series_list:
        cols_to_keep = []
        for series in series_list:
            if series in available_data.columns:
                cols_to_keep.append(series)
            if f"{series}_is_avail" in available_data.columns:
                cols_to_keep.append(f"{series}_is_avail")
        
        available_data = available_data[cols_to_keep]
    
    # 가용성 플래그에 따라 NA 처리
    for col in available_data.columns:
        if col.endswith("_is_avail"):
            continue
        
        avail_col = f"{col}_is_avail"
        if avail_col in available_data.columns:
            # 사용 불가능한 값은 NA로 처리
            mask = ~available_data[avail_col]
            available_data.loc[mask, col] = np.nan
    
    return available_data


# 헬퍼 함수들
def merge_frequency_data(
    daily_data: pd.DataFrame,
    weekly_data: pd.DataFrame,
    monthly_data: pd.DataFrame,
    calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """서로 다른 빈도의 데이터를 병합
    
    Args:
        daily_data: 일간 데이터
        weekly_data: 주간 데이터
        monthly_data: 월간 데이터
        calendar: 목표 캘린더
        
    Returns:
        병합된 DataFrame
    """
    # 각 데이터를 캘린더에 맞춰 정렬
    aligned_daily = align_to_calendar(daily_data, calendar)
    aligned_weekly = align_to_calendar(weekly_data, calendar)
    aligned_monthly = align_to_calendar(monthly_data, calendar)
    
    # 모두 병합
    merged = pd.concat([aligned_daily, aligned_weekly, aligned_monthly], axis=1)
    
    return merged


def align_to_calendar(
    data: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    method: str = "asof"
) -> pd.DataFrame:
    """데이터를 목표 캘린더에 정렬
    
    Args:
        data: 원본 데이터
        calendar: 목표 캘린더
        method: 정렬 방법 ("asof" 또는 "nearest")
        
    Returns:
        정렬된 DataFrame
    """
    if method == "asof":
        # asof 조인 사용
        return data.reindex(calendar, method="ffill", limit=0)
    elif method == "nearest":
        # 가장 가까운 날짜 매칭
        return data.reindex(calendar, method="nearest")
    else:
        raise ValueError(f"알 수 없는 정렬 방법: {method}")
