"""타깃 변수 생성 모듈

S&P 500 지수의 1/3/6/12개월 로그수익률을 계산하여
분위 회귀 모델의 타깃 변수로 사용.

Example:
    >>> from targets import compute_forward_returns
    >>> targets_df = compute_forward_returns(price_df, horizons=["1M", "3M", "6M", "12M"])
"""

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from structlog import get_logger

logger = get_logger()


def compute_forward_returns(
    price_df: pd.DataFrame,
    price_col: str = "SP500",
    horizons: List[str] = ["1M", "3M", "6M", "12M"],
    method: str = "log"
) -> pd.DataFrame:
    """미래 수익률 계산
    
    r_{t→t+h} = log(S_{t+h} / S_t) for log returns
    r_{t→t+h} = (S_{t+h} / S_t) - 1 for simple returns
    
    Args:
        price_df: 가격 데이터가 포함된 DataFrame
        price_col: 가격 컬럼명 (기본값: "SP500")
        horizons: 수익률 계산 기간 목록
        method: 수익률 계산 방법 ("log" 또는 "simple")
        
    Returns:
        각 기간별 수익률이 포함된 DataFrame
    """
    if price_col not in price_df.columns:
        raise ValueError(f"가격 컬럼 '{price_col}'이 DataFrame에 없습니다.")
    
    result = pd.DataFrame(index=price_df.index)
    
    # 가격 데이터 추출
    prices = price_df[price_col].copy()
    
    # 결측치 확인
    if prices.isna().any():
        logger.warning(
            "가격 데이터에 결측치 존재",
            missing_count=prices.isna().sum(),
            total_count=len(prices)
        )
    
    for horizon in horizons:
        logger.info("수익률 계산 시작", horizon=horizon, method=method)
        
        # 기간 파싱
        periods = parse_horizon(horizon, price_df.index)
        
        # 미래 가격
        future_prices = prices.shift(-periods)
        
        # 수익률 계산
        if method == "log":
            # 로그 수익률: log(S_{t+h} / S_t)
            returns = np.log(future_prices / prices)
        else:
            # 단순 수익률: (S_{t+h} / S_t) - 1
            returns = (future_prices / prices) - 1
        
        # 컬럼명 설정
        col_name = f"return_{horizon}"
        result[col_name] = returns
        
        # 통계 로깅
        valid_returns = returns.dropna()
        if len(valid_returns) > 0:
            logger.info(
                "수익률 계산 완료",
                horizon=horizon,
                valid_count=len(valid_returns),
                mean=round(valid_returns.mean(), 4),
                std=round(valid_returns.std(), 4),
                min=round(valid_returns.min(), 4),
                max=round(valid_returns.max(), 4)
            )
    
    return result


def parse_horizon(horizon: str, index: pd.DatetimeIndex) -> int:
    """수익률 기간 문자열을 인덱스 기준 정수로 변환
    
    Args:
        horizon: 기간 문자열 (예: "1M", "3M", "6M", "12M")
        index: 데이터의 DatetimeIndex
        
    Returns:
        shift할 기간 수
    """
    # 빈도 추론
    freq = pd.infer_freq(index)
    
    # 월간 데이터 가정 (기본값)
    if freq in ["M", "MS", "ME"]:
        periods_per_month = 1
    elif freq in ["W", "W-SUN", "W-MON"]:
        periods_per_month = 4  # 대략 주 4개
    elif freq in ["D", "B"]:
        periods_per_month = 21  # 대략 영업일 21일
    else:
        # 빈도를 추론할 수 없는 경우 월간 가정
        logger.warning("빈도 추론 실패, 월간 데이터로 가정", inferred_freq=freq)
        periods_per_month = 1
    
    # 기간 파싱
    if horizon.endswith("M"):
        months = int(horizon[:-1])
        return months * periods_per_month
    elif horizon.endswith("Y"):
        years = int(horizon[:-1])
        return years * 12 * periods_per_month
    else:
        raise ValueError(f"지원하지 않는 기간 형식: {horizon}")


def align_targets_with_features(
    targets_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """타깃과 피처를 정렬하여 학습 가능한 형태로 병합
    
    Args:
        targets_df: 타깃 변수 DataFrame
        features_df: 피처 DataFrame
        target_cols: 사용할 타깃 컬럼 목록 (None이면 모든 타깃)
        
    Returns:
        정렬된 DataFrame (피처 + 타깃)
    """
    # 타깃 컬럼 선택
    if target_cols is None:
        target_cols = [col for col in targets_df.columns if col.startswith("return_")]
    
    # 인덱스 정렬
    common_index = features_df.index.intersection(targets_df.index)
    
    if len(common_index) == 0:
        raise ValueError("피처와 타깃의 공통 인덱스가 없습니다.")
    
    logger.info(
        "피처-타깃 정렬",
        feature_shape=features_df.shape,
        target_shape=targets_df.shape,
        common_dates=len(common_index)
    )
    
    # 정렬된 데이터 추출
    aligned_features = features_df.loc[common_index]
    aligned_targets = targets_df.loc[common_index, target_cols]
    
    # 병합
    result = pd.concat([aligned_features, aligned_targets], axis=1)
    
    # 타깃이 있는 행만 선택 (미래 수익률이므로 마지막 행들은 NA)
    mask = aligned_targets.notna().any(axis=1)
    result_valid = result[mask]
    
    logger.info(
        "학습 가능 데이터 생성",
        total_rows=len(result),
        valid_rows=len(result_valid),
        dropped_rows=len(result) - len(result_valid)
    )
    
    return result_valid


def create_target_weights(
    targets_df: pd.DataFrame,
    method: str = "uniform",
    decay_factor: float = 0.95
) -> pd.DataFrame:
    """타깃 샘플 가중치 생성
    
    최근 데이터에 더 높은 가중치를 부여하거나
    변동성이 큰 시기에 가중치를 조정
    
    Args:
        targets_df: 타깃 변수 DataFrame
        method: 가중치 방법 ("uniform", "exponential", "volatility")
        decay_factor: 지수 감쇠율 (exponential 방법용)
        
    Returns:
        각 샘플의 가중치가 포함된 DataFrame
    """
    weights = pd.DataFrame(index=targets_df.index)
    
    n_samples = len(targets_df)
    
    if method == "uniform":
        # 동일 가중치
        weights["weight"] = 1.0 / n_samples
        
    elif method == "exponential":
        # 지수 감쇠 가중치 (최근일수록 높음)
        decay_weights = decay_factor ** np.arange(n_samples - 1, -1, -1)
        decay_weights = decay_weights / decay_weights.sum()
        weights["weight"] = decay_weights
        
    elif method == "volatility":
        # 변동성 기반 가중치
        # 롤링 변동성이 높은 시기에 더 높은 가중치
        for col in targets_df.columns:
            if col.startswith("return_"):
                vol = targets_df[col].rolling(window=12, min_periods=1).std()
                vol_weight = vol / vol.sum()
                weights[f"weight_{col}"] = vol_weight
        
        # 평균 가중치
        if weights.shape[1] > 0:
            weights["weight"] = weights.mean(axis=1)
        else:
            weights["weight"] = 1.0 / n_samples
    
    else:
        raise ValueError(f"지원하지 않는 가중치 방법: {method}")
    
    # 정규화
    weights["weight"] = weights["weight"] / weights["weight"].sum()
    
    logger.info(
        "샘플 가중치 생성",
        method=method,
        min_weight=weights["weight"].min(),
        max_weight=weights["weight"].max(),
        weight_sum=weights["weight"].sum()
    )
    
    return weights


def split_train_test(
    data: pd.DataFrame,
    test_size: Union[float, int] = 0.2,
    gap: int = 0
) -> Dict[str, pd.DataFrame]:
    """시계열 데이터를 훈련/테스트셋으로 분할
    
    시계열 특성상 무작위 분할이 아닌 시간 순서 기준 분할
    
    Args:
        data: 전체 데이터
        test_size: 테스트셋 크기 (비율 또는 절대 개수)
        gap: 훈련셋과 테스트셋 사이 갭 (데이터 누수 방지)
        
    Returns:
        {"train": 훈련셋, "test": 테스트셋} 딕셔너리
    """
    n_samples = len(data)
    
    # 테스트셋 크기 계산
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size
    
    # 훈련셋 크기
    n_train = n_samples - n_test - gap
    
    if n_train <= 0:
        raise ValueError(f"훈련 데이터가 충분하지 않습니다. (n_train={n_train})")
    
    # 분할
    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train + gap:]
    
    logger.info(
        "데이터 분할",
        total_samples=n_samples,
        train_samples=len(train_data),
        test_samples=len(test_data),
        gap=gap,
        train_period=(str(train_data.index[0].date()), str(train_data.index[-1].date())),
        test_period=(str(test_data.index[0].date()), str(test_data.index[-1].date()))
    )
    
    return {"train": train_data, "test": test_data}


def calculate_target_statistics(
    targets_df: pd.DataFrame,
    groupby: Optional[str] = None
) -> pd.DataFrame:
    """타깃 변수의 통계량 계산
    
    Args:
        targets_df: 타깃 변수 DataFrame
        groupby: 그룹화 기준 (예: "year", "month")
        
    Returns:
        통계량 DataFrame
    """
    stats_list = []
    
    for col in targets_df.columns:
        if not col.startswith("return_"):
            continue
        
        data = targets_df[col].dropna()
        
        if groupby == "year":
            grouped = data.groupby(data.index.year)
        elif groupby == "month":
            grouped = data.groupby(data.index.month)
        elif groupby:
            raise ValueError(f"지원하지 않는 그룹화 기준: {groupby}")
        else:
            # 전체 통계
            stats = {
                "horizon": col,
                "count": len(data),
                "mean": data.mean(),
                "std": data.std(),
                "skew": data.skew(),
                "kurt": data.kurt(),
                "min": data.min(),
                "q25": data.quantile(0.25),
                "median": data.median(),
                "q75": data.quantile(0.75),
                "max": data.max(),
                "sharpe": data.mean() / data.std() if data.std() > 0 else 0
            }
            stats_list.append(stats)
            continue
        
        # 그룹별 통계
        for name, group in grouped:
            if len(group) < 2:
                continue
            
            stats = {
                "horizon": col,
                groupby: name,
                "count": len(group),
                "mean": group.mean(),
                "std": group.std(),
                "skew": group.skew(),
                "kurt": group.kurt(),
                "min": group.min(),
                "q25": group.quantile(0.25),
                "median": group.median(),
                "q75": group.quantile(0.75),
                "max": group.max(),
                "sharpe": group.mean() / group.std() if group.std() > 0 else 0
            }
            stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    logger.info(
        "타깃 통계량 계산",
        n_horizons=len(targets_df.columns),
        groupby=groupby,
        n_groups=len(stats_df) if groupby else 1
    )
    
    return stats_df


# 타깃 저장/로드 헬퍼
def save_targets(
    targets_df: pd.DataFrame,
    path: Path,
    format: str = "parquet"
) -> None:
    """타깃 데이터 저장
    
    Args:
        targets_df: 저장할 타깃 DataFrame
        path: 저장 경로
        format: 파일 형식 ("parquet" 또는 "csv")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        targets_df.to_parquet(path, compression="snappy")
    elif format == "csv":
        targets_df.to_csv(path, index=True)
    else:
        raise ValueError(f"지원하지 않는 형식: {format}")
    
    logger.info(
        "타깃 저장 완료",
        path=str(path),
        shape=targets_df.shape,
        format=format
    )


def load_targets(
    path: Path,
    format: str = "parquet"
) -> pd.DataFrame:
    """타깃 데이터 로드
    
    Args:
        path: 파일 경로
        format: 파일 형식 ("parquet" 또는 "csv")
        
    Returns:
        로드된 타깃 DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"타깃 파일을 찾을 수 없음: {path}")
    
    if format == "parquet":
        targets_df = pd.read_parquet(path)
    elif format == "csv":
        targets_df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"지원하지 않는 형식: {format}")
    
    logger.info(
        "타깃 로드 완료",
        path=str(path),
        shape=targets_df.shape
    )
    
    return targets_df
