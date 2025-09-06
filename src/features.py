"""피처 엔지니어링 모듈

레벨, 1차차분(Δ), 2차차분(Δ²), 10년 분위 스케일링,
MacroScore/MacroBreadth 계산, 레짐 배지 생성 등을 담당.

Example:
    >>> from features import make_deltas, percentile_score_10y, compute_macro_score
    >>> df_with_deltas = make_deltas(df, ["VIX", "DGS10"])
    >>> df_with_scores = percentile_score_10y(df_with_deltas, ["VIX", "DGS10"])
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from structlog import get_logger

logger = get_logger()


# 카테고리별 지표 매핑
CATEGORY_MAPPING = {
    "inflation": ["CPIAUCSL", "CPILFESL", "PCEPI"],
    "growth": ["GDP", "GDPC1", "ADS"],
    "labor": ["UNRATE", "ICSA", "PAYEMS"],
    "rates_curve": ["DGS10", "DGS3MO", "DGS2", "TERM_SPREAD"],
    "financial_conditions": ["NFCI", "ANFCI", "VIX"],
    "credit": ["BAA", "BAMLH0A0HYM2", "CREDIT_SPREAD"],
    "volatility": ["VIXCLS", "SKEW", "MOVE"],
    "commodities": ["DCOILWTICO", "GOLDAMGBD228NLBM", "DJP"],
    "fx": ["USDKRW", "USDJPY", "USDEUR", "DTWEXBGS"]
}

# 기본 가중치
DEFAULT_WEIGHTS = {
    "level": 0.5,
    "momentum": 0.35,
    "acceleration": 0.15
}


def make_deltas(
    df: pd.DataFrame, 
    cols: List[str],
    periods: int = 1
) -> pd.DataFrame:
    """1차 차분 (Δ) 생성
    
    Args:
        df: 원본 DataFrame
        cols: 차분을 계산할 컬럼 목록
        periods: 차분 기간 (기본값: 1)
        
    Returns:
        차분 컬럼이 추가된 DataFrame
    """
    result = df.copy()
    
    for col in cols:
        if col not in df.columns:
            logger.warning("컬럼 없음", column=col)
            continue
        
        delta_col = f"{col}_d1"
        result[delta_col] = df[col].diff(periods)
        
        logger.info("1차 차분 생성", column=col, new_column=delta_col)
    
    return result


def make_delta_squared(
    df: pd.DataFrame,
    cols: List[str],
    periods: int = 1
) -> pd.DataFrame:
    """2차 차분 (Δ²) 생성
    
    Δ²x_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})
    
    Args:
        df: 원본 DataFrame (1차 차분 포함)
        cols: 2차 차분을 계산할 컬럼 목록
        periods: 차분 기간 (기본값: 1)
        
    Returns:
        2차 차분 컬럼이 추가된 DataFrame
    """
    result = df.copy()
    
    for col in cols:
        if col not in df.columns:
            logger.warning("컬럼 없음", column=col)
            continue
        
        # 1차 차분이 있는지 확인
        delta_col = f"{col}_d1"
        if delta_col not in result.columns:
            # 1차 차분 먼저 생성
            result[delta_col] = df[col].diff(periods)
        
        # 2차 차분 = 1차 차분의 차분
        delta2_col = f"{col}_d2"
        result[delta2_col] = result[delta_col].diff(periods)
        
        logger.info("2차 차분 생성", column=col, new_column=delta2_col)
    
    return result


def percentile_score_10y(
    df: pd.DataFrame,
    cols: List[str],
    window_years: int = 10,
    min_periods: Optional[int] = None,  # 빈도에 따라 기본값 결정
) -> pd.DataFrame:
    """10년 롤링 윈도우 백분위 스코어 계산
    
    각 시점에서 과거 10년 데이터 대비 백분위를 계산하고
    [-1, +1] 범위로 스케일링 (0% -> -1, 50% -> 0, 100% -> +1)
    
    Args:
        df: 원본 DataFrame
        cols: 백분위를 계산할 컬럼 목록
        window_years: 롤링 윈도우 연수 (기본값: 10)
        min_periods: 최소 필요 관측치 수
        
    Returns:
        백분위 스코어가 추가된 DataFrame
    """
    result = df.copy()
    
    # 연간 기간 수 추정 (월별=12)
    if len(df) > 0:
        freq = pd.infer_freq(df.index)
        if freq in {"M", "MS", "ME"}:
            periods_per_year = 12
        elif freq and freq.startswith("W"):
            periods_per_year = 52
        elif freq and (freq.startswith("Q") or freq.startswith("BQ")):
            periods_per_year = 4
        elif freq in {"A", "Y", "AS", "YS"}:
            periods_per_year = 1
        else:
            # 일간/영업일 등은 252로 근사
            periods_per_year = 252
    else:
        periods_per_year = 12  # 기본값
    
    window_size = window_years * periods_per_year
    
    for col in cols:
        if col not in df.columns:
            logger.warning("컬럼 없음", column=col)
            continue
        
        score_col = f"{col}_pctscore"
        
        eff_min_periods = min_periods if min_periods is not None else max(3, periods_per_year)
        # 롤링 백분위 계산
        percentiles = df[col].rolling(
            window=window_size,
            min_periods=eff_min_periods
        ).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 
            if len(x.dropna()) >= eff_min_periods else np.nan
        )
        
        # [-1, +1] 스케일링
        result[score_col] = 2 * percentiles - 1
        
        logger.info(
            "백분위 스코어 생성", 
            column=col, 
            new_column=score_col,
            window_size=window_size
        )
    
    return result


def compute_z_scores(
    df: pd.DataFrame,
    cols: List[str],
    window: Optional[int] = None
) -> pd.DataFrame:
    """표준화 (z-score) 계산
    
    Args:
        df: 원본 DataFrame
        cols: 표준화할 컬럼 목록
        window: 롤링 윈도우 크기 (None이면 전체 기간)
        
    Returns:
        z-score가 추가된 DataFrame
    """
    result = df.copy()
    
    for col in cols:
        if col not in df.columns:
            continue
        
        z_col = f"{col}_z"
        
        if window:
            # 롤링 z-score
            rolling = df[col].rolling(window=window, min_periods=1)
            mean = rolling.mean()
            std = rolling.std()
            result[z_col] = (df[col] - mean) / std
        else:
            # 전체 기간 z-score
            mean = df[col].mean()
            std = df[col].std()
            result[z_col] = (df[col] - mean) / std
    
    return result


def compute_macro_breadth(
    df: pd.DataFrame,
    score_cols: List[str],
    thresholds: Dict[str, float] = {"top50": 0.0, "top20": 0.6}
) -> pd.DataFrame:
    """MacroBreadth 계산
    
    상위 분위에 위치한 지표들의 비중을 계산
    
    Args:
        df: 백분위 스코어가 포함된 DataFrame
        score_cols: 백분위 스코어 컬럼 목록
        thresholds: 임계값 매핑 (기본값: 상위 50% = 0.0, 상위 20% = 0.6)
        
    Returns:
        브레드스가 추가된 DataFrame
    """
    result = df.copy()
    
    for threshold_name, threshold_value in thresholds.items():
        breadth_col = f"MacroBreadth_{threshold_name}"
        
        # 각 시점에서 임계값을 넘는 지표 비중
        breadth = df[score_cols].apply(
            lambda row: (row > threshold_value).sum() / len(row.dropna()) 
            if len(row.dropna()) > 0 else np.nan,
            axis=1
        )
        
        result[breadth_col] = breadth
        
        # 변화량 계산
        delta_col = f"MacroBreadth_{threshold_name}_delta"
        result[delta_col] = breadth.diff()
        
        logger.info(
            "브레드스 계산",
            threshold=threshold_name,
            threshold_value=threshold_value,
            columns=[breadth_col, delta_col]
        )
    
    return result


def compute_macro_score(
    df: pd.DataFrame,
    category_scores: Dict[str, List[str]],
    weights: Dict[str, float] = None,
    category_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """MacroScore 계산
    
    카테고리별 합성 점수를 가중 평균하여 전체 MacroScore 산출
    
    Args:
        df: 피처가 포함된 DataFrame
        category_scores: 카테고리별 점수 컬럼 매핑
        weights: 레벨/모멘텀/가속도 가중치
        category_weights: 카테고리별 가중치 (None이면 동일가중)
        
    Returns:
        MacroScore가 추가된 DataFrame
    """
    result = df.copy()
    
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # 각 카테고리별 점수 계산
    category_composite_scores = {}
    
    for category, indicators in category_scores.items():
        # 레벨, 모멘텀, 가속도 점수 수집
        level_scores = []
        momentum_scores = []
        acceleration_scores = []
        
        for indicator in indicators:
            # 레벨 (백분위 스코어)
            level_col = f"{indicator}_pctscore"
            if level_col in df.columns:
                level_scores.append(df[level_col])
            
            # 모멘텀 (1차 차분의 z-score)
            momentum_col = f"{indicator}_d1_z"
            if momentum_col in df.columns:
                momentum_scores.append(df[momentum_col])
            
            # 가속도 (2차 차분의 z-score)
            accel_col = f"{indicator}_d2_z"
            if accel_col in df.columns:
                acceleration_scores.append(df[accel_col])
        
        # 카테고리 내 평균
        if level_scores:
            cat_level = pd.concat(level_scores, axis=1).mean(axis=1)
        else:
            cat_level = pd.Series(0, index=df.index)
        
        if momentum_scores:
            cat_momentum = pd.concat(momentum_scores, axis=1).mean(axis=1)
        else:
            cat_momentum = pd.Series(0, index=df.index)
        
        if acceleration_scores:
            cat_acceleration = pd.concat(acceleration_scores, axis=1).mean(axis=1)
        else:
            cat_acceleration = pd.Series(0, index=df.index)
        
        # 가중 합성
        category_score = (
            weights["level"] * cat_level +
            weights["momentum"] * cat_momentum +
            weights["acceleration"] * cat_acceleration
        )
        
        category_composite_scores[category] = category_score
        result[f"MacroScore_{category}"] = category_score
    
    # 전체 MacroScore (카테고리 가중 평균)
    if category_weights is None:
        # 동일 가중
        macro_score = pd.concat(
            list(category_composite_scores.values()), 
            axis=1
        ).mean(axis=1)
    else:
        # 가중 평균
        weighted_scores = []
        for cat, score in category_composite_scores.items():
            if cat in category_weights:
                weighted_scores.append(score * category_weights[cat])
        
        macro_score = pd.concat(weighted_scores, axis=1).sum(axis=1)
    
    result["MacroScore"] = macro_score
    
    logger.info(
        "MacroScore 계산 완료",
        categories=list(category_scores.keys()),
        weights=weights
    )
    
    return result


def assign_regimes(
    df: pd.DataFrame,
    rules: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """레짐 배지 할당
    
    금융여건, 경기모멘텀, 물가모멘텀 등의 레짐을 분류
    
    Args:
        df: 피처가 포함된 DataFrame
        rules: 레짐 분류 규칙 (None이면 기본 규칙 사용)
        
    Returns:
        레짐 배지가 추가된 DataFrame
    """
    result = df.copy()
    
    if rules is None:
        rules = {
            "financial_conditions": {
                "tight": lambda x: x.get("NFCI_pctscore", 0) > 0.6,
                "easy": lambda x: x.get("NFCI_pctscore", 0) < 0.4,
                "neutral": lambda x: 0.4 <= x.get("NFCI_pctscore", 0) <= 0.6
            },
            "growth_momentum": {
                "expanding": lambda x: (x.get("TERM_SPREAD_d1", 0) > 0) & 
                                     (x.get("ADS_d1", 0) > 0),
                "slowing": lambda x: (x.get("TERM_SPREAD_d1", 0) < 0) | 
                                   (x.get("ADS_d1", 0) < 0),
                "mixed": lambda x: True  # 기본값
            },
            "inflation_momentum": {
                "rising": lambda x: x.get("CPIAUCSL_d2", 0) > 0,
                "falling": lambda x: x.get("CPIAUCSL_d2", 0) < 0,
                "stable": lambda x: x.get("CPIAUCSL_d2", 0) == 0
            }
        }
    
    # 각 레짐 타입별로 배지 할당
    for regime_type, regime_rules in rules.items():
        regime_col = f"Regime_{regime_type}"
        
        # 각 행에 대해 규칙 적용
        regime_values = []
        
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            
            # 규칙을 순서대로 확인
            assigned = False
            for regime_name, rule_func in regime_rules.items():
                try:
                    if rule_func(row_dict):
                        regime_values.append(regime_name)
                        assigned = True
                        break
                except Exception as e:
                    logger.warning(
                        "레짐 규칙 적용 실패",
                        regime_type=regime_type,
                        regime_name=regime_name,
                        error=str(e)
                    )
            
            if not assigned:
                regime_values.append("unknown")
        
        result[regime_col] = regime_values
        
        logger.info(
            "레짐 배지 할당",
            regime_type=regime_type,
            unique_values=result[regime_col].unique()
        )
    
    return result


def create_regime_dummies(
    df: pd.DataFrame,
    regime_cols: List[str]
) -> pd.DataFrame:
    """레짐 배지를 원-핫 인코딩
    
    Args:
        df: 레짐 배지가 포함된 DataFrame
        regime_cols: 레짐 컬럼 목록
        
    Returns:
        원-핫 인코딩된 레짐이 추가된 DataFrame
    """
    result = df.copy()
    
    for col in regime_cols:
        if col not in df.columns:
            continue
        
        # pd.get_dummies 사용 (숫자형 더미, 원본 문자열 컬럼은 제거)
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        
        # 기존 DataFrame에 추가
        result = pd.concat([result, dummies], axis=1)
        # 원본 문자열 컬럼 제거하여 모델 입력에 문자열이 섞이지 않도록 함
        if col in result.columns:
            result = result.drop(columns=[col])
        
        logger.info(
            "레짐 더미 생성",
            regime_col=col,
            dummy_cols=list(dummies.columns)
        )
    
    return result


def build_feature_panel(
    master_panel: pd.DataFrame,
    feature_config: Optional[Dict] = None
) -> pd.DataFrame:
    """전체 피처 패널 구축
    
    마스터 패널에서 모든 피처를 계산하여 통합
    
    Args:
        master_panel: 정합된 원본 데이터
        feature_config: 피처 설정 (None이면 기본값 사용)
        
    Returns:
        모든 피처가 포함된 DataFrame
    """
    logger.info("피처 패널 구축 시작")
    
    # 기본 설정
    if feature_config is None:
        feature_config = {
            "indicators": list(master_panel.columns),
            "use_deltas": True,
            "use_percentiles": True,
            "use_regimes": True,
            "compute_scores": True
        }
    
    # 사용할 지표 선택
    indicators = [col for col in feature_config["indicators"] 
                 if col in master_panel.columns and not col.endswith("_is_avail")]
    
    result = master_panel.copy()
    
    # 1. 차분 생성
    if feature_config.get("use_deltas", True):
        logger.info("차분 피처 생성")
        result = make_deltas(result, indicators)
        result = make_delta_squared(result, indicators)
        
        # 차분의 z-score
        delta_cols = [f"{ind}_d1" for ind in indicators if f"{ind}_d1" in result.columns]
        delta2_cols = [f"{ind}_d2" for ind in indicators if f"{ind}_d2" in result.columns]
        
        result = compute_z_scores(result, delta_cols)
        result = compute_z_scores(result, delta2_cols)
    
    # 2. 백분위 스코어
    if feature_config.get("use_percentiles", True):
        logger.info("백분위 스코어 생성")
        result = percentile_score_10y(result, indicators)
    
    # 3. 특수 피처 생성
    # Term spread (10Y - 3M)
    if "DGS10" in result.columns and "DGS3MO" in result.columns:
        result["TERM_SPREAD"] = result["DGS10"] - result["DGS3MO"]
        result = make_deltas(result, ["TERM_SPREAD"])
        result = percentile_score_10y(result, ["TERM_SPREAD"])
    
    # Credit spread
    if "BAA" in result.columns and "DGS10" in result.columns:
        result["CREDIT_SPREAD"] = result["BAA"] - result["DGS10"]
        result = make_deltas(result, ["CREDIT_SPREAD"])
        result = percentile_score_10y(result, ["CREDIT_SPREAD"])
    
    # 4. MacroScore와 Breadth
    if feature_config.get("compute_scores", True):
        logger.info("MacroScore/Breadth 계산")
        
        # 백분위 스코어 컬럼 수집
        score_cols = [col for col in result.columns if col.endswith("_pctscore")]
        
        if score_cols:
            # Breadth 계산
            result = compute_macro_breadth(result, score_cols)
            
            # MacroScore 계산 (카테고리 매핑 사용)
            available_categories = {}
            for cat, indicators in CATEGORY_MAPPING.items():
                available_indicators = [ind for ind in indicators 
                                      if ind in result.columns or f"{ind}_pctscore" in result.columns]
                if available_indicators:
                    available_categories[cat] = available_indicators
            
            if available_categories:
                result = compute_macro_score(result, available_categories)
    
    # 5. 레짐 배지
    if feature_config.get("use_regimes", True):
        logger.info("레짐 배지 생성")
        result = assign_regimes(result)
        
        # 원-핫 인코딩
        regime_cols = [col for col in result.columns if col.startswith("Regime_")]
        if regime_cols:
            result = create_regime_dummies(result, regime_cols)
    
    logger.info(
        "피처 패널 구축 완료",
        shape=result.shape,
        features=len([col for col in result.columns if col not in master_panel.columns]),
        total_columns=len(result.columns)
    )
    
    return result
