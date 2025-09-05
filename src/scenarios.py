"""시나리오 엔진 모듈

사용자가 정의한 쇼크를 특성에 적용하고,
예측 변화에 대한 기여도를 계산.

Example:
    >>> from scenarios import apply_shocks, compute_scenario_impact
    >>> shocked_features = apply_shocks(X_current, {"VIX": 5, "DGS10": -0.5})
    >>> impact = compute_scenario_impact(model, X_current, shocked_features)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from structlog import get_logger

logger = get_logger()


# 기본 시나리오 정의
DEFAULT_SCENARIOS = {
    "risk_on": {
        "name": "위험선호 (Risk-On)",
        "description": "시장 참가자들이 위험 자산을 선호하는 시나리오",
        "shocks": {
            "VIX": -5,
            "TERM_SPREAD": 0.3,
            "CREDIT_SPREAD": -0.5,
            "DGS10": 0.2
        }
    },
    "risk_off": {
        "name": "위험회피 (Risk-Off)",
        "description": "불확실성 증가로 안전자산 선호가 강해지는 시나리오",
        "shocks": {
            "VIX": 10,
            "TERM_SPREAD": -0.5,
            "CREDIT_SPREAD": 1.0,
            "DGS10": -0.3
        }
    },
    "fed_tightening": {
        "name": "연준 긴축",
        "description": "연준이 금리를 인상하고 긴축적 통화정책을 펴는 시나리오",
        "shocks": {
            "DGS3MO": 0.5,
            "DGS2": 0.4,
            "DGS10": 0.2,
            "NFCI": 0.3,
            "TERM_SPREAD": -0.3
        }
    },
    "inflation_spike": {
        "name": "인플레이션 급등",
        "description": "예상보다 높은 인플레이션으로 시장이 충격을 받는 시나리오",
        "shocks": {
            "CPIAUCSL": 0.5,  # CPI 서프라이즈
            "DGS10": 0.3,
            "VIX": 5,
            "DCOILWTICO": 20  # 유가 상승
        }
    },
    "growth_scare": {
        "name": "성장 우려",
        "description": "경제 성장 둔화 우려가 확산되는 시나리오",
        "shocks": {
            "UNRATE": 0.5,  # 실업률 상승
            "VIX": 8,
            "TERM_SPREAD": -0.4,
            "DGS10": -0.4,
            "CREDIT_SPREAD": 0.8
        }
    }
}


def apply_shocks(
    features: Union[pd.DataFrame, pd.Series],
    shocks: Dict[str, float],
    shock_type: str = "absolute"
) -> Union[pd.DataFrame, pd.Series]:
    """특성에 쇼크 적용
    
    Args:
        features: 원본 특성 데이터
        shocks: 적용할 쇼크 딕셔너리 (특성명: 쇼크값)
        shock_type: 쇼크 타입 ("absolute" 또는 "relative")
        
    Returns:
        쇼크가 적용된 특성
    """
    # 복사본 생성
    if isinstance(features, pd.Series):
        shocked = features.copy()
    else:
        shocked = features.copy()
    
    applied_shocks = {}
    
    for feature, shock_value in shocks.items():
        if feature not in features.index if isinstance(features, pd.Series) else feature not in features.columns:
            logger.warning(f"특성 '{feature}'을 찾을 수 없음")
            continue
        
        if shock_type == "absolute":
            # 절대적 변화
            if isinstance(features, pd.Series):
                original_value = features[feature]
                shocked[feature] = original_value + shock_value
            else:
                original_value = features[feature].iloc[0] if len(features) > 0 else 0
                shocked[feature] = features[feature] + shock_value
        else:
            # 상대적 변화 (%)
            if isinstance(features, pd.Series):
                original_value = features[feature]
                shocked[feature] = original_value * (1 + shock_value / 100)
            else:
                original_value = features[feature].iloc[0] if len(features) > 0 else 0
                shocked[feature] = features[feature] * (1 + shock_value / 100)
        
        applied_shocks[feature] = {
            "original": original_value,
            "shocked": shocked[feature].iloc[0] if isinstance(shocked, pd.DataFrame) else shocked[feature],
            "change": shock_value
        }
    
    logger.info(
        "쇼크 적용 완료",
        n_shocks=len(applied_shocks),
        shock_type=shock_type,
        features=list(applied_shocks.keys())
    )
    
    return shocked


def compute_scenario_impact(
    model: Any,  # QuantileRegressionModel
    original_features: Union[pd.DataFrame, pd.Series],
    shocked_features: Union[pd.DataFrame, pd.Series],
    horizon: str = "1M",
    quantiles: List[float] = [0.05, 0.50, 0.95]
) -> Dict[str, Any]:
    """시나리오의 영향 계산
    
    원본 예측과 쇼크 후 예측을 비교하여 영향도 산출
    
    Args:
        model: 학습된 모델
        original_features: 원본 특성
        shocked_features: 쇼크가 적용된 특성
        horizon: 예측 기간
        quantiles: 계산할 분위수
        
    Returns:
        영향도 분석 결과
    """
    # DataFrame 형태로 변환
    if isinstance(original_features, pd.Series):
        original_features = original_features.to_frame().T
    if isinstance(shocked_features, pd.Series):
        shocked_features = shocked_features.to_frame().T
    
    # 원본 예측
    original_pred = model.predict(original_features)
    
    # 쇼크 후 예측
    shocked_pred = model.predict(shocked_features)
    
    # 영향도 계산
    impact = {}
    
    for q_key in original_pred.keys():
        q_value = float(q_key[1:]) / 100
        if q_value in quantiles:
            original_val = original_pred[q_key][0]
            shocked_val = shocked_pred[q_key][0]
            
            impact[q_key] = {
                "original": original_val,
                "shocked": shocked_val,
                "change": shocked_val - original_val,
                "change_pct": ((shocked_val - original_val) / abs(original_val) * 100) if original_val != 0 else 0
            }
    
    # 중앙값 변화
    median_change = impact.get("q50", {}).get("change", 0)
    
    # IQR 변화
    if "q75" in impact and "q25" in impact:
        original_iqr = impact["q75"]["original"] - impact["q25"]["original"]
        shocked_iqr = impact["q75"]["shocked"] - impact["q25"]["shocked"]
        iqr_change = shocked_iqr - original_iqr
    else:
        iqr_change = 0
    
    result = {
        "horizon": horizon,
        "quantile_impacts": impact,
        "median_change": median_change,
        "iqr_change": iqr_change,
        "uncertainty_change": "증가" if iqr_change > 0 else "감소" if iqr_change < 0 else "불변"
    }
    
    logger.info(
        "시나리오 영향 계산",
        horizon=horizon,
        median_change=round(median_change, 4),
        iqr_change=round(iqr_change, 4)
    )
    
    return result


def compute_contributions(
    model: Any,  # QuantileRegressionModel  
    features_delta: pd.Series,
    quantile: float = 0.5
) -> pd.DataFrame:
    """특성 변화에 대한 기여도 계산
    
    선형 근사를 통한 간단한 기여도 계산 (계수 × Δ특성)
    
    Args:
        model: 학습된 모델
        features_delta: 특성 변화량
        quantile: 기여도를 계산할 분위
        
    Returns:
        기여도 DataFrame
    """
    # 계수 가져오기
    coef = model.get_coefficients(quantile)
    
    if coef is None:
        logger.warning(f"분위 {quantile}에 대한 계수를 찾을 수 없음")
        return pd.DataFrame()
    
    # 특성 변화량과 계수 정렬
    common_features = coef.index.intersection(features_delta.index)
    
    if len(common_features) == 0:
        logger.warning("공통 특성이 없음")
        return pd.DataFrame()
    
    # 기여도 계산
    contributions = coef[common_features] * features_delta[common_features]
    
    # DataFrame으로 정리
    contrib_df = pd.DataFrame({
        "feature": contributions.index,
        "coefficient": coef[common_features].values,
        "delta": features_delta[common_features].values,
        "contribution": contributions.values,
        "abs_contribution": contributions.abs().values
    })
    
    # 절대값 기준 정렬
    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)
    
    # 상위/하위 기여 특성
    total_contribution = contrib_df["contribution"].sum()
    contrib_df["contribution_pct"] = contrib_df["contribution"] / abs(total_contribution) * 100 if total_contribution != 0 else 0
    
    logger.info(
        "기여도 계산 완료",
        quantile=quantile,
        total_contribution=round(total_contribution, 4),
        top_contributors=list(contrib_df.head(3)["feature"])
    )
    
    return contrib_df


def create_scenario_comparison(
    model: Any,
    base_features: Union[pd.DataFrame, pd.Series],
    scenarios: Dict[str, Dict[str, Any]],
    horizons: List[str] = ["1M", "3M", "6M", "12M"]
) -> pd.DataFrame:
    """여러 시나리오의 영향 비교
    
    Args:
        model: 학습된 모델 딕셔너리
        base_features: 기준 특성
        scenarios: 시나리오 딕셔너리
        horizons: 비교할 기간 목록
        
    Returns:
        시나리오별 영향 비교 DataFrame
    """
    results = []
    
    for scenario_name, scenario_def in scenarios.items():
        shocks = scenario_def.get("shocks", {})
        
        # 쇼크 적용
        shocked_features = apply_shocks(base_features, shocks)
        
        # 각 기간별 영향 계산
        for horizon in horizons:
            if horizon not in model:
                continue
            
            impact = compute_scenario_impact(
                model[horizon],
                base_features,
                shocked_features,
                horizon
            )
            
            result_row = {
                "scenario": scenario_name,
                "scenario_name": scenario_def.get("name", scenario_name),
                "horizon": horizon,
                "median_change": impact["median_change"],
                "iqr_change": impact["iqr_change"],
                "q05_change": impact["quantile_impacts"].get("q05", {}).get("change", 0),
                "q95_change": impact["quantile_impacts"].get("q95", {}).get("change", 0)
            }
            
            results.append(result_row)
    
    comparison_df = pd.DataFrame(results)
    
    # 피벗 테이블 형태로 변환 (선택적)
    if len(comparison_df) > 0:
        comparison_pivot = comparison_df.pivot(
            index="scenario_name",
            columns="horizon",
            values="median_change"
        )
        
        logger.info(
            "시나리오 비교 완료",
            n_scenarios=len(scenarios),
            n_horizons=len(horizons),
            shape=comparison_pivot.shape
        )
        
        return comparison_pivot
    
    return comparison_df


def generate_shock_grid(
    feature: str,
    shock_range: Tuple[float, float],
    n_points: int = 10,
    shock_type: str = "absolute"
) -> List[Dict[str, float]]:
    """단일 특성에 대한 쇼크 그리드 생성
    
    민감도 분석을 위한 쇼크 범위 생성
    
    Args:
        feature: 특성명
        shock_range: 쇼크 범위 (min, max)
        n_points: 그리드 포인트 수
        shock_type: 쇼크 타입
        
    Returns:
        쇼크 딕셔너리 리스트
    """
    shock_values = np.linspace(shock_range[0], shock_range[1], n_points)
    
    shock_grid = []
    for value in shock_values:
        shock_grid.append({feature: value})
    
    logger.info(
        "쇼크 그리드 생성",
        feature=feature,
        range=shock_range,
        n_points=n_points
    )
    
    return shock_grid


def run_sensitivity_analysis(
    model: Any,
    base_features: Union[pd.DataFrame, pd.Series],
    sensitivity_features: List[str],
    shock_ranges: Dict[str, Tuple[float, float]],
    n_points: int = 10,
    horizon: str = "1M"
) -> pd.DataFrame:
    """민감도 분석 실행
    
    각 특성의 변화에 따른 예측 변화를 분석
    
    Args:
        model: 학습된 모델
        base_features: 기준 특성
        sensitivity_features: 분석할 특성 목록
        shock_ranges: 특성별 쇼크 범위
        n_points: 각 특성별 분석 포인트 수
        horizon: 예측 기간
        
    Returns:
        민감도 분석 결과 DataFrame
    """
    results = []
    
    for feature in sensitivity_features:
        if feature not in shock_ranges:
            logger.warning(f"특성 '{feature}'의 쇼크 범위가 정의되지 않음")
            continue
        
        # 쇼크 그리드 생성
        shock_grid = generate_shock_grid(
            feature,
            shock_ranges[feature],
            n_points
        )
        
        # 각 쇼크에 대한 영향 계산
        for shock in shock_grid:
            shocked_features = apply_shocks(base_features, shock)
            
            impact = compute_scenario_impact(
                model,
                base_features,
                shocked_features,
                horizon
            )
            
            result_row = {
                "feature": feature,
                "shock_value": shock[feature],
                "median_return": impact["quantile_impacts"].get("q50", {}).get("shocked", 0),
                "q05_return": impact["quantile_impacts"].get("q05", {}).get("shocked", 0),
                "q95_return": impact["quantile_impacts"].get("q95", {}).get("shocked", 0),
                "iqr": (impact["quantile_impacts"].get("q75", {}).get("shocked", 0) - 
                       impact["quantile_impacts"].get("q25", {}).get("shocked", 0))
            }
            
            results.append(result_row)
    
    sensitivity_df = pd.DataFrame(results)
    
    logger.info(
        "민감도 분석 완료",
        n_features=len(sensitivity_features),
        n_total_points=len(sensitivity_df),
        horizon=horizon
    )
    
    return sensitivity_df


def create_scenario_report(
    scenario_name: str,
    shocks: Dict[str, float],
    impacts: Dict[str, Dict[str, Any]],
    contributions: pd.DataFrame
) -> Dict[str, Any]:
    """시나리오 분석 리포트 생성
    
    Args:
        scenario_name: 시나리오 이름
        shocks: 적용된 쇼크
        impacts: 기간별 영향
        contributions: 기여도 분석
        
    Returns:
        리포트 딕셔너리
    """
    # 주요 기여 요인 (상위 3개)
    if len(contributions) > 0:
        top_positive = contributions[contributions["contribution"] > 0].head(3)
        top_negative = contributions[contributions["contribution"] < 0].head(3)
        
        drivers = {
            "positive": [
                {"feature": row["feature"], "contribution": row["contribution"]}
                for _, row in top_positive.iterrows()
            ],
            "negative": [
                {"feature": row["feature"], "contribution": row["contribution"]}
                for _, row in top_negative.iterrows()
            ]
        }
    else:
        drivers = {"positive": [], "negative": []}
    
    # 요약 통계
    median_changes = {
        horizon: impact["median_change"]
        for horizon, impact in impacts.items()
    }
    
    # 리포트 생성
    report = {
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "applied_shocks": shocks,
        "median_changes": median_changes,
        "key_drivers": drivers,
        "impacts_by_horizon": impacts,
        "summary": {
            "1M_impact": median_changes.get("1M", 0),
            "12M_impact": median_changes.get("12M", 0),
            "max_impact": max(median_changes.values()) if median_changes else 0,
            "min_impact": min(median_changes.values()) if median_changes else 0
        }
    }
    
    logger.info(
        "시나리오 리포트 생성",
        scenario=scenario_name,
        n_shocks=len(shocks),
        n_horizons=len(impacts)
    )
    
    return report
