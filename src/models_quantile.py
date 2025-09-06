"""분위 회귀 모델 모듈

각 기간(h)과 분위(q)별로 분위 회귀를 학습하고,
단기 예측의 경우 VIX 기반 앵커와 블렌드하여 안정성 확보.

Example:
    >>> from models_quantile import train_models, predict_fanchart
    >>> models = train_models(X_train, y_train, horizons=["1M"], quantiles=[0.05, 0.5, 0.95])
    >>> fanchart = predict_fanchart(models, X_test, vix_anchor=20)
"""

import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from structlog import get_logger

logger = get_logger()


# 기본 분위수
DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# VIX 기반 변동성 앵커링 파라미터
VIX_ANNUALIZED_FACTOR = np.sqrt(12)  # 월간 변환
VIX_SCALING = 0.01  # VIX를 소수로 변환 (20 -> 0.20)


class QuantileRegressionModel:
    """분위 회귀 모델 래퍼
    
    scikit-learn의 QuantileRegressor를 사용하여
    각 분위별 예측을 수행하고 결과를 관리
    """
    
    def __init__(
        self,
        quantiles: List[float] = DEFAULT_QUANTILES,
        alpha: float = 1.0,
        solver: str = "highs",
        fit_intercept: bool = True
    ):
        """
        Args:
            quantiles: 예측할 분위수 목록
            alpha: L1 규제 강도
            solver: 최적화 솔버
            fit_intercept: 절편 포함 여부
        """
        self.quantiles = quantiles
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        
        # 각 분위별 모델
        self.models: Dict[float, QuantileRegressor] = {}
        self.scalers: Dict[float, StandardScaler] = {}
        self.feature_names: Optional[List[str]] = None
        
        # 학습 상태
        self.is_fitted = False
        
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> "QuantileRegressionModel":
        """모델 학습
        
        Args:
            X: 특성 데이터
            y: 타깃 데이터  
            sample_weight: 샘플 가중치
            
        Returns:
            학습된 모델 인스턴스
        """
        # DataFrame인 경우 특성명 저장
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            X_array = X
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # 각 분위별로 모델 학습
        for q in self.quantiles:
            logger.info(f"분위 {q} 모델 학습 시작")
            
            # 스케일러
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
            self.scalers[q] = scaler
            
            # 모델 생성 및 학습
            model = QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver=self.solver,
                fit_intercept=self.fit_intercept
            )
            
            model.fit(X_scaled, y_array, sample_weight=sample_weight)
            self.models[q] = model
            
            # 학습 통계
            train_pred = model.predict(X_scaled)
            train_loss = self._pinball_loss(y_array, train_pred, q)
            
            logger.info(
                f"분위 {q} 학습 완료",
                train_loss=round(train_loss, 6),
                n_features=X_scaled.shape[1],
                n_samples=X_scaled.shape[0]
            )
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """모든 분위에 대한 예측
        
        Args:
            X: 특성 데이터
            
        Returns:
            분위별 예측값 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        predictions = {}
        
        for q in self.quantiles:
            # 스케일링
            X_scaled = self.scalers[q].transform(X_array)
            
            # 예측
            pred = self.models[q].predict(X_scaled)
            predictions[f"q{int(q*100):02d}"] = pred
        
        return predictions
    
    def get_coefficients(self, quantile: float) -> Optional[pd.Series]:
        """특정 분위의 계수 반환
        
        Args:
            quantile: 분위수
            
        Returns:
            계수 Series (특성명 포함)
        """
        if quantile not in self.models:
            return None
        
        model = self.models[quantile]
        coef = model.coef_
        
        if self.feature_names:
            return pd.Series(coef, index=self.feature_names)
        else:
            return pd.Series(coef)
    
    def _pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """핀볼 손실 계산"""
        errors = y_true - y_pred
        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def train_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    horizons: List[str],
    quantiles: List[float] = DEFAULT_QUANTILES,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, QuantileRegressionModel]:
    """각 기간별 분위 회귀 모델 학습
    
    Args:
        X: 특성 데이터
        y: 타깃 데이터 (각 horizon별 컬럼 포함)
        horizons: 학습할 기간 목록
        quantiles: 학습할 분위수 목록
        config: 모델 설정
        
    Returns:
        기간별 학습된 모델 딕셔너리
    """
    if config is None:
        config = {"alpha": 1.0, "solver": "highs"}
    
    models = {}
    
    for horizon in horizons:
        target_col = f"return_{horizon}"
        
        if target_col not in y.columns:
            logger.warning(f"타깃 컬럼 {target_col} 없음")
            continue
        
        logger.info(f"기간 {horizon} 모델 학습 시작")
        
        # 타깃 데이터 추출
        y_horizon = y[target_col]
        
        # 결측치 제거: 타깃 및 특성 모두 유효한 행만 학습에 사용
        mask = y_horizon.notna()
        X_clean = X[mask]
        y_clean = y_horizon[mask]
        # 특성 쪽 NaN 제거
        if isinstance(X_clean, pd.DataFrame):
            valid_rows = X_clean.notna().all(axis=1)
            if not valid_rows.all():
                dropped = int((~valid_rows).sum())
                logger.info("결측 행 제거", dropped_rows=dropped)
            X_clean = X_clean[valid_rows]
            y_clean = y_clean[valid_rows]
        
        if len(y_clean) < 50:
            logger.warning(f"학습 데이터 부족: {len(y_clean)} 샘플 (<50)")
            continue
        
        # 모델 생성 및 학습
        model = QuantileRegressionModel(
            quantiles=quantiles,
            alpha=config.get("alpha", 1.0),
            solver=config.get("solver", "highs")
        )
        
        model.fit(X_clean, y_clean)
        models[horizon] = model
        
        logger.info(
            f"기간 {horizon} 모델 학습 완료",
            n_samples=len(y_clean),
            n_quantiles=len(quantiles)
        )
    
    return models


def predict_fanchart(
    models: Dict[str, QuantileRegressionModel],
    X_t: Union[pd.DataFrame, pd.Series],
    vix_current: Optional[float] = None,
    blend_config: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """팬차트 예측 생성
    
    Args:
        models: 학습된 모델 딕셔너리
        X_t: 현재 시점 특성
        vix_current: 현재 VIX 값 (단기 폭 앵커링용)
        blend_config: 블렌딩 설정
        
    Returns:
        팬차트 DTO
    """
    if isinstance(X_t, pd.Series):
        X_t = X_t.to_frame().T
    
    if blend_config is None:
        blend_config = {
            "1M": {"vix_weight": 0.5},
            "3M": {"vix_weight": 0.3},
            "6M": {"vix_weight": 0.1},
            "12M": {"vix_weight": 0.0}
        }
    
    fancharts = {}
    
    for horizon, model in models.items():
        # 입력 특성 정렬/보정: 학습시 특성 목록에 맞춰 컬럼 정렬, 결측은 0으로 대체
        X_in = X_t
        if isinstance(X_t, pd.DataFrame) and getattr(model, "feature_names", None):
            X_in = X_t.reindex(columns=model.feature_names).fillna(0.0)

        # 기본 예측
        predictions = model.predict(X_in)
        
        # VIX 앵커링 (단기에만 적용)
        if vix_current and horizon in blend_config:
            vix_weight = blend_config[horizon].get("vix_weight", 0)
            
            if vix_weight > 0:
                predictions = apply_vix_anchor(
                    predictions, 
                    vix_current, 
                    horizon,
                    vix_weight
                )
        
        # 팬차트 DTO 생성
        fanchart = {
            "horizon": horizon,
            "asof": datetime.now().strftime("%Y-%m-%d"),
            "quantiles": predictions,
            "vix_anchor": vix_current if vix_current else None,
            "blend_weight": blend_config.get(horizon, {}).get("vix_weight", 0)
        }
        
        fancharts[horizon] = fanchart
        
        logger.info(
            f"팬차트 생성",
            horizon=horizon,
            median=round(predictions.get("q50", [0])[0], 4),
            iqr=round(predictions.get("q75", [0])[0] - predictions.get("q25", [0])[0], 4)
        )
    
    return fancharts


def apply_vix_anchor(
    predictions: Dict[str, np.ndarray],
    vix: float,
    horizon: str,
    weight: float = 0.5
) -> Dict[str, np.ndarray]:
    """VIX 기반 변동성 앵커링
    
    단기 예측의 폭을 VIX 기반 변동성과 블렌드
    
    Args:
        predictions: 원본 예측
        vix: 현재 VIX 값
        horizon: 예측 기간
        weight: VIX 앵커 가중치
        
    Returns:
        조정된 예측
    """
    # VIX를 월간 변동성으로 변환
    if horizon == "1M":
        vix_monthly_vol = vix * VIX_SCALING / VIX_ANNUALIZED_FACTOR
    elif horizon == "3M":
        vix_monthly_vol = vix * VIX_SCALING / VIX_ANNUALIZED_FACTOR * np.sqrt(3)
    else:
        # 6M 이상은 앵커링 안함
        return predictions
    
    # 중앙값
    median = predictions.get("q50", np.array([0]))[0]
    
    # VIX 기반 분위별 값
    vix_based = {}
    for key in predictions.keys():
        q = float(key[1:]) / 100  # q05 -> 0.05
        
        # 정규분포 가정하에 분위수 계산
        z_score = stats.norm.ppf(q)
        vix_based[key] = np.array([median + z_score * vix_monthly_vol])
    
    # 블렌딩
    blended = {}
    for key in predictions.keys():
        original = predictions[key]
        anchor = vix_based[key]
        blended[key] = weight * anchor + (1 - weight) * original
    
    logger.debug(
        "VIX 앵커링 적용",
        horizon=horizon,
        vix=vix,
        weight=weight,
        original_iqr=predictions.get("q75", [0])[0] - predictions.get("q25", [0])[0],
        blended_iqr=blended.get("q75", [0])[0] - blended.get("q25", [0])[0]
    )
    
    return blended


def compute_prediction_intervals(
    predictions: Dict[str, np.ndarray],
    confidence_levels: List[float] = [0.50, 0.80, 0.90]
) -> Dict[str, Tuple[float, float]]:
    """예측 구간 계산
    
    Args:
        predictions: 분위별 예측
        confidence_levels: 신뢰수준 목록
        
    Returns:
        신뢰구간별 (하한, 상한) 튜플
    """
    intervals = {}
    
    for level in confidence_levels:
        alpha = 1 - level
        lower_q = f"q{int(alpha/2 * 100):02d}"
        upper_q = f"q{int((1-alpha/2) * 100):02d}"
        
        if lower_q in predictions and upper_q in predictions:
            lower = predictions[lower_q][0]
            upper = predictions[upper_q][0]
            intervals[f"{int(level*100)}%"] = (lower, upper)
    
    return intervals


def calculate_feature_contributions(
    model: QuantileRegressionModel,
    X_t: Union[pd.DataFrame, pd.Series],
    X_baseline: Optional[pd.DataFrame] = None,
    quantile: float = 0.5
) -> pd.Series:
    """피처 기여도 계산 (간단한 선형 근사)
    
    계수 × Δ특징량 방식으로 기여도 계산
    
    Args:
        model: 학습된 모델
        X_t: 현재 시점 특성
        X_baseline: 기준 특성 (None이면 학습 데이터 평균)
        quantile: 계산할 분위
        
    Returns:
        피처별 기여도
    """
    if isinstance(X_t, pd.Series):
        X_t = X_t.to_frame().T
    
    # 계수 가져오기
    coef = model.get_coefficients(quantile)
    
    if coef is None:
        return pd.Series()
    
    # 기준점 설정
    if X_baseline is None:
        # 학습 데이터의 평균을 기준으로 사용
        X_baseline = pd.DataFrame(
            model.scalers[quantile].mean_,
            index=X_t.columns,
            columns=["baseline"]
        ).T
    
    # 특성 변화량
    delta_X = X_t.values[0] - X_baseline.values[0]
    
    # 스케일링된 변화량
    scaler = model.scalers[quantile]
    delta_X_scaled = delta_X / scaler.scale_
    
    # 기여도 = 계수 × 변화량
    contributions = coef * delta_X_scaled
    
    # Series로 변환
    if model.feature_names:
        contributions_series = pd.Series(contributions, index=model.feature_names)
    else:
        contributions_series = pd.Series(contributions)
    
    # 상위/하위 기여 특성
    top_contributors = contributions_series.abs().nlargest(5)
    
    logger.info(
        "기여도 계산",
        quantile=quantile,
        top_features=list(top_contributors.index),
        top_values=list(top_contributors.values)
    )
    
    return contributions_series


# 모델 저장/로드
def save_models(
    models: Dict[str, QuantileRegressionModel],
    path: Path
) -> None:
    """모델 저장
    
    Args:
        models: 저장할 모델 딕셔너리
        path: 저장 경로
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    
    logger.info(
        "모델 저장 완료",
        path=str(path),
        n_horizons=len(models)
    )


def load_models(path: Path) -> Dict[str, QuantileRegressionModel]:
    """모델 로드
    
    Args:
        path: 모델 파일 경로
        
    Returns:
        로드된 모델 딕셔너리
    """
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {path}")
    
    with open(path, 'rb') as f:
        models = pickle.load(f)
    
    logger.info(
        "모델 로드 완료",
        path=str(path),
        n_horizons=len(models)
    )
    
    return models


# 진단 도구
def diagnose_quantile_crossing(
    predictions: Dict[str, np.ndarray]
) -> bool:
    """분위 교차 진단
    
    낮은 분위의 예측값이 높은 분위보다 큰 경우 감지
    
    Args:
        predictions: 분위별 예측
        
    Returns:
        교차 발생 여부
    """
    quantiles = sorted(predictions.keys())
    has_crossing = False
    
    for i in range(len(quantiles) - 1):
        q_low = quantiles[i]
        q_high = quantiles[i + 1]
        
        if predictions[q_low][0] > predictions[q_high][0]:
            logger.warning(
                "분위 교차 감지",
                lower_quantile=q_low,
                lower_value=predictions[q_low][0],
                higher_quantile=q_high, 
                higher_value=predictions[q_high][0]
            )
            has_crossing = True
    
    return has_crossing


def create_model_summary(
    models: Dict[str, QuantileRegressionModel]
) -> pd.DataFrame:
    """모델 요약 정보 생성
    
    Args:
        models: 모델 딕셔너리
        
    Returns:
        요약 정보 DataFrame
    """
    summary_data = []
    
    for horizon, model in models.items():
        for q in model.quantiles:
            coef = model.get_coefficients(q)
            
            if coef is not None:
                summary = {
                    "horizon": horizon,
                    "quantile": q,
                    "n_features": len(coef),
                    "n_nonzero": (coef != 0).sum(),
                    "max_coef": coef.abs().max(),
                    "mean_coef": coef.abs().mean()
                }
                summary_data.append(summary)
    
    return pd.DataFrame(summary_data)
