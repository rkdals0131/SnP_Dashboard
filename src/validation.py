"""검증 파이프라인 모듈

롤링 OOS(Out-of-Sample) 평가, 핀볼 손실/CRPS/커버리지 계산,
요약 리포트 생성을 담당합니다. v0.4.1 Lean: 재학습 포함/미포함 모두 지원.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from structlog import get_logger

from .models_quantile import (
    DEFAULT_QUANTILES,
    QuantileRegressionModel,
    train_models,
)

logger = get_logger()


@dataclass
class ValidationConfig:
    horizons: List[str]
    quantiles: List[float] = None
    alpha: float = 1.0
    solver: str = "highs"
    retrain_each_step: bool = True

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = DEFAULT_QUANTILES


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def _crps_from_quantiles(y_true: np.ndarray, qvals: Dict[str, np.ndarray]) -> float:
    """간단한 CRPS 근사: 분위 격자에서 핀볼손실을 적분 근사.

    참고: 정밀 CRPS는 분포형 예측이 필요하나, 여기서는 균등 가중 분위 격자 근사를 사용.
    """
    qs = sorted([float(k[1:]) / 100 for k in qvals.keys()])
    preds = np.vstack([qvals[f"q{int(q*100):02d}"] for q in qs])  # (n_q, n)
    losses = []
    for i, q in enumerate(qs):
        losses.append(_pinball_loss(y_true, preds[i], q))
    return float(np.mean(losses))


def _coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside.astype(float))) if len(inside) else float("nan")


def run_rolling_validation(
    models: Dict[str, QuantileRegressionModel],
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_window: int = 120,
    test_window: int = 12,
    step_size: int = 1,
    config: Optional[ValidationConfig] = None,
) -> Dict[str, Any]:
    """롤링 OOS 평가 실행.

    v0.4.1 Lean: 각 스텝마다 재학습(retrain_each_step=True) 권장. 데이터가 작을 경우만 사용 가능.
    성능 상 이유로 재학습을 생략(retrain_each_step=False)하면 주어진 models로 전기간 예측.

    Returns:
        {
          'by_date': {date: {horizon: {metric: value, ...}}},
          'summary': {horizon: {metric: avg_value, ...}}
        }
    """
    if config is None:
        # quantiles는 제공된 models에서 유추
        any_model = next(iter(models.values())) if models else None
        config = ValidationConfig(horizons=list(models.keys()) or ["1M"], quantiles=(any_model.quantiles if any_model else DEFAULT_QUANTILES))

    horizons = config.horizons
    quantiles = config.quantiles

    idx = X.index
    n = len(idx)
    results_by_date: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] = {}

    if n < train_window + test_window:
        raise ValueError("검증에 필요한 데이터 길이가 부족합니다.")

    start = train_window
    end = n - test_window

    for start_idx in range(start, end + 1, step_size):
        train_slice = slice(start_idx - train_window, start_idx)
        test_slice = slice(start_idx, start_idx + test_window)

        X_train = X.iloc[train_slice]
        X_test = X.iloc[test_slice]
        y_test_all = y.iloc[test_slice]

        # 스텝 시점(테스트 시작 시점)의 날짜
        t0 = idx[start_idx]
        step_metrics: Dict[str, Dict[str, float]] = {}

        # 재학습 or 고정 모델
        if config.retrain_each_step:
            step_models = train_models(
                X_train,
                # y_train은 각 horizon별 컬럼을 X_train의 인덱스에 맞춰 잘라야 한다
                y.loc[X_train.index],
                horizons=horizons,
                quantiles=quantiles,
                config={"alpha": config.alpha, "solver": config.solver},
            )
        else:
            step_models = models

        # 각 horizon별 예측 및 메트릭 계산
        for h in horizons:
            target_col = f"return_{h}"
            if target_col not in y.columns or h not in step_models:
                continue

            y_true = y_test_all[target_col].values
            # 예측 생성 (모든 테스트 포인트에 대해)
            preds_q: Dict[str, List[float]] = {f"q{int(q*100):02d}": [] for q in quantiles}
            for i in range(len(X_test)):
                pred = step_models[h].predict(X_test.iloc[i:i+1])
                for q in quantiles:
                    preds_q[f"q{int(q*100):02d}"].append(float(pred[f"q{int(q*100):02d}"][0]))

            preds_q_arr = {k: np.array(v) for k, v in preds_q.items()}

            # 메트릭: 평균 핀볼손실(분위 평균), CRPS 근사, 커버리지(50%, 80%)
            pinballs = []
            for q in quantiles:
                pinballs.append(_pinball_loss(y_true, preds_q_arr[f"q{int(q*100):02d}"], q))
            pinball_mean = float(np.mean(pinballs)) if pinballs else float("nan")

            crps = _crps_from_quantiles(y_true, preds_q_arr)

            # 커버리지
            coverage_50 = float("nan")
            coverage_80 = float("nan")
            if {"q25", "q75"}.issubset(preds_q_arr.keys()):
                coverage_50 = _coverage(y_true, preds_q_arr["q25"], preds_q_arr["q75"])
            if {"q10", "q90"}.issubset(preds_q_arr.keys()):
                coverage_80 = _coverage(y_true, preds_q_arr["q10"], preds_q_arr["q90"])

            step_metrics[h] = {
                "pinball_loss": pinball_mean,
                "crps": crps,
                "coverage_50": coverage_50,
                "coverage_80": coverage_80,
            }

        results_by_date[t0] = step_metrics

    # 요약 집계
    summary: Dict[str, Dict[str, float]] = {}
    for h in horizons:
        collect: Dict[str, List[float]] = {"pinball_loss": [], "crps": [], "coverage_50": [], "coverage_80": []}
        for t0, metrics in results_by_date.items():
            if h in metrics:
                for k in collect.keys():
                    v = metrics[h].get(k)
                    if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        collect[k].append(float(v))
        if any(collect.values()):
            summary[h] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in collect.items()}

    logger.info("롤링 검증 완료", n_steps=len(results_by_date))

    return {"by_date": results_by_date, "summary": summary}


def create_validation_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """검증 결과를 요약 리포트 형태로 정리"""
    report = {
        "summary": results.get("summary", {}),
        "n_steps": len(results.get("by_date", {})),
    }
    return report

