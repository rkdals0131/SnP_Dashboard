"""자동화 파이프라인 모듈

웹 UI에서 CLI 파이프라인을 프로그래매틱하게 실행할 수 있도록
래퍼 함수들을 제공합니다.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict
import traceback

import pandas as pd
from structlog import get_logger

from .data_sources import FREDSource, FXSource, persist_parquet
from .align_vintage import (
    create_month_end_calendar,
    build_master_panel,
    apply_publication_delays,
    PUBLICATION_RULES,
)
from .features import build_feature_panel
from .targets import compute_forward_returns, save_targets
from .models_quantile import train_models, save_models, DEFAULT_QUANTILES

logger = get_logger()

# 데이터 디렉토리
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
TARGETS_DIR = DATA_DIR / "targets"
MODELS_DIR = Path("models")

# 기본 시리즈
DEFAULT_SERIES = [
    "DGS10", "DGS3MO", "DGS2",
    "VIXCLS", "SP500", "DCOILWTICO",
    "GOLDAMGBD228NLBM", "BAA", "BAMLH0A0HYM2",
    "UNRATE", "CPIAUCSL", "CPILFESL", "NFCI",
    "FEDFUNDS", "PAYEMS", "CIVPART", "JTSJOL", "DTWEXBGS"
]

DEFAULT_FX_PAIRS = ["USDKRW", "USDJPY", "USDEUR"]


class PipelineProgress:
    """파이프라인 진행 상황 추적"""

    def __init__(self) -> None:
        self.current_step: str = ""
        self.current_progress: float = 0.0
        self.total_steps: int = 0
        self.completed_steps: int = 0
        self.status: str = "idle"  # idle, running, success, error
        self.error_message: str = ""
        self.logs: List[str] = []

    def start(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.completed_steps = 0
        self.status = "running"
        self.error_message = ""
        self.logs = []

    def update(self, step: str, progress: float = 0.0) -> None:
        self.current_step = step
        self.current_progress = progress
        self.add_log(f"[{datetime.now().strftime('%H:%M:%S')}] {step}")

    def complete_step(self) -> None:
        self.completed_steps += 1
        self.current_progress = self.completed_steps / self.total_steps

    def finish(self, success: bool = True, error: str = "") -> None:
        self.status = "success" if success else "error"
        self.error_message = error
        self.current_progress = 1.0 if success else self.current_progress
        if success:
            self.add_log(f"[{datetime.now().strftime('%H:%M:%S')}] 완료!")
        else:
            self.add_log(f"[{datetime.now().strftime('%H:%M:%S')}] 오류: {error}")

    def add_log(self, message: str) -> None:
        self.logs.append(message)
        logger.info(message)


def run_ingest_pipeline(
    progress: Optional[PipelineProgress] = None,
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    series_list: Optional[List[str]] = None,
) -> bool:
    """데이터 수집 파이프라인 실행

    Args:
        progress: 진행 상황 추적 객체
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (없으면 오늘)
        series_list: 수집할 시리즈 목록 (없으면 기본 시리즈)

    Returns:
        성공 여부
    """
    if progress:
        progress.update("데이터 수집 시작")

    try:
        # 디렉토리 생성
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "fred").mkdir(exist_ok=True)
        (RAW_DIR / "fx").mkdir(exist_ok=True)

        # 날짜 파싱
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date() if end_date else date.today()

        # 시리즈 목록
        series = series_list or DEFAULT_SERIES

        # FRED 데이터 수집
        if progress:
            progress.update(f"FRED 데이터 수집 중 ({len(series)}개 시리즈)")

        fred = FREDSource()
        fred_data = fred.fetch(series, start, end)

        if progress:
            progress.update(f"FRED 데이터 수집 완료: {len(fred_data)}개 시리즈")

        # FX 데이터 수집
        if progress:
            progress.update(f"FX 데이터 수집 중 ({len(DEFAULT_FX_PAIRS)}개)")

        fx = FXSource()
        fx_data = fx.fetch(DEFAULT_FX_PAIRS, start, end)

        if progress:
            progress.update(f"FX 데이터 수집 완료: {len(fx_data)}개")

        total_series = len(fred_data) + len(fx_data)

        if progress:
            progress.add_log(f"총 {total_series}개 시리즈 수집 완료")

        return True

    except Exception as e:
        error_msg = f"데이터 수집 실패: {str(e)}"
        if progress:
            progress.add_log(error_msg)
            progress.add_log(traceback.format_exc())
        logger.error(error_msg, error=str(e))
        return False


def run_features_pipeline(
    progress: Optional[PipelineProgress] = None,
    calendar_start: str = "2010-01-01",
    calendar_end: Optional[str] = None,
) -> bool:
    """피처 생성 파이프라인 실행

    Args:
        progress: 진행 상황 추적 객체
        calendar_start: 캘린더 시작일
        calendar_end: 캘린더 종료일 (없으면 오늘)

    Returns:
        성공 여부
    """
    if progress:
        progress.update("피처 생성 시작")

    try:
        # 디렉토리 생성
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        TARGETS_DIR.mkdir(parents=True, exist_ok=True)

        # 원천 데이터 경로 수집
        if progress:
            progress.update("원천 데이터 확인 중")

        raw_paths: Dict[str, Path] = {}

        # FRED 데이터
        fred_dir = RAW_DIR / "fred"
        if fred_dir.exists():
            for p in fred_dir.glob("*.parquet"):
                raw_paths[p.stem] = p

        # FX 데이터
        fx_dir = RAW_DIR / "fx"
        if fx_dir.exists():
            for p in fx_dir.glob("*.parquet"):
                raw_paths[p.stem] = p

        if not raw_paths:
            raise RuntimeError("원천 데이터가 없습니다. 먼저 데이터 수집을 실행하세요.")

        if progress:
            progress.add_log(f"원천 데이터: {len(raw_paths)}개 시리즈")

        # 캘린더 생성
        if progress:
            progress.update("월말 캘린더 생성 중")

        end = calendar_end or date.today().strftime("%Y-%m-%d")
        calendar = create_month_end_calendar(calendar_start, end)

        if progress:
            progress.add_log(f"캘린더 생성 완료: {len(calendar)}개 월말")

        # 마스터 패널 구축
        if progress:
            progress.update("마스터 패널 구축 중")

        master_panel = build_master_panel(raw_paths, calendar)

        if progress:
            progress.add_log(f"마스터 패널: {master_panel.shape}")

        # 빈티지 처리
        if progress:
            progress.update("발표 지연 적용 중")

        vintage_panel = apply_publication_delays(master_panel, PUBLICATION_RULES)

        # 마스터 패널 저장
        master_path = PROCESSED_DIR / "master_timeseries.parquet"
        vintage_panel.to_parquet(master_path)

        if progress:
            progress.add_log(f"마스터 패널 저장: {master_path}")

        # 피처 생성
        if progress:
            progress.update("피처 생성 중")

        features = build_feature_panel(vintage_panel)

        # 피처 저장
        features_path = FEATURES_DIR / "features.parquet"
        features.to_parquet(features_path)

        if progress:
            progress.add_log(f"피처 저장: {features_path} ({features.shape})")

        # 타깃 생성
        if progress:
            progress.update("타깃 생성 중")

        if "SP500" not in vintage_panel.columns:
            raise RuntimeError("타깃 생성에 필요한 SP500 데이터가 없습니다.")

        targets = compute_forward_returns(
            vintage_panel["SP500"],
            horizons=["1M", "3M", "6M", "12M"]
        )

        # 타깃 저장
        targets_path = TARGETS_DIR / "targets.parquet"
        save_targets(targets, targets_path)

        if progress:
            progress.add_log(f"타깃 저장: {targets_path} ({targets.shape})")

        return True

    except Exception as e:
        error_msg = f"피처 생성 실패: {str(e)}"
        if progress:
            progress.add_log(error_msg)
            progress.add_log(traceback.format_exc())
        logger.error(error_msg, error=str(e))
        return False


def run_training_pipeline(
    progress: Optional[PipelineProgress] = None,
    horizons: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    alpha: float = 1.0,
) -> bool:
    """모델 학습 파이프라인 실행

    Args:
        progress: 진행 상황 추적 객체
        horizons: 학습할 기간 목록 (예: ["1M", "3M"])
        quantiles: 분위수 목록
        alpha: L1 규제 강도

    Returns:
        성공 여부
    """
    if progress:
        progress.update("모델 학습 시작")

    try:
        # 디렉토리 생성
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # 피처와 타깃 로드
        if progress:
            progress.update("피처 및 타깃 로드 중")

        features_path = FEATURES_DIR / "features.parquet"
        targets_path = TARGETS_DIR / "targets.parquet"

        if not features_path.exists():
            raise RuntimeError(f"피처 파일이 없습니다: {features_path}")
        if not targets_path.exists():
            raise RuntimeError(f"타깃 파일이 없습니다: {targets_path}")

        features = pd.read_parquet(features_path)
        targets = pd.read_parquet(targets_path)

        if progress:
            progress.add_log(f"피처: {features.shape}, 타깃: {targets.shape}")

        # 학습
        if progress:
            progress.update("모델 학습 중")

        horizons_list = horizons or ["1M", "3M", "6M", "12M"]
        quantiles_list = quantiles or DEFAULT_QUANTILES

        models = train_models(
            features=features,
            targets=targets,
            horizons=horizons_list,
            quantiles=quantiles_list,
            alpha=alpha,
        )

        if progress:
            progress.add_log(f"학습 완료: {len(models)}개 horizon × {len(quantiles_list)}개 quantile")

        # 스냅샷 저장
        snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = MODELS_DIR / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        models_path = snapshot_dir / "models.pkl"
        save_models(models, models_path)

        if progress:
            progress.add_log(f"모델 저장: {models_path}")

        return True

    except Exception as e:
        error_msg = f"모델 학습 실패: {str(e)}"
        if progress:
            progress.add_log(error_msg)
            progress.add_log(traceback.format_exc())
        logger.error(error_msg, error=str(e))
        return False


def run_full_pipeline(
    progress: Optional[PipelineProgress] = None,
    start_date: str = "2010-01-01",
) -> bool:
    """전체 파이프라인 실행 (데이터 수집 → 피처 생성 → 모델 학습)

    Args:
        progress: 진행 상황 추적 객체
        start_date: 시작 날짜

    Returns:
        성공 여부
    """
    if progress:
        progress.start(total_steps=3)

    try:
        # 1. 데이터 수집
        if progress:
            progress.update("1/3: 데이터 수집")

        if not run_ingest_pipeline(progress, start_date=start_date):
            raise RuntimeError("데이터 수집 실패")

        if progress:
            progress.complete_step()

        # 2. 피처 생성
        if progress:
            progress.update("2/3: 피처 생성")

        if not run_features_pipeline(progress, calendar_start=start_date):
            raise RuntimeError("피처 생성 실패")

        if progress:
            progress.complete_step()

        # 3. 모델 학습
        if progress:
            progress.update("3/3: 모델 학습")

        if not run_training_pipeline(progress):
            raise RuntimeError("모델 학습 실패")

        if progress:
            progress.complete_step()
            progress.finish(success=True)

        return True

    except Exception as e:
        error_msg = f"전체 파이프라인 실패: {str(e)}"
        if progress:
            progress.finish(success=False, error=error_msg)
        logger.error(error_msg, error=str(e))
        return False
