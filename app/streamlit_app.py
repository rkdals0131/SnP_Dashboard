"""Streamlit 대시보드 앱

탭 A: 매크로 시황 요약(합성 점수/브레드스/레짐/지표 카드)
탭 B: S&P 500 팬 차트(예측 분포/게이지/시나리오/기여도)
"""

from __future__ import annotations

from pathlib import Path
from datetime import date
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from structlog import get_logger

from src.models_quantile import load_models, predict_fanchart
from src.scenarios import apply_shocks, compute_contributions, DEFAULT_SCENARIOS
from src.data_sources import FREDSource
from src.align_vintage import (
    create_month_end_calendar,
    build_master_panel,
    apply_publication_delays,
    PUBLICATION_RULES,
)
from src.features import build_feature_panel
from src.viz import (
    create_dashboard_layout,
    render_macro_score_tile,
    render_breadth_bar,
    render_regime_badges,
    render_indicator_card,
    render_fan_chart,
    render_uncertainty_gauge,
    render_contribution_chart,
    render_scenario_comparison,
    render_auto_summary,
)


logger = get_logger()

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features" / "features.parquet"
PROCESSED_PATH = DATA_DIR / "processed" / "master_timeseries.parquet"
MODELS_DIR = Path("models")

# .env 로드 (선택)
# .env 로드 (선택)
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except Exception:
    pass

# 패키지 경로 보정: 프로젝트 루트를 sys.path에 추가하여 `src` 임포트 가능하게 함
try:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except Exception:
    pass


def _load_features() -> Optional[pd.DataFrame]:
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    return None


def _load_prices() -> Optional[pd.DataFrame]:
    if PROCESSED_PATH.exists():
        df = pd.read_parquet(PROCESSED_PATH)
        if "SP500" in df.columns:
            return df[["SP500"]]
    return None


def _default_series_for_tab_a() -> List[str]:
    # CLI의 기본 셋과 정합(필요 핵심 위주)
    return [
        "DGS10", "DGS3MO", "DGS2",  # 국채 수익률
        "VIXCLS",                      # VIX
        "SP500",                       # S&P 500
        "DCOILWTICO",                  # WTI 유가
        "BAA", "BAMLH0A0HYM2",       # 크레딧 스프레드
        "UNRATE",                      # 실업률
        "CPIAUCSL", "CPILFESL",      # 인플레이션
        "NFCI"                         # 금융여건 지수
    ]


@st.cache_data(ttl=300, show_spinner=False)
def _build_live_features_for_tab_a(
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """FRED에서 실시간 수집 → 정합/빈티지 → 피처 계산(탭 A용)

    Notes:
        - 5분 캐시 적용(ttl=300)
        - 환경변수 FRED_API_KEY 필요
    """
    # 기간 설정
    end_str = end or date.today().strftime("%Y-%m-%d")

    # 데이터 수집(캐시 저장 병행)
    fred = FREDSource()  # env FRED_API_KEY 사용
    series_list = _default_series_for_tab_a()
    _ = fred.fetch(series_list, pd.to_datetime(start).date(), pd.to_datetime(end_str).date())

    # 원천 경로 구성(방금 저장된 캐시 사용)
    raw_dir = Path("data/raw/fred")
    raw_paths: Dict[str, Path] = {}
    for sid in series_list:
        p = raw_dir / f"{sid}.parquet"
        if p.exists():
            raw_paths[sid] = p

    if not raw_paths:
        raise RuntimeError("원천 데이터를 찾을 수 없습니다(FRED/API/네트워크 확인).")

    # 캘린더 → 마스터 패널 → 발표 지연 적용
    calendar = create_month_end_calendar(start, end_str)
    master_panel = build_master_panel(raw_paths, calendar)
    vintage_panel = apply_publication_delays(master_panel, PUBLICATION_RULES)

    # 피처 패널 구축(레벨/Δ/Δ²/분위/브레드스/레짐)
    feature_panel = build_feature_panel(vintage_panel)
    return feature_panel


def _list_model_snapshots() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.name for p in MODELS_DIR.iterdir() if p.is_dir()])


def _latest_snapshot() -> Optional[str]:
    snaps = _list_model_snapshots()
    return snaps[-1] if snaps else None


def _select_feature_row(features: pd.DataFrame) -> pd.Series:
    """마지막 유효 관측치를 선택

    월말 빈티지 처리로 인해 최신 행에 결측이 많을 수 있으므로
    시계열 방향으로 forward-fill 후 마지막 행을 사용합니다.
    모든 값이 결측인 경우를 대비해 back-fill도 시도합니다.
    """
    if features.empty:
        return pd.Series(dtype=float)
    filled = features.ffill().bfill()
    return filled.iloc[-1]


def main() -> None:
    tab_a, tab_b = create_dashboard_layout()

    # 사이드바: 설정/시나리오 및 탭 A 새로고침
    with st.sidebar:
        st.header("설정 및 시나리오")
        refresh = st.button("탭 A 데이터 새로고침")
        if refresh:
            st.cache_data.clear()

    # 데이터 준비(탭 A: 우선 실시간, 실패 시 파일)
    features_live: Optional[pd.DataFrame] = None
    features_file = _load_features()
    prices = _load_prices()

    try:
        features_live = _build_live_features_for_tab_a()
    except Exception as e:
        logger.warning("라이브 피처 생성 실패, 파일로 대체", error=str(e))

    # 사이드바: 모델 스냅샷 선택 및 시나리오 입력
    with st.sidebar:
        # 위에서 헤더/버튼 렌더링됨
        snaps = _list_model_snapshots()
        default_snap = _latest_snapshot()
        snapshot = st.selectbox("모델 스냅샷", options=["(없음)"] + snaps, index=(snaps.index(default_snap) + 1) if default_snap else 0)

        # 시나리오 슬라이더(시즌 1)
        st.subheader("시나리오 쇼크")
        shock_defs = {
            "DGS10": st.slider("10Y UST (bp)", -100, 100, 0) / 100.0,
            "DGS3MO": st.slider("3M UST (bp)", -100, 100, 0) / 100.0,
            "VIXCLS": st.slider("VIX (포인트)", -20, 20, 0),
            "DCOILWTICO": st.slider("WTI (달러)", -20, 20, 0),
            "NFCI": st.slider("NFCI", -1, 1, 0),
            "CPIAUCSL": st.slider("CPI 서프라이즈", -1, 1, 0),
        }
        use_preset = st.selectbox("사전 시나리오", options=["(없음)"] + list(DEFAULT_SCENARIOS.keys()))

    # 탭 A — 매크로 시황 요약
    with tab_a:
        st.subheader("매크로 시황 요약")
        # 우선순위: 라이브 → 파일 → 경고
        features = features_live if features_live is not None else features_file
        if features is None:
            st.warning("실시간/파일 데이터가 없습니다. FRED_API_KEY 설정 후 재시도하거나 CLI로 ingest ➜ features를 실행하세요.")
        else:
            if features_live is not None:
                st.caption("데이터 소스: 실시간(FRED, 5분 캐시)")
            else:
                st.caption("데이터 소스: 파일(features.parquet)")
            # 최근 값 기준 간단 요약
            last_row = _select_feature_row(features)
            score = float(last_row.get("MacroScore", 0))
            momentum = float(last_row.filter(like="_d1_z").mean()) if any(last_row.index.str.endswith("_d1_z")) else 0.0
            acceleration = float(last_row.filter(like="_d2_z").mean()) if any(last_row.index.str.endswith("_d2_z")) else 0.0
            render_macro_score_tile(score, momentum, acceleration)

            breadth = float(last_row.get("MacroBreadth_top50", np.nan)) if "MacroBreadth_top50" in features.columns else np.nan
            breadth_delta = float(last_row.get("MacroBreadth_top50_delta", 0)) if "MacroBreadth_top50_delta" in features.columns else 0.0
            if not np.isnan(breadth):
                render_breadth_bar(breadth, breadth_delta, "상위 50%")

            regime_cols = [c for c in features.columns if c.startswith("Regime_")]
            if regime_cols:
                regimes = {c.replace("Regime_", ""): str(last_row.get(c)) for c in regime_cols}
                render_regime_badges(regimes)

            # 지표 카드(샘플)
            st.markdown("---")
            st.subheader("핵심 지표")
            key_inds = ["VIXCLS", "DGS10", "DGS3MO", "TERM_SPREAD", "NFCI", "DCOILWTICO"]
            cols = st.columns(3)
            for i, ind in enumerate(key_inds):
                if ind in features.columns:
                    level = float(last_row.get(ind, np.nan))
                    level_score = float(last_row.get(f"{ind}_pctscore", np.nan))
                    mom = float(last_row.get(f"{ind}_d1", np.nan))
                    acc = float(last_row.get(f"{ind}_d2", np.nan))
                    pct = (level_score + 1) / 2 if not np.isnan(level_score) else np.nan
                    with cols[i % 3]:
                        render_indicator_card(ind, level, level_score, mom, acc, pct)

            # 자동 요약(간단 스텁)
            changes = []
            if not np.isnan(breadth_delta) and abs(breadth_delta) >= 0.1:
                changes.append({
                    "indicator": "MacroBreadth(상위50%)",
                    "direction": "up" if breadth_delta > 0 else "down",
                    "magnitude": breadth_delta * 100,
                    "context": "브레드스 임계 변화"
                })
            render_auto_summary(changes)

    # 탭 B — 팬 차트
    with tab_b:
        st.subheader("S&P 500 팬 차트")

        if snapshot == "(없음)":
            st.warning("모델 스냅샷이 없습니다. CLI의 train 명령으로 모델을 학습하세요.")
            return
        
        model_path = MODELS_DIR / snapshot / "models.pkl"
        if not model_path.exists():
            st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return

        if features is None or prices is None:
            st.warning("필요 데이터(features/prices)가 없습니다. CLI 파이프라인을 먼저 실행하세요.")
            return

        models = load_models(model_path)

        # 모델이 비어있는 경우 처리
        if not models:
            st.warning("선택한 스냅샷에 학습된 모델이 없습니다. 충분한 데이터로 'train'을 다시 실행하세요.")
            return

        # 현재 특성행과 시나리오 적용
        x_t = _select_feature_row(features)

        # 사전 시나리오 적용 선택 시 덮어쓰기
        shocks = {k: v for k, v in shock_defs.items() if v != 0}
        if use_preset and use_preset in DEFAULT_SCENARIOS:
            shocks.update(DEFAULT_SCENARIOS[use_preset]["shocks"])  # preset 우선 적용

        if shocks:
            x_shocked = apply_shocks(x_t, shocks)
        else:
            x_shocked = x_t

        # 현재 가격 및 VIX
        current_price = float(prices.iloc[-1]["SP500"]) if not prices.empty else 0.0
        vix_val = float(x_shocked.get("VIXCLS", x_t.get("VIXCLS", 20)))

        # 팬차트 생성(전 기간)
        fancharts = predict_fanchart(models, x_shocked, vix_current=vix_val)

        # 팬차트가 비어있는 경우 처리
        if not fancharts:
            st.warning("예측 결과가 없습니다. 모델 또는 입력 데이터를 확인하세요.")
            return

        # 1M 우선 선택(있으면), 없으면 첫 항목
        keys = list(fancharts.keys())
        default_index = keys.index("1M") if "1M" in keys else 0
        horizon = st.selectbox("예측 기간", options=keys, index=default_index)
        render_fan_chart(fancharts[horizon], prices, current_price, show_history=True)

        # 불확실성 게이지(간단): 80% 폭/12M 최근 실현 변동
        q = fancharts[horizon]["quantiles"]
        if all(k in q for k in ("q10", "q90")):
            iqr_width = float(q["q90"][0] - q["q10"][0])
            # 역사 평균: 최근 24개월 실현 수익률 폭 근사
            historical_avg = 0.0
            if prices is not None and len(prices) > 24:
                rets = np.log(prices["SP500"]).diff(21).dropna()
                historical_avg = float(rets.rolling(24).std().dropna().mean() * 2)
            historical_avg = historical_avg or max(iqr_width, 1e-3)
            render_uncertainty_gauge(iqr_width, historical_avg)

        # 기여도(중앙, 계수×Δ특징량)
        if models and horizon in models:
            # Δ는 시나리오 쇼크만 고려(단순)
            delta = (x_shocked - x_t).fillna(0)
            contrib = compute_contributions(models[horizon], delta, quantile=0.5)
            if not contrib.empty:
                render_contribution_chart(contrib, top_n=5)


if __name__ == "__main__":
    main()
