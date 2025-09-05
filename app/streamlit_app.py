"""Streamlit 대시보드 앱

탭 A: 매크로 시황 요약(합성 점수/브레드스/레짐/지표 카드)
탭 B: S&P 500 팬 차트(예측 분포/게이지/시나리오/기여도)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from structlog import get_logger

from src.models_quantile import load_models, predict_fanchart
from src.scenarios import apply_shocks, compute_contributions, DEFAULT_SCENARIOS
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


def _list_model_snapshots() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.name for p in MODELS_DIR.iterdir() if p.is_dir()])


def _latest_snapshot() -> Optional[str]:
    snaps = _list_model_snapshots()
    return snaps[-1] if snaps else None


def _select_feature_row(features: pd.DataFrame) -> pd.Series:
    # 마지막 유효 행을 선택
    return features.iloc[-1].dropna()


def main() -> None:
    tab_a, tab_b = create_dashboard_layout()

    features = _load_features()
    prices = _load_prices()

    # 사이드바: 모델 스냅샷 선택 및 시나리오 입력
    with st.sidebar:
        st.header("⚙️ 설정 및 시나리오")
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
        if features is None:
            st.warning("피처 파일을 찾을 수 없습니다. 먼저 CLI로 ingest ➜ features를 실행하세요.")
        else:
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

        # 1M 차트 우선 표시
        horizon = st.selectbox("예측 기간", options=list(fancharts.keys()), index=0)
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
