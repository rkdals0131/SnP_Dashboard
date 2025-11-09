"""시각화 컴포넌트 모듈

Streamlit과 Plotly를 사용하여 탭 A(신호 집계판)와 
탭 B(팬 차트)의 UI 컴포넌트를 렌더링.

Example:
    >>> import streamlit as st
    >>> from viz import render_macro_score_tile, render_fan_chart
    >>> render_macro_score_tile(macro_score, momentum, acceleration)
    >>> render_fan_chart(fanchart_data, price_history)
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from structlog import get_logger

logger = get_logger()


# 색상 팔레트
COLOR_PALETTE = {
    "positive": "#22c55e",  # 녹색
    "negative": "#ef4444",  # 빨간색
    "neutral": "#94a3b8",   # 회색
    "primary": "#3b82f6",   # 파란색
    "warning": "#f59e0b",   # 주황색
    "background": "#f8fafc",
    "text": "#1e293b"
}

# 레짐 색상
REGIME_COLORS = {
    "easy": "#22c55e",
    "tight": "#ef4444",
    "neutral": "#94a3b8",
    "expanding": "#3b82f6",
    "slowing": "#f59e0b",
    "mixed": "#94a3b8",
    "rising": "#ef4444",
    "falling": "#22c55e",
    "stable": "#94a3b8"
}


# 툴팁 정의
TOOLTIP_DEFINITIONS = {
    # Macro tiles
    "MacroScore": "여러 카테고리의 레벨·모멘텀·가속도를 가중 평균한 합성 점수입니다. +는 우호적, -는 비우호적.",
    "모멘텀": "최근 변화의 방향과 크기(1차 차분의 표준화).",
    "Momentum": "최근 변화의 방향과 크기(1차 차분의 표준화).",
    "가속도": "변화의 변화(2차 차분의 표준화). 추세 강화/둔화 판단.",
    "Acceleration": "변화의 변화(2차 차분의 표준화). 추세 강화/둔화 판단.",

    # Breadth
    "MacroBreadth_top50": "상위 50% 분위 이상에 위치한 지표의 비중입니다. 시장·지표의 동조화 정도를 나타냅니다.",
    "MacroBreadth_top20": "상위 20% 분위 이상에 위치한 지표의 비중입니다.",

    # Indicators
    "VIXCLS": "S&P 500 옵션 내재변동성(VIX). 향후 30일 변동성 기대.",
    "TERM_SPREAD": "장단기 금리차(10년물−3개월물). 경기 국면 시그널로 활용.",
    "DGS10": "미국 10년 만기 국채 금리.",
    "DGS3MO": "미국 3개월 만기 국채 금리.",
    "DCOILWTICO": "WTI 원유 가격.",
    "GOLDAMGBD228NLBM": "금 가격(LBMA 오전 고시, 달러/온스).",
    "NFCI": "시카고 연은 금융여건지수. 0보다 크면 긴축, 작으면 완화.",
    # Labor / Inflation / Policy
    "UNRATE": "실업률(%) — 노동시장 상황.",
    "PAYEMS": "비농업 신규고용(천 명).",
    "CIVPART": "경제활동참가율(%).",
    "FEDFUNDS": "연방기금 금리(월 평균).",
    "JTSJOL": "구인건수(JOLTS, 백만).",
    # FX
    "USDKRW": "미달러/원 환율(USDKRW). 값 상승=원화 약세.",
    "USDJPY": "미달러/엔 환율(USDJPY). 값 상승=엔화 약세.",
    "USDEUR": "미달러/유로 환율(USDEUR). 값 상승=유로 약세.",
    "DTWEXBGS": "미 달러(광의) 무역가중지수.",
}

def render_macro_score_tile(
    score: float,
    momentum: float,
    acceleration: float,
    title: str = "MacroScore"
) -> None:
    """MacroScore 타일 렌더링
    
    Args:
        score: MacroScore 값 (-1 ~ +1)
        momentum: 모멘텀 값
        acceleration: 가속도 값
        title: 타일 제목
    """
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # 메인 스코어 표시
        score_color = COLOR_PALETTE["positive"] if score > 0 else COLOR_PALETTE["negative"]
        st.metric(
            label=title,
            value=f"{score:.3f}",
            delta=f"Δ {momentum:.3f}",
            delta_color="normal" if momentum > 0 else "inverse",
            help=TOOLTIP_DEFINITIONS.get(title, "")
        )
        
        # 진행 바
        progress = (score + 1) / 2  # -1~1을 0~1로 변환
        st.progress(progress)
    
    with col2:
        st.metric(
            label="모멘텀",
            value=f"{momentum:.3f}",
            delta=None,
            help=TOOLTIP_DEFINITIONS.get("모멘텀", TOOLTIP_DEFINITIONS.get("Momentum", ""))
        )
    
    with col3:
        st.metric(
            label="가속도",
            value=f"{acceleration:.3f}",
            delta=None,
            help=TOOLTIP_DEFINITIONS.get("가속도", TOOLTIP_DEFINITIONS.get("Acceleration", ""))
        )


def render_breadth_bar(
    breadth: float,
    breadth_delta: float,
    threshold: str = "상위 50%"
) -> None:
    """브레드스 바 렌더링
    
    Args:
        breadth: 브레드스 값 (0 ~ 1)
        breadth_delta: 브레드스 변화량
        threshold: 임계값 설명
    """
    st.subheader(f"브레드스 ({threshold})")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 브레드스 바 차트
        fig = go.Figure()
        
        # 배경 바
        fig.add_trace(go.Bar(
            x=[1 - breadth],
            y=["브레드스"],
            orientation='h',
            marker=dict(color=COLOR_PALETTE["neutral"], opacity=0.3),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # 실제 브레드스
        color = COLOR_PALETTE["positive"] if breadth > 0.5 else COLOR_PALETTE["negative"]
        fig.add_trace(go.Bar(
            x=[breadth],
            y=["브레드스"],
            orientation='h',
            marker=dict(color=color),
            showlegend=False,
            text=[f"{breadth:.1%}"],
            textposition='inside'
        ))
        
        fig.update_layout(
            barmode='stack',
            height=80,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        delta_color = "normal" if breadth_delta > 0 else "inverse"
        st.metric(
            label="변화",
            value=f"{breadth:.1%}",
            delta=f"{breadth_delta:+.1%}",
            delta_color=delta_color,
            help=TOOLTIP_DEFINITIONS.get(
                "MacroBreadth_top50" if "50" in threshold else "MacroBreadth_top20",
                "브레드스(상위 분위 이상 지표 비중)입니다."
            )
        )


def render_regime_badges(regimes: Dict[str, str]) -> None:
    """레짐 배지 렌더링
    
    Args:
        regimes: 레짐 타입별 현재 상태
    """
    cols = st.columns(len(regimes))
    
    for idx, (regime_type, regime_value) in enumerate(regimes.items()):
        with cols[idx]:
            # 레짐 타입별 한글명
            type_names = {
                "financial_conditions": "금융여건",
                "growth_momentum": "경기모멘텀",
                "inflation_momentum": "물가모멘텀"
            }
            
            # 레짐 값별 한글명
            value_names = {
                "easy": "완화",
                "tight": "긴축",
                "neutral": "중립",
                "expanding": "확장",
                "slowing": "둔화",
                "mixed": "혼재",
                "rising": "상승",
                "falling": "하락",
                "stable": "안정"
            }
            
            type_name = type_names.get(regime_type, regime_type)
            value_name = value_names.get(regime_value, regime_value)
            color = REGIME_COLORS.get(regime_value, COLOR_PALETTE["neutral"])
            
            # 배지 렌더링
            st.markdown(
                f"""
                <div style="
                    background-color: {color}20;
                    border: 2px solid {color};
                    border-radius: 20px;
                    padding: 8px 16px;
                    text-align: center;
                ">
                    <div style="font-size: 12px; color: #64748b;">{type_name}</div>
                    <div style="font-size: 16px; font-weight: bold; color: {color};">
                        {value_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_indicator_card(
    name: str,
    value: float,
    level_score: float,
    momentum: float,
    acceleration: float,
    percentile: float,
    unit: str = ""
) -> None:
    """개별 지표 카드 렌더링
    
    Args:
        name: 지표명
        value: 현재 값
        level_score: 레벨 스코어 (-1 ~ 1)
        momentum: 모멘텀
        acceleration: 가속도
        percentile: 역사적 백분위
        unit: 단위
    """
    with st.container():
        st.markdown(f"### {name}")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # 현재 값과 백분위
            value_str = f"{value:.2f}{unit}" if unit else f"{value:.2f}"
            st.metric(
                label="현재 값",
                value=value_str,
                delta=f"상위 {percentile:.0%}",
                help=TOOLTIP_DEFINITIONS.get(name, f"{name} 지표입니다.")
            )
        
        with col2:
            # 모멘텀
            momentum_color = "normal" if momentum > 0 else "inverse"
            st.metric(
                label="Δ",
                value=f"{momentum:.3f}",
                delta=None,
                help=TOOLTIP_DEFINITIONS.get("모멘텀", TOOLTIP_DEFINITIONS.get("Momentum", ""))
            )
        
        with col3:
            # 가속도
            accel_color = "normal" if acceleration > 0 else "inverse"
            st.metric(
                label="Δ²",
                value=f"{acceleration:.3f}",
                delta=None,
                help=TOOLTIP_DEFINITIONS.get("가속도", TOOLTIP_DEFINITIONS.get("Acceleration", ""))
            )
        
        # 미니 차트 (스파크라인)
        # TODO: 실제 히스토리 데이터로 교체
        st.markdown("---")


def _classify_percentile(pct: float) -> str:
    if np.isnan(pct):
        return "자료부족"
    if pct >= 0.8:
        return "매우 높음"
    if pct >= 0.6:
        return "높음"
    if pct <= 0.2:
        return "매우 낮음"
    if pct <= 0.4:
        return "낮음"
    return "중립"


def generate_indicator_narrative(
    name: str,
    last_row: pd.Series,
    units: Optional[Dict[str, str]] = None
) -> str:
    """단일 지표에 대한 자연어 설명 생성(규칙 기반)

    - 현재 값과 10년 백분위 수준
    - 최근 변화(Δ) 방향
    - 가속도(Δ²) 유무
    """
    units = units or {}
    level = last_row.get(name, np.nan)
    pctscore = last_row.get(f"{name}_pctscore", np.nan)
    mom = last_row.get(f"{name}_d1", np.nan)
    acc = last_row.get(f"{name}_d2", np.nan)

    pct = (pctscore + 1) / 2 if not pd.isna(pctscore) else np.nan
    lvl_str = f"{level:.2f}{units.get(name, '')}" if not pd.isna(level) else "N/A"
    pct_str = f"{(pct if not pd.isna(pct) else 0):.0%}" if not pd.isna(pct) else "자료부족"
    pct_bucket = _classify_percentile(pct) if not pd.isna(pct) else "자료부족"

    direction = "상승" if (not pd.isna(mom) and mom > 0) else "하락" if (not pd.isna(mom) and mom < 0) else "변화없음"
    accel = (
        "가속" if (not pd.isna(acc) and acc > 0) else
        "감속" if (not pd.isna(acc) and acc < 0) else
        "안정"
    )

    return f"{name}: 현재 {lvl_str}, 10년 대비 {pct_str}({pct_bucket}). 최근 {direction}, {accel} 경향."


def render_indicator_narratives(
    indicators: List[str],
    last_row: pd.Series,
    units: Optional[Dict[str, str]] = None,
) -> None:
    """지표별 규칙 기반 자연어 요약을 리스트로 렌더링"""
    for ind in indicators:
        if ind in last_row.index or f"{ind}_pctscore" in last_row.index:
            st.markdown(f"- {generate_indicator_narrative(ind, last_row, units)}")


def render_ai_one_liner(
    last_row: pd.Series,
    indicators: List[str],
    model: Optional[str] = None,
    max_tokens: int = 60
) -> None:
    """Gemini를 사용한 한 줄 요약(옵션)

    - 환경변수 GEMINI_API_KEY 또는 GOOGLE_API_KEY가 필요하며, `google-generativeai` 패키지를 사용합니다.
    - 사용 불가 시 규칙 기반 한 줄 요약으로 대체합니다.
    """
    # 준비: 컨텍스트 추출
    ctx = []
    for ind in indicators:
        level = last_row.get(ind, np.nan)
        pctscore = last_row.get(f"{ind}_pctscore", np.nan)
        mom = last_row.get(f"{ind}_d1", np.nan)
        acc = last_row.get(f"{ind}_d2", np.nan)
        pct = (pctscore + 1) / 2 if not pd.isna(pctscore) else np.nan
        ctx.append(
            {
                "name": ind,
                "level": None if pd.isna(level) else float(level),
                "pct": None if pd.isna(pct) else float(pct),
                "d1": None if pd.isna(mom) else float(mom),
                "d2": None if pd.isna(acc) else float(acc),
            }
        )

    # 규칙 기반 백업 문장
    def fallback_line() -> str:
        parts = []
        for c in ctx[:5]:
            if c["pct"] is None:
                continue
            bucket = _classify_percentile(c["pct"]) if c["pct"] is not None else "자료부족"
            dir_txt = "↑" if (c["d1"] or 0) > 0 else "↓" if (c["d1"] or 0) < 0 else "→"
            parts.append(f"{c['name']} {bucket}{dir_txt}")
        return " / ".join(parts) or "핵심 지표 변화가 제한적입니다."

    # Gemini 호출 시도
    try:
        import os
        import google.generativeai as genai  # type: ignore

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY 없음")

        genai.configure(api_key=api_key)

        # 모델 선택: 명시 인자 → 환경변수 → 합리적 기본/폴백
        env_model = os.getenv("GEMINI_MODEL")
        candidate_models: List[str] = []
        if model:
            candidate_models = [model]
        elif env_model:
            candidate_models = [env_model]
        else:
            candidate_models = [
                "gemini-2.5-flash-lite",  # 요청 우선
                "gemini-2.5-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash",
                "gemini-1.5-flash-8b",
                "gemini-1.5-flash",
            ]

        prompt = (
            "다음 핵심 지표의 현재 수준(10년 백분위)과 최근 변화 방향을 바탕으로, "
            "S&P 500에 대한 위험/유동성/물가/성장 환경을 1문장(한국어)으로 중립적으로 요약하세요. "
            "숫자 나열 대신 '변동성 완화', '달러 강세 지속'처럼 요점만 기술: "
            f"{ctx}"
        )

        gen_cfg = genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=max_tokens,
        )

        last_error: Optional[Exception] = None
        text_out: Optional[str] = None
        for m in candidate_models:
            try:
                mdl = genai.GenerativeModel(m)
                resp = mdl.generate_content(prompt, generation_config=gen_cfg)
                # 일부 SDK 버전은 resp.text, 일부는 candidates[0].content.parts
                if hasattr(resp, "text") and resp.text:
                    text_out = resp.text.strip()
                elif getattr(resp, "candidates", None):
                    parts = resp.candidates[0].content.parts
                    text_out = "".join(getattr(p, "text", "") for p in parts).strip()
                if text_out:
                    break
            except Exception as e:  # 다음 후보로 폴백
                last_error = e
                continue

        if text_out:
            st.success(text_out)
        else:
            raise RuntimeError(str(last_error) if last_error else "응답 없음")

    except Exception as e:
        st.info(f"AI 요약 비활성화({e}). 대체 요약: {fallback_line()}")


def render_fan_chart(
    fanchart_data: Dict[str, Any],
    price_history: pd.DataFrame,
    current_price: float,
    show_history: bool = True
) -> None:
    """팬 차트 렌더링
    
    Args:
        fanchart_data: 팬차트 데이터 (quantiles 포함)
        price_history: 과거 가격 데이터
        current_price: 현재 가격
        show_history: 과거 데이터 표시 여부
    """
    fig = go.Figure()
    
    # 과거 가격 데이터
    if show_history and not price_history.empty:
        fig.add_trace(go.Scatter(
            x=price_history.index,
            y=price_history["SP500"],
            mode='lines',
            name='S&P 500',
            line=dict(color=COLOR_PALETTE["primary"], width=2)
        ))
    
    # 팬차트 예측
    horizon = fanchart_data["horizon"]
    quantiles = fanchart_data["quantiles"]
    
    # 예측 날짜 생성 (간단한 예시)
    last_date = price_history.index[-1] if not price_history.empty else pd.Timestamp.now()
    
    if horizon == "1M":
        future_dates = pd.date_range(start=last_date, periods=22, freq='B')
    elif horizon == "3M":
        future_dates = pd.date_range(start=last_date, periods=65, freq='B')
    elif horizon == "6M":
        future_dates = pd.date_range(start=last_date, periods=130, freq='B')
    else:  # 12M
        future_dates = pd.date_range(start=last_date, periods=252, freq='B')
    
    # 분위별 가격 계산
    price_paths = {}
    for q_key, return_val in quantiles.items():
        price_path = current_price * np.exp(return_val[0])
        price_paths[q_key] = [current_price, price_path]
    
    # 팬차트 영역 그리기 (밴드)
    bands = [
        ("q05", "q95", "rgba(59, 130, 246, 0.1)", "90% 구간"),
        ("q10", "q90", "rgba(59, 130, 246, 0.2)", "80% 구간"),
        ("q25", "q75", "rgba(59, 130, 246, 0.3)", "50% 구간")
    ]
    
    x_fan = [last_date, future_dates[-1]]
    
    for lower, upper, color, name in bands:
        if lower in price_paths and upper in price_paths:
            fig.add_trace(go.Scatter(
                x=x_fan + x_fan[::-1],
                y=price_paths[lower] + price_paths[upper][::-1],
                fill='toself',
                fillcolor=color,
                line=dict(width=0),
                name=name,
                hoverinfo='skip'
            ))
    
    # 중앙값 경로
    if "q50" in price_paths:
        fig.add_trace(go.Scatter(
            x=x_fan,
            y=price_paths["q50"],
            mode='lines',
            name='중앙 예측',
            line=dict(color=COLOR_PALETTE["primary"], width=3, dash='dash')
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"S&P 500 {horizon} 팬 차트",
        xaxis_title="날짜",
        yaxis_title="S&P 500 지수",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_uncertainty_gauge(
    iqr_width: float,
    historical_avg: float,
    title: str = "불확실성 게이지"
) -> None:
    """불확실성 게이지 렌더링
    
    Args:
        iqr_width: 현재 IQR 폭
        historical_avg: 역사적 평균 IQR
        title: 게이지 제목
    """
    ratio = iqr_width / historical_avg if historical_avg > 0 else 1.0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 1.0},
        gauge={
            'axis': {'range': [0, 2]},
            'bar': {'color': COLOR_PALETTE["primary"]},
            'steps': [
                {'range': [0, 0.8], 'color': COLOR_PALETTE["positive"] + "40"},
                {'range': [0.8, 1.2], 'color': COLOR_PALETTE["neutral"] + "40"},
                {'range': [1.2, 2], 'color': COLOR_PALETTE["negative"] + "40"}
            ],
            'threshold': {
                'line': {'color': COLOR_PALETTE["text"], 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_contribution_chart(
    contributions: pd.DataFrame,
    top_n: int = 5,
    chart_type: str = "bar"
) -> None:
    """기여도 차트 렌더링
    
    Args:
        contributions: 기여도 데이터
        top_n: 상위 N개 표시
        chart_type: 차트 타입 ("bar" 또는 "waterfall")
    """
    # 상위 기여 요인 선택
    top_contrib = contributions.nlargest(top_n, "abs_contribution")
    
    if chart_type == "bar":
        fig = go.Figure()
        
        # 색상 설정
        colors = [
            COLOR_PALETTE["positive"] if x > 0 else COLOR_PALETTE["negative"]
            for x in top_contrib["contribution"]
        ]
        
        fig.add_trace(go.Bar(
            x=top_contrib["contribution"],
            y=top_contrib["feature"],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{x:.3f}" for x in top_contrib["contribution"]],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="주요 기여 요인",
            xaxis_title="기여도",
            yaxis_title="",
            height=300,
            margin=dict(l=100, r=20, t=40, b=40)
        )
        
    else:  # waterfall
        fig = go.Figure(go.Waterfall(
            name="기여도",
            orientation="v",
            x=top_contrib["feature"],
            textposition="outside",
            text=[f"{x:.3f}" for x in top_contrib["contribution"]],
            y=top_contrib["contribution"],
            connector={"line": {"color": COLOR_PALETTE["neutral"]}}
        ))
        
        fig.update_layout(
            title="기여도 폭포차트",
            showlegend=False,
            height=400
        )
    
    st.plotly_chart(fig, use_container_width=True)


def render_scenario_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "median_change"
) -> None:
    """시나리오 비교 히트맵 렌더링
    
    Args:
        comparison_df: 시나리오 비교 데이터
        metric: 표시할 메트릭
    """
    fig = px.imshow(
        comparison_df,
        labels=dict(x="예측 기간", y="시나리오", color=metric),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto"
    )
    
    # 값 표시
    for i in range(len(comparison_df.index)):
        for j in range(len(comparison_df.columns)):
            value = comparison_df.iloc[i, j]
            if not pd.isna(value):
                fig.add_annotation(
                    text=f"{value:.1%}",
                    x=j,
                    y=i,
                    showarrow=False,
                    font=dict(color="white" if abs(value) > 0.02 else "black")
                )
    
    fig.update_layout(
        title="시나리오별 예상 수익률 변화",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_auto_summary(
    changes: List[Dict[str, Any]],
    max_items: int = 5
) -> None:
    """자동 요약 텍스트 렌더링
    
    Args:
        changes: 주요 변화 리스트
        max_items: 최대 표시 항목 수
    """
    st.markdown("### 주요 변화")
    
    if not changes:
        st.info("현재 특별한 변화가 감지되지 않았습니다.")
        return
    
    # 상위 변화 선택
    top_changes = sorted(changes, key=lambda x: abs(x.get("magnitude", 0)), reverse=True)[:max_items]
    
    for change in top_changes:
        indicator = change.get("indicator", "")
        direction = change.get("direction", "")
        magnitude = change.get("magnitude", 0)
        context = change.get("context", "")
        
        # 방향/색상(이모지 제거)
        if direction == "up":
            color = COLOR_PALETTE["positive"]
            dir_text = "상승"
        elif direction == "down":
            color = COLOR_PALETTE["negative"]
            dir_text = "하락"
        else:
            color = COLOR_PALETTE["neutral"]
            dir_text = "변화"
        
        # 변화 설명 생성
        change_text = f"**{indicator}**가 {abs(magnitude):.1f}% {dir_text}"
        
        if context:
            change_text += f" ({context})"
        
        st.markdown(
            f'<p style="color: {color};">{change_text}</p>',
            unsafe_allow_html=True
        )


def render_candlestick_chart(
    df: pd.DataFrame,
    title: str = "S&P 500 실시간 차트",
    height: int = 600,
    show_volume: bool = True,
    show_sma: bool = True,
    show_bb: bool = False,
) -> None:
    """캔들스틱 차트 렌더링

    Args:
        df: OHLCV 데이터프레임 (컬럼: Open, High, Low, Close, Volume)
        title: 차트 제목
        height: 차트 높이
        show_volume: 거래량 표시 여부
        show_sma: 이동평균선 표시 여부
        show_bb: 볼린저 밴드 표시 여부
    """
    if df.empty:
        st.warning("데이터가 없습니다.")
        return

    # 서브플롯 생성 (가격 + 거래량)
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(title, "거래량") if show_volume else (title,)
    )

    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
            increasing_line_color=COLOR_PALETTE["positive"],
            decreasing_line_color=COLOR_PALETTE["negative"],
        ),
        row=1, col=1
    )

    # 이동평균선
    if show_sma:
        if "SMA_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_20"],
                    mode="lines",
                    name="SMA 20",
                    line=dict(color="#f59e0b", width=1),
                ),
                row=1, col=1
            )
        if "SMA_50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_50"],
                    mode="lines",
                    name="SMA 50",
                    line=dict(color="#3b82f6", width=1),
                ),
                row=1, col=1
            )

    # 볼린저 밴드
    if show_bb and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                mode="lines",
                name="BB 상단",
                line=dict(color="#94a3b8", width=1, dash="dash"),
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                mode="lines",
                name="BB 하단",
                line=dict(color="#94a3b8", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.1)",
            ),
            row=1, col=1
        )

    # 거래량
    if show_volume:
        colors = [COLOR_PALETTE["positive"] if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else COLOR_PALETTE["negative"] for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="거래량",
                marker_color=colors,
                showlegend=False,
            ),
            row=2, col=1
        )

    # 레이아웃
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(title_text="시간", row=rows, col=1)
    fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="거래량", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_technical_indicators(df: pd.DataFrame) -> None:
    """기술적 지표 차트 렌더링 (RSI, MACD)

    Args:
        df: 기술적 지표가 포함된 데이터프레임
    """
    if df.empty:
        return

    # RSI와 MACD 서브플롯
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("RSI (Relative Strength Index)", "MACD"),
        row_heights=[0.5, 0.5]
    )

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color=COLOR_PALETTE["primary"], width=2),
            ),
            row=1, col=1
        )
        # 과매수/과매도 라인
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도", row=1, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color=COLOR_PALETTE["primary"], width=2),
            ),
            row=2, col=1
        )
        if "MACD_Signal" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD_Signal"],
                    mode="lines",
                    name="Signal",
                    line=dict(color=COLOR_PALETTE["warning"], width=2),
                ),
                row=2, col=1
            )
        if "MACD_Hist" in df.columns:
            colors = [COLOR_PALETTE["positive"] if v >= 0 else COLOR_PALETTE["negative"]
                      for v in df["MACD_Hist"]]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["MACD_Hist"],
                    name="Histogram",
                    marker_color=colors,
                ),
                row=2, col=1
            )

    # 레이아웃
    fig.update_layout(
        height=500,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
    )

    fig.update_yaxes(title_text="RSI", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_xaxes(title_text="시간", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def create_dashboard_layout() -> Tuple[Any, Any, Any]:
    """대시보드 레이아웃 생성

    Returns:
        tab_a, tab_b, tab_c 컨테이너
    """
    st.set_page_config(
        page_title="S&P 리스크·신호 집계판",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 헤더
    st.title("S&P 500 리스크·신호 집계판 + 확률 콘 대시보드")

    # 최종 업데이트 시간
    st.caption(f"최종 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 탭 생성
    tab_a, tab_b, tab_c = st.tabs(["매크로 시황 요약", "S&P 500 팬 차트", "실시간 차트"])

    return tab_a, tab_b, tab_c
