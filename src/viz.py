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
    "NFCI": "시카고 연은 금융여건지수. 0보다 크면 긴축, 작으면 완화.",
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


def create_dashboard_layout() -> Tuple[Any, Any]:
    """대시보드 레이아웃 생성
    
    Returns:
        tab_a, tab_b 컨테이너
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
    tab_a, tab_b = st.tabs(["매크로 시황 요약", "S&P 500 팬 차트"])
    
    return tab_a, tab_b
