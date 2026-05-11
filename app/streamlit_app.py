"""Streamlit dashboard for GLD/SIVR/UPRO rebalancing."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog


def _configure_structlog_for_streamlit() -> None:
    if getattr(_configure_structlog_for_streamlit, "_done", False):
        return
    structlog.configure(
        processors=[],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.ReturnLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configure_structlog_for_streamlit._done = True


_configure_structlog_for_streamlit()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting import (  # noqa: E402
    BacktestConfig,
    close_panel,
    cost_sensitivity_report,
    out_of_sample_report,
    parameter_sensitivity_report,
    regime_report,
    run_backtest_suite,
    signal_quality_report,
    stress_period_report,
    walk_forward_report,
)
from src.market_data import (  # noqa: E402
    DEFAULT_PRICE_TICKERS,
    TICKER_LABELS,
    fetch_cnn_fear_greed,
    fetch_crypto_fear_greed,
    fetch_yfinance_prices,
)
from src.portfolio import (  # noqa: E402
    BollingerConfig,
    classify_core_signal,
    classify_upro_tactical_signal,
    compute_bollinger_features,
    compute_rebalance_orders,
    latest_signal_row,
)

PORTFOLIO_TICKERS = ["GLD", "SIVR", "UPRO"]
REFERENCE_TICKERS = ["^GSPC", "^IXIC", "^VIX"]
TONE_COLORS = {
    "positive": "#16a34a",
    "negative": "#dc2626",
    "warning": "#d97706",
    "neutral": "#475569",
}


@st.cache_data(ttl=300, show_spinner=False)
def _load_market_data(start: date, end: date, use_cache: bool) -> dict[str, pd.DataFrame]:
    return fetch_yfinance_prices(DEFAULT_PRICE_TICKERS, start=start, end=end, use_cache=use_cache)


@st.cache_data(ttl=600, show_spinner=False)
def _load_sentiment() -> dict[str, dict[str, object]]:
    return {
        "cnn": fetch_cnn_fear_greed(),
        "crypto": fetch_crypto_fear_greed(),
    }


def _last_close(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty or "close" not in frame.columns:
        return float("nan")
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        return float("nan")
    return float(close.iloc[-1])


def _price_delta(frame: pd.DataFrame, periods: int = 5) -> float:
    if frame is None or frame.empty or "close" not in frame.columns:
        return float("nan")
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if len(close) <= periods:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def _format_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.0f}"


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


def _render_signal_badge(signal: dict[str, object]) -> None:
    tone = str(signal.get("tone", "neutral"))
    color = TONE_COLORS.get(tone, TONE_COLORS["neutral"])
    st.markdown(
        f"""
        <div style="border-left:4px solid {color};padding:0.6rem 0.8rem;background:#f8fafc">
          <strong>{signal.get("label", "-")}</strong><br>
          <span style="color:#475569;font-size:0.9rem">{signal.get("detail", "")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_bollinger_chart(
    ticker: str,
    features: pd.DataFrame,
    lookback: int = 180,
) -> None:
    if features.empty:
        st.info(f"{ticker} 차트 데이터가 없습니다.")
        return

    chart = features.dropna(subset=["close"]).iloc[-lookback:]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["upper"],
            name="Upper",
            line={"color": "#d97706", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["lower"],
            name="Lower",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.10)",
            line={"color": "#d97706", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["middle"],
            name="20D SMA",
            line={"color": "#64748b", "width": 1, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["close"],
            name="Close",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.update_layout(
        title=f"{ticker} Daily Bollinger Bands",
        height=360,
        margin={"l": 10, "r": 10, "t": 45, "b": 25},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_sentiment_card(title: str, payload: dict[str, object]) -> None:
    score = payload.get("score", np.nan)
    rating = str(payload.get("rating", ""))
    if not payload.get("ok", False) or pd.isna(score):
        st.metric(title, "N/A", help="외부 데이터 호출 실패 또는 응답 없음")
        return
    st.metric(title, f"{float(score):.0f}", rating)


def _build_feature_map(
    market_data: dict[str, pd.DataFrame],
    config: BollingerConfig,
) -> dict[str, pd.DataFrame]:
    features: dict[str, pd.DataFrame] = {}
    for ticker, frame in market_data.items():
        if frame is None or frame.empty:
            features[ticker] = pd.DataFrame()
            continue
        try:
            features[ticker] = compute_bollinger_features(frame, config)
        except Exception:
            features[ticker] = pd.DataFrame()
    return features


def _render_backtest_equity(equity: pd.DataFrame) -> None:
    if equity.empty:
        st.info("백테스트 지수화 수익률을 계산할 데이터가 부족합니다.")
        return
    fig = go.Figure()
    for column in equity.columns:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity[column],
                mode="lines",
                name=column,
            )
        )
    fig.update_layout(
        title="전략별 지수화 자산가치",
        height=420,
        margin={"l": 10, "r": 10, "t": 45, "b": 25},
        hovermode="x unified",
        yaxis_title="Initial = 1.0",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metric_table(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info("백테스트 성과 지표를 계산할 데이터가 부족합니다.")
        return
    fmt = {
        "CAGR": "{:.1%}",
        "MDD": "{:.1%}",
        "Vol": "{:.1%}",
        "Sharpe": "{:.2f}",
        "Worst 1M": "{:.1%}",
        "Worst 3M": "{:.1%}",
        "Worst 1Y": "{:.1%}",
        "Recovery Days": "{:.0f}",
        "Trades": "{:.0f}",
        "Avg Turnover": "{:.1%}",
    }
    st.dataframe(metrics.style.format(fmt, na_rep="-"), use_container_width=True, hide_index=True)


def _render_signal_quality(report: pd.DataFrame) -> None:
    if report.empty:
        st.info("신호 품질 검증에 필요한 데이터가 부족합니다.")
        return
    fmt = {
        "events": "{:.0f}",
        "5D_avg": "{:.1%}",
        "5D_win_rate": "{:.1%}",
        "10D_avg": "{:.1%}",
        "10D_win_rate": "{:.1%}",
        "20D_avg": "{:.1%}",
        "20D_win_rate": "{:.1%}",
        "40D_avg": "{:.1%}",
        "40D_win_rate": "{:.1%}",
    }
    st.dataframe(report.style.format(fmt, na_rep="-"), use_container_width=True, hide_index=True)


def _render_analysis_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("표시할 분석 결과가 없습니다.")
        return
    fmt = {
        "CAGR": "{:.1%}",
        "MDD": "{:.1%}",
        "Vol": "{:.1%}",
        "Sharpe": "{:.2f}",
        "Worst 1M": "{:.1%}",
        "Worst 3M": "{:.1%}",
        "Worst 1Y": "{:.1%}",
        "Recovery Days": "{:.0f}",
        "Trades": "{:.0f}",
        "Avg Turnover": "{:.1%}",
        "ann_return": "{:.1%}",
        "ann_vol": "{:.1%}",
        "hit_rate": "{:.1%}",
        "worst_day": "{:.1%}",
    }
    available_fmt = {key: value for key, value in fmt.items() if key in frame.columns}
    st.dataframe(
        frame.style.format(available_fmt, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="GLD/SIVR/UPRO 리밸런싱 매니저",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("GLD/SIVR/UPRO 리밸런싱 매니저")

    with st.sidebar:
        st.header("Portfolio")
        today = date.today()
        start = st.date_input("가격 시작일", value=date(2010, 1, 1))
        end = st.date_input("가격 종료일", value=today)
        use_cache = st.checkbox("가격 캐시 사용", value=True)

        st.subheader("현재 보유")
        quantities = {
            ticker: st.number_input(f"{ticker} 수량", min_value=0.0, value=0.0, step=1.0)
            for ticker in PORTFOLIO_TICKERS
        }
        cash = st.number_input("현금", min_value=0.0, value=0.0, step=100.0)

        st.subheader("목표 비중")
        ratio_gld = st.number_input("GLD core ratio", min_value=0.0, value=2.0, step=0.5)
        ratio_sivr = st.number_input("SIVR core ratio", min_value=0.0, value=1.0, step=0.5)
        ratio_upro = st.number_input("UPRO core ratio", min_value=0.0, value=1.0, step=0.5)
        tactical_weight = st.slider("UPRO 단타 슬리브", 0.0, 0.5, 0.20, 0.01)
        currently_tactical = st.checkbox("현재 UPRO 단타 슬리브 보유 중", value=False)

        st.subheader("신호 설정")
        bb_window = st.number_input("Bollinger window", min_value=5, max_value=80, value=20, step=1)
        bb_std = st.number_input("표준편차 배수", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
        adx_threshold = st.number_input(
            "UPRO 단타 ADX 기준",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
        )
        min_trade_value = st.number_input("최소 주문 금액", min_value=0.0, value=100.0, step=50.0)
        drift_tolerance = st.slider("리밸런싱 허용 drift", 0.0, 0.10, 0.01, 0.005)

        st.subheader("백테스트")
        initial_capital = st.number_input("초기 테스트 자산", min_value=1000.0, value=10_000.0)
        transaction_cost_bps = st.number_input("거래비용 bps", min_value=0.0, value=5.0)
        max_upro_weight = st.slider("백테스트 UPRO 최대비중", 0.10, 0.60, 0.35, 0.01)
        max_hold_days = st.number_input("위성 최대 보유일", min_value=1, max_value=60, value=7)
        oos_split = st.text_input("OOS split date", value="2021-01-01")

        if st.button("데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()

    if start >= end:
        st.error("가격 시작일은 종료일보다 이전이어야 합니다.")
        return

    with st.spinner("시장 데이터 로딩 중..."):
        market_data = _load_market_data(start, end, use_cache)
        sentiment = _load_sentiment()

    config = BollingerConfig(window=int(bb_window), num_std=float(bb_std))
    features = _build_feature_map(market_data, config)
    prices = {
        ticker: _last_close(market_data.get(ticker, pd.DataFrame()))
        for ticker in PORTFOLIO_TICKERS
    }

    tactical_signal = classify_upro_tactical_signal(
        features.get("UPRO", pd.DataFrame()),
        features.get("^GSPC", pd.DataFrame()),
        features.get("^IXIC", pd.DataFrame()),
        features.get("^VIX", pd.DataFrame()),
        sleeve_weight=tactical_weight,
        adx_threshold=float(adx_threshold),
    )
    tactical_active = tactical_signal["action"] == "ENTER" or (
        currently_tactical and tactical_signal["action"] != "EXIT"
    )

    try:
        orders = compute_rebalance_orders(
            quantities=quantities,
            cash=float(cash),
            prices=prices,
            core_ratio={"GLD": ratio_gld, "SIVR": ratio_sivr, "UPRO": ratio_upro},
            tactical_weight=float(tactical_weight),
            tactical_active=bool(tactical_active),
            min_trade_value=float(min_trade_value),
            drift_tolerance=float(drift_tolerance),
            transaction_cost_bps=float(transaction_cost_bps),
        )
    except ValueError as exc:
        st.error(str(exc))
        st.warning(
            "보유수량이 있는 종목의 가격 데이터가 복구된 뒤 "
            "리밸런싱 계산을 다시 실행하세요."
        )
        return

    total_value = float(orders["current_value"].sum())
    invested_value = float(
        orders.loc[orders["ticker"].isin(PORTFOLIO_TICKERS), "current_value"].sum()
    )

    overview_cols = st.columns(5)
    overview_cols[0].metric("총자산", _format_money(total_value))
    overview_cols[1].metric("투자금액", _format_money(invested_value))
    overview_cols[2].metric(
        "현금비중",
        _format_pct(float(cash) / total_value if total_value > 0 else np.nan),
    )
    overview_cols[3].metric("UPRO 단타", str(tactical_signal.get("label", "-")))
    overview_cols[4].metric("VIX", f"{_last_close(market_data.get('^VIX', pd.DataFrame())):.2f}")

    st.caption(
        "신호는 주문 자동화가 아닌 리밸런싱 참고용입니다. UPRO는 일일 3배 목표 상품이라 "
        "변동성과 보유기간에 따라 지수 누적수익률의 3배와 달라질 수 있습니다."
    )

    tab_rebalance, tab_signals, tab_reference, tab_backtest = st.tabs(
        ["리밸런싱", "Bollinger 신호", "참고 지표", "백테스트/신호검증"]
    )

    with tab_rebalance:
        st.subheader("권장 리밸런싱")
        display = orders.copy()
        money_cols = [
            "price",
            "current_value",
            "target_value",
            "drift_value",
            "trade_value",
            "post_value",
            "residual_drift",
            "cash_from_sells",
            "cash_used_for_buys",
            "estimated_fee",
            "post_trade_cash",
        ]
        for col in money_cols:
            if col in display.columns:
                display[col] = display[col].map(
                    lambda value: "-" if pd.isna(value) else f"${value:,.2f}"
                )
        for col in ["target_weight", "drift_pct", "post_weight"]:
            if col in display.columns:
                display[col] = display[col].map(lambda value: f"{value:.1%}")
        for col in ["quantity", "post_qty"]:
            if col in display.columns:
                display[col] = display[col].map(
                    lambda value: "-" if pd.isna(value) else f"{value:,.2f}"
                )
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("#### UPRO 단타 슬리브")
        st.info(str(tactical_signal.get("detail", "")))

    with tab_signals:
        signal_cols = st.columns(3)
        for idx, ticker in enumerate(PORTFOLIO_TICKERS):
            frame = features.get(ticker, pd.DataFrame())
            row = latest_signal_row(frame)
            signal = classify_core_signal(row)
            with signal_cols[idx]:
                price = prices.get(ticker, np.nan)
                st.metric(
                    ticker,
                    f"${price:,.2f}" if not pd.isna(price) else "-",
                    f"%B {float(row['pct_b']):.2f}" if row is not None else "-",
                )
                _render_signal_badge(signal)

        for ticker in PORTFOLIO_TICKERS:
            _render_bollinger_chart(ticker, features.get(ticker, pd.DataFrame()))

    with tab_reference:
        st.subheader("시장 참고 지표")
        ref_cols = st.columns(5)
        ref_cols[0].metric(
            "UPRO 5D",
            _format_pct(_price_delta(market_data.get("UPRO", pd.DataFrame()))),
        )
        ref_cols[1].metric(
            "S&P500 5D",
            _format_pct(_price_delta(market_data.get("^GSPC", pd.DataFrame()))),
        )
        ref_cols[2].metric(
            "NASDAQ 5D",
            _format_pct(_price_delta(market_data.get("^IXIC", pd.DataFrame()))),
        )
        with ref_cols[3]:
            _render_sentiment_card("CNN F&G", sentiment["cnn"])
        with ref_cols[4]:
            _render_sentiment_card("Crypto F&G", sentiment["crypto"])

        ref_table_rows = []
        for ticker in ["UPRO", *REFERENCE_TICKERS]:
            frame = features.get(ticker, pd.DataFrame())
            row = latest_signal_row(frame)
            ref_table_rows.append(
                {
                    "name": TICKER_LABELS.get(ticker, ticker),
                    "close": _last_close(market_data.get(ticker, pd.DataFrame())),
                    "5d_return": _price_delta(market_data.get(ticker, pd.DataFrame())),
                    "pct_b": float(row["pct_b"]) if row is not None else np.nan,
                    "adx": float(row.get("adx", np.nan)) if row is not None else np.nan,
                }
            )
        ref_df = pd.DataFrame(ref_table_rows)
        st.dataframe(
            ref_df.style.format(
                {
                    "close": "${:,.2f}",
                    "5d_return": "{:.1%}",
                    "pct_b": "{:.2f}",
                    "adx": "{:.1f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        for ticker in REFERENCE_TICKERS:
            _render_bollinger_chart(
                TICKER_LABELS.get(ticker, ticker),
                features.get(ticker, pd.DataFrame()),
            )

    with tab_backtest:
        st.subheader("코어-새틀라이트 자동 검증")
        st.caption(
            "목표는 최적 파라미터 탐색이 아니라 2:1:1 리밸런싱, 볼린저 평균회귀, "
            "UPRO 추세추종 위성이 어떤 손실 구조를 갖는지 비교하는 것입니다."
        )
        price_panel = close_panel(market_data)
        if price_panel.empty:
            st.warning("백테스트에 사용할 가격 패널이 없습니다.")
            return

        bt_config = BacktestConfig(
            initial_capital=float(initial_capital),
            core_weights=(0.50, 0.25, 0.25),
            rebalance_band=float(drift_tolerance),
            transaction_cost_bps=float(transaction_cost_bps),
            satellite_weight=float(tactical_weight),
            satellite_unit_weight=0.05,
            max_upro_weight=float(max_upro_weight),
            min_cash_weight=0.05,
            bb_window=int(bb_window),
            bb_std=float(bb_std),
            max_satellite_hold_days=int(max_hold_days),
        )
        bt_tabs = st.tabs(
            [
                "기본",
                "신호품질",
                "OOS",
                "워크포워드",
                "민감도/비용",
                "레짐/스트레스",
            ]
        )
        with bt_tabs[0]:
            metrics, equity, _ = run_backtest_suite(price_panel, bt_config)
            _render_metric_table(metrics)
            _render_backtest_equity(equity)
        with bt_tabs[1]:
            signal_report = signal_quality_report(price_panel, bt_config)
            _render_signal_quality(signal_report)
        with bt_tabs[2]:
            _render_analysis_table(out_of_sample_report(price_panel, oos_split, bt_config))
        with bt_tabs[3]:
            st.markdown("##### Rolling 5Y -> 1Y")
            _render_analysis_table(
                walk_forward_report(price_panel, bt_config, train_years=5, test_years=1)
            )
            st.markdown("##### Anchored -> 1Y")
            _render_analysis_table(
                walk_forward_report(
                    price_panel,
                    bt_config,
                    train_years=5,
                    test_years=1,
                    anchored=True,
                )
            )
        with bt_tabs[4]:
            st.markdown("##### UPRO 추세추종 파라미터 민감도")
            _render_analysis_table(parameter_sensitivity_report(price_panel, base_config=bt_config))
            st.markdown("##### 거래비용/슬리피지 민감도")
            _render_analysis_table(cost_sensitivity_report(price_panel, base_config=bt_config))
        with bt_tabs[5]:
            st.markdown("##### 레짐별 성과")
            _render_analysis_table(regime_report(price_panel, config=bt_config))
            st.markdown("##### 스트레스 구간")
            _render_analysis_table(stress_period_report(price_panel, bt_config))


if __name__ == "__main__":
    main()
