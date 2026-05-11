GLD/SIVR/UPRO 리밸런싱 매니저

GLD, SIVR, UPRO 3종목만 투자 대상으로 두고 일봉 Bollinger Band 중심의 매수·매도 참고 신호와 목표비중 리밸런싱 금액을 계산하는 Streamlit 대시보드입니다. 매크로 지표는 대폭 줄여 VIX, CNN Fear & Greed, Crypto Fear & Greed, UPRO, S&P500, NASDAQ만 참고합니다.

## 주요 기능

- 투자 대상: GLD, SIVR, UPRO
- 가격 데이터: yfinance 일봉 OHLCV 수집 및 `data/raw/yfinance/*.parquet` 캐시
- Core 리밸런싱: 기본 `GLD:SIVR:UPRO = 2:1:1`, 총 포트폴리오의 80%에 배분
- Tactical 슬리브: 기본 총 포트폴리오 20%, UPRO 3~7거래일 추세추종 신호에 따라 UPRO 또는 현금으로 관리
- Core 신호: 일봉 Bollinger Band 20D/2σ 기반 평균회귀 신호
- 참고 지표: VIX, CNN Fear & Greed, Alternative.me Crypto Fear & Greed, UPRO/S&P500/NASDAQ 단기 흐름
- 백테스트/신호검증: 정적 2:1:1, 볼린저 평균회귀, UPRO 추세추종 위성, SPY 보유, GLD/SPY/SIVR 비교

## 설치

- 요구 사항: Python 3.11+
- 권장 설치: `uv sync --extra dev`
- pip 사용 시: `pip install -r requirements.txt`

## 실행

- WSL/LAN 접근 권장 실행: `python -m src.cli app --host 0.0.0.0 --port 8501`
- 포트를 바꾸려면: `python -m src.cli app --host 0.0.0.0 --port 8601`
- 또는 직접 실행: `streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true`
- 브라우저: WSL 내부에서는 http://localhost:8501, 같은 네트워크에서는 호스트/WSL IP와 지정 포트로 접속합니다.

공유기 게이트웨이 `192.168.0.1`은 보통 접속 대상이 아니라 라우터 주소입니다. 대시보드는 `0.0.0.0`에 바인드하고, Windows 호스트 IP 또는 WSL IP의 `:8501` 포트로 접속하는 방식이 안전합니다.

## 대시보드 사용 흐름

1. 사이드바에서 GLD/SIVR/UPRO 현재 수량과 현금을 입력합니다.
2. 기본 목표비중은 core 80% 안에서 GLD 2, SIVR 1, UPRO 1입니다. 필요하면 ratio를 조정합니다.
3. UPRO tactical 슬리브는 기본 20%입니다. 단타 슬리브를 이미 보유 중이면 체크박스를 켭니다.
4. 리밸런싱 탭에서 목표금액, drift, 권장 매수·매도 수량을 확인합니다.
5. Bollinger 신호 탭에서 각 ETF의 평균회귀 매수/보유/축소 신호와 차트를 확인합니다.
6. 참고 지표 탭에서 UPRO, S&P500, NASDAQ, VIX, 공포탐욕지수를 확인합니다.
7. 백테스트/신호검증 탭에서 전략별 CAGR, MDD, 최악의 1개월/3개월/1년, 회복기간, 거래 횟수, 평균 turnover를 비교합니다.

## 신호 해석

Core 평균회귀 신호는 `%B` 기준입니다.

- `%B <= 0`: 하단 밴드 이탈, 매수 후보
- `0 < %B <= 0.2`: 매수 관심
- `0.2 < %B < 0.8`: 보유
- `0.8 <= %B < 1.0`: 익절/축소 관심
- `%B >= 1.0`: 상단 밴드 돌파, 축소/매도 후보

UPRO tactical 신호는 UPRO 2거래일 상단 밴드 돌파, ADX 기준 충족, S&P500/NASDAQ 20D SMA 상회, VIX 안정 조건을 함께 봅니다. UPRO가 5D EMA 또는 20D SMA를 하회하거나 VIX가 급등하면 청산 신호로 봅니다.

## 백테스트 비교군

- `2:1:1 월간 리밸런싱`: GLD 50%, UPRO 25%, SIVR 25%
- `2:1:1 분기 리밸런싱`: 같은 비중을 분기별로 조정
- `볼린저 평균회귀`: 하단 이탈 시 5%p 증액, 상단 돌파 시 5%p 감량, 최소 현금 5%
- `UPRO 추세추종 위성`: 코어 80% + UPRO 위성 20%, 상단 돌파/ADX/지수/VIX 확인 조건
- `평균회귀 + 200D 필터`: UPRO는 200일선 위에서만 하단 밴드 매수 허용
- `SPY 보유`: 단순 시장 벤치마크
- `GLD/SPY/SIVR 월간`: UPRO 대신 SPY를 넣은 비교군

신호검증 표는 GLD/SIVR/UPRO별로 상단 돌파, 하단 이탈, 하단 재진입, 추세추종 후보, 200D 위/아래 하단 이탈 이후 5/10/20/40거래일 평균 수익률과 승률을 보여줍니다.

## 주의

이 앱은 주문 자동화 도구가 아니라 리밸런싱 참고용 대시보드입니다. UPRO는 일일 3배 목표 레버리지 ETF라 변동성, 일일 리셋, 보유기간에 따라 기초지수 누적수익률의 3배와 달라질 수 있습니다.

## 테스트

- 전체 테스트: `uv run --extra dev pytest -q`
- 신규 포트폴리오 로직 테스트: `uv run --extra dev pytest tests/unit/test_portfolio_dashboard.py -q`
- 백테스트 로직 테스트: `uv run --extra dev pytest tests/unit/test_backtesting.py -q`
