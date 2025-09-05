# System Component Design & Development TODOs v0.4.1

문서 상태: 제안 초안(아키텍처·PRD 정합 확인용)
연관 문서: prd.md, dev-architecture.md

## 1) 목표 정렬(PRD ↔ 아키텍처)

- 탭 A(신호 집계판): 표준화된 지표의 레벨/모멘텀/가속도, 합성 점수, 브레드스, 레짐 배지, 자동 요약을 30~60초 내 파악 가능하게 제공.
- 탭 B(팬 차트): 1·3·6·12M 조건부분포를 분위로 제시, 폭 게이지, 과거 예측 vs 실현 비교, 시나리오 슬라이더와 단순 기여도(계수×Δ특징).
- Lean 제약 준수: ALFRED 보류(발표 지연 규칙 기반 빈티지 시뮬), Typer 단일 CLI, 유닛/통합 테스트 집중, Streamlit 단일 앱.

## 2) 아키텍처 → 컴포넌트 매핑

- 수집층(`src/data_sources.py`): FRED·BLS·BEA 등 공공 API 어댑터. 재시도/백오프, Parquet 저장.
- 정합층(`src/align_vintage.py`): 발표 지연 규칙 적용 asof 조인, 주/월/일 혼합 정렬, 결측 정책.
- 피처층(`src/features.py`): 레벨, Δ, Δ², 10년 분위 스케일, 브레드스/MacroScore, 레짐 배지.
- 타깃/수익률(`src/targets.py`): 1·3·6·12M 로그수익률 생성, 월말 기준 리샘플.
- 모델층(`src/models_quantile.py`): 각 h,q 분위회귀 학습/추론, 단기폭 VIX 앵커, 거시축 블렌드.
- 시나리오(`src/scenarios.py`): 입력 피처벡터에 Δ 쇼크 적용, 중앙·폭 재계산, 단순 기여도.
- 시각화(`src/viz.py`): 카드/브레드스/배지/팬 차트/폭 게이지/기여도 바 컴포넌트.
- 앱(`app/streamlit_app.py`): 탭 A/B UI, 시나리오 슬라이더, 비교 토글, 캐시.
- CLI(`src/cli.py`): ingest → features → train → validate → app 실행 파이프라인.
- 검증(`src/validation.py`): 롤링 OOS, 핀볼·CRPS, 캘리브레이션, 산출물 저장.

## 3) 데이터 계약·경로·스키마

- 저장 경로
  - `data/raw/<source>/<series>.parquet` — 원천(컬럼: date, series_id, value, realtime_start, realtime_end)
  - `data/processed/master_timeseries.parquet` — asof·정합 후 마스터 패널(컬럼: date, <series...>)
  - `data/features/features.parquet` — 스케일·Δ·Δ²·분위·배지 포함
  - `data/targets/targets.parquet` — 수익률 타깃(h=1M/3M/6M/12M)
  - `models/<snapshot>/quantile_*.pkl` — 학습 결과(직렬화)
  - `evaluations/<YYYYMM>/*` — 검증 산출물(metrics.csv, calibration.csv 등)
- DataSource 인터페이스(요지)
  - 입력: series(list[str]), start(date), end(date)
  - 출력: DataFrame(index=date, columns=[series_id, value, realtime_start, realtime_end])
- 피처 패널(예)
  - 인덱스: date(月末 정렬)
  - 컬럼 예: `VIX`, `VIX_d1`, `VIX_d2`, `VIX_pctscore`, `TERM_pctscore`, `MacroScore`, `MacroBreadth`, `Regime_FCI`, …

## 4) 발표 지연 규칙(Lean 빈티지 시뮬)

- 원칙: 지표별 공개/유통 관행을 근사하여 t 시점에 사용 가능한 정보만 사용.
- 예시 규칙(초안; `align_vintage.py`에 매핑):
  - 일간(VIX/UST/WTI/SPX): T+1 영업일
  - 주간(신규실업 등): 공표 주의 금요일 T+1 영업일
  - 월간(CPI/PCE/UNRATE/BAA 스프레드): 다음 달 3 영업일
  - 분기(GDP): 다음 분기 20 영업일
- 구현: `asof(date - lag_by_rule(series_id))`로 좌측조인, forward-fill 금지(결측 명시 플래그).

## 5) 컴포넌트 설계 상세

### 5.1 `src/data_sources.py`

- 책임: 공공 API 호출 래핑, 재시도/백오프, 파라미터 검증, 로컬 Parquet 저장.
- 주요 함수
  - `fetch_fred(series: list[str], start: date, end: date) -> pd.DataFrame`
  - `persist_parquet(df, path) -> None`
  - `load_parquet(path) -> pd.DataFrame`
- 입출력
  - 입력: 시리즈 목록, 기간
  - 출력: 시계열 DF 저장 및 반환
- 테스트 포인트: 재시도 동작, 스키마 일치, 누락/결측 처리.

### 5.2 `src/align_vintage.py`

- 책임: 원천 단위 시계열을 공통 캘린더(月末)로 정렬, 발표 지연 규칙 기반 asof 조인.
- 주요 함수
  - `build_master_panel(raw_paths: dict[str,str], calendar: pd.DatetimeIndex) -> pd.DataFrame`
  - `apply_publication_delays(panel: pd.DataFrame, rules: dict[str, str]) -> pd.DataFrame`
- 정책: forward-fill 금지, 결측은 `*_is_avail` 플래그로 표시하여 모델 입력에서 마스킹.

### 5.3 `src/features.py`

- 책임: 레벨·Δ·Δ²·분위 스케일, 카테고리 합성, MacroScore·MacroBreadth, 레짐 배지 생성.
- 주요 함수
  - `make_deltas(df, cols: list[str]) -> pd.DataFrame`
  - `percentile_score_10y(df, cols) -> pd.DataFrame`  # [-1,+1] 스케일
  - `compute_macro_breadth(df, rules) -> pd.Series`   # 상위 50% 비중 및 Δ
  - `compute_macro_score(df, weights) -> pd.Series`   # 0.5/0.35/0.15 기본
  - `assign_regimes(df) -> pd.DataFrame`              # FCI/TERM/INFL 모멘텀 등

### 5.4 `src/targets.py`

- 책임: SPX 로그수익률 타깃 r_{t→t+h} 산출(h=1/3/6/12M), 월말 리샘플·정렬.
- 주요 함수
  - `compute_forward_returns(price_df, horizons=["1M","3M","6M","12M"]) -> pd.DataFrame`

### 5.5 `src/models_quantile.py`

- 책임: 분위회귀 학습/추론, 단기폭 VIX 앵커, 거시축 블렌드.
- 주요 함수
  - `train_models(X, y, horizons, quantiles, config) -> dict`
  - `predict_fanchart(models, X_t, anchors) -> dict`  # DTO 반환
- 블렌드(초안)
  - 단기(1M): 폭(q90-q10) = α·VIX_anch + (1-α)·QR 폭, α∈[0.3,0.7]
  - 중기(3~12M): QR 중심, 일부 거시축 폭 가중(NFCI 등)
- 저장: `models/<snapshot>/qreg_{h}_{q}.pkl`

### 5.6 `src/scenarios.py`

- 책임: 피처 벡터에 사용자 쇼크 적용(예: ΔVIX=+5, TERM=-20bp), 중앙·폭 재계산.
- 주요 함수
  - `apply_shocks(X_t, shocks: dict[str,float]) -> pd.Series`
  - `compute_contributions(beta, dX) -> list[dict]`  # 계수×Δ특징량 Top-N

### 5.7 `src/viz.py`

- 책임: 탭 A 카드/바/배지, 탭 B 팬 차트/게이지/기여도 바 렌더링.
- 주요 함수
  - `render_tab_a(state) -> None`
  - `render_tab_b(fanchart, compare, drivers) -> None`
  - `make_auto_summary(changes) -> str`  # 변화 Top-N 문장화

### 5.8 `app/streamlit_app.py`

- 책임: 단일 앱, 탭 A/B, 시나리오 슬라이더, 비교 토글, 캐시.
- 주요 섹션
  - 사이드바: 시나리오 입력(10Y, 3M, VIX, WTI, NFCI, CPI surprise)
  - 탭 A: 합성 점수, 브레드스, 배지, 카테고리 카드, 자동요약
  - 탭 B: 가격+팬 차트, 폭 게이지, 비교 토글, 기여도 카드

### 5.9 `src/validation.py`

- 책임: 롤링 OOS, 핀볼·CRPS, 캘리브레이션, 리포트 산출.
- 산출물
  - `metrics.csv(date,h,metric,value,model,baseline)`
  - `calibration.csv(date,h,ci,hit)`
  - `backtest_fanchart.parquet`

## 6) CLI 명령(단일 엔트리)

- `ingest --source fred --series DGS10,DGS3MO,VIXCLS,SP500 --start 2000-01-01`
- `features`  # 정합→피처→타깃 일괄 빌드
- `train --snapshot YYYYMMDD --horizons 1M,3M,6M,12M`
- `validate --snapshot YYYYMMDD`
- `app`  # Streamlit 실행

## 7) 개발 순서·마일스톤

- M1 PoC(우선): FRED 중심(UST 곡선, VIX, WTI, SPX), 카드 6종, 1M 팬 차트, 시나리오 3종, 검증 요약
  - 1) 데이터 어댑터(FRED)·저장
  - 2) 발표 지연 규칙·정합 패널
  - 3) 피처(Δ/Δ²/분위)·MacroBreadth/Score
  - 4) 타깃(1M)·학습·앵커 블렌드
  - 5) Streamlit 탭 A/B 최소 UI
  - 6) 검증 파이프라인(핵심 지표)
- M2 확장: 3/6/12M 수평, 기여도 카드, 검증 히스토리
- M3 안정화: 빈티지 백테스트 고도화, 알림/내보내기

## 8) 평행 개발 가능한 작업 묶음

- 수집 어댑터: FRED, BLS/BEA 스텁 병렬
- 정합·지연 규칙 vs 피처 엔지니어링 분리
- 모델 학습 래퍼 vs 검증 파이프라인 독립
- Streamlit 탭 A UI vs 탭 B UI 분리
- 시나리오 엔진 vs 기여도 카드 분리
- CLI 래핑·Makefile vs 내부 모듈 병렬

## 9) 탭 A 계산 정의(초안)

- 분위 스코어: 10년 창으로 백분위 p ∈ [0,1], S = 2p-1 ∈ [-1,1]
- 카테고리 합성: MacroCategory = wL·Level + wM·Δz + wA·Δ²z (기본 0.5/0.35/0.15)
- MacroScore: 카테고리 동일가중 평균
- MacroBreadth: 상위 50% 이상(또는 상위 20%)인 지표 비중; ΔBreadth는 전월 대비 변화
- 레짐 배지 예시
  - 금융여건: NFCI 분위 > 0.6 → 긴축, < 0.4 → 완화, 그 외 중립
  - 경기모멘텀: TERM_d1<0 & ISM_new_orders_d1<0 → 둔화, 반대는 확장(지표 대체 허용)
  - 물가모멘텀: CPI_d2>0 → 상승, <0 → 하락

## 10) 탭 B 모델링 정의(초안)

- 타깃: r_{t→t+h} = log(S_{t+h}/S_t), h∈{1M,3M,6M,12M}
- 특징: 거시·시장 축 레벨/Δ/Δ²·분위 + 레짐 배지(One-hot)
- 학습: scikit-learn QuantileRegressor 또는 statsmodels QuantReg
- 폭 앵커: 1M 폭은 VIX 기반 앵커와 블렌드
- DTO 형태(요지)
  - `{asof, horizon, quantiles{q05..q95}, drivers{center[], width[]}}`

## 11) 테스트 전략(요지)

- 유닛: 어댑터(fetch), asof/지연 규칙, Δ/Δ², 분위 계산, 폭 블렌드
- 통합: 수집→정합→피처→모델 사이클(작은 픽스처 데이터)
- UI: 수동 스냅샷 보관(E2E 제외)

## 12) TODOs & 수용 기준(컴포넌트별)

- Data Sources
  - 구현: FRED 어댑터(리트라이, Parquet 저장)
  - 수용: 지정 시리즈·기간 수집, 스키마·결측 플래그 포함
- Align/Vintage
  - 구현: 월말 캘린더 생성, asof 조인, 지연 규칙 매핑
  - 수용: forward-fill 금지, 가용/비가용 구분, 단위/빈도 혼합 정합
- Features
  - 구현: Δ·Δ², 분위 스케일, MacroBreadth/Score, 레짐 배지
  - 수용: 10년 창 고정, 카테고리 가중 기본값, 브레드스 Δ 탐지(±10%p)
- Targets
  - 구현: SPX 로그수익률 1/3/6/12M, 월말 정렬
  - 수용: 결측 처리 일관성, 라벨 누수 없음
- Models
  - 구현: QR 학습/직렬화, 예측, 폭 앵커 블렌드
  - 수용: 1M 팬 차트 생성, 80%·50% 구간 존재
- Scenarios
  - 구현: Δ 쇼크 적용, 기여도 Top-N(계수×Δ)
  - 수용: 슬라이더 입력 즉시 2초 내 재계산
- Viz
  - 구현: 카드/브레드스/배지/팬/게이지/기여도
  - 수용: 탭 A 30~60초 내 시황 구술 가능, 탭 B 비교 토글 표시
- Validation
  - 구현: 롤링 OOS, 핀볼/CRPS, 캘리브레이션·기준선 대비
  - 수용: 80% 구간 75~85%, 50% 구간 45~55% 목표 지표 산출 및 리포팅
- CLI/App
  - 구현: Typer 명령 세트, Streamlit 구동
  - 수용: Make 타깃과 연동, 파이프라인 전 단계 실행 가능

## 13) 시리즈·신호(시즌 1 제안)

- FRED: DGS10, DGS3MO, DGS2, VIXCLS, SP500, BAA, BAMLH0A0HYM2, DCOILWTICO
- 연준/기관: NFCI(시카고 연은)
- 노동/실물(확장 여지): UNRATE, PCE, GDP(후속)

## 14) 구성·관측성·품질

- 설정: `.env`/`pyproject.toml` 기반 경로/스냅샷 id, 난수시드
- 로깅: 구조화 로깅(JSONLines), 단계별 메트릭(레코드 수, 결측 비율)
- 품질: ruff, black, pytest, coverage, mypy(strict 선택)

## 15) 리스크·대응(요약)

- 데이터 지연/중단: 캐시·대체 소스 준비
- 레짐 변화: 롤링 재학습/윈도 탐색
- 과최적화: L1 규제, 기준선 대비 리포팅 의무화

