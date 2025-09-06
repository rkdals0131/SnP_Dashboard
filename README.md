S&P 500 리스크·신호 집계판 (SNP Dashboard)

정합된 매크로·시장 지표로 신호를 집계하고, 분위 회귀(Quantile Regression) 기반의 확률 콘(팬 차트)으로 S&P 500의 단·중기 분포형 예측을 제공합니다. CLI로 데이터 수집→정합·피처→학습→검증→대시보드 실행까지 전체 파이프라인을 돌릴 수 있으며, Streamlit 대시보드에서 시각적으로 탐색할 수 있습니다.


## 무엇을 제공하나요?

- 데이터 파이프라인: FRED 등 공공 데이터 수집(캐시), 월말 빈티지 정렬, 발표 지연 규칙 적용
- 피처 엔지니어링: 1차·2차 차분, 백분위 스코어, TERM_SPREAD, Breadth, MacroScore, 레짐 배지 등
- 타깃 생성: S&P 500의 1M/3M/6M/12M 로그 수익률(선행) 타깃
- 모델링: 분위 회귀 모델(여러 분위) + 단기 VIX 앵커링 블렌드(안정성)
- 검증: 롤링 OOS 검증(pinball loss, CRPS 근사, 50/80% 커버리지)
- 대시보드: 매크로 시황 요약(합성 스코어·브레드스·레짐·지표 카드) + 팬 차트, 불확실성 게이지, 시나리오, 기여도


## 설치 및 환경 준비

- 요구 사항: Python 3.11+
- 가상환경(권장)
  - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
  - Windows (PowerShell): `py -3.11 -m venv .venv; .venv\\Scripts\\Activate.ps1`
- 설치
  - 표준: `pip install -r requirements.txt`
  - 개발(에디터블): `pip install -e .[dev]`
- 환경변수
  - 루트에 `.env` 생성: `FRED_API_KEY=...`
  - 키가 없으면 수집이 실패합니다(캐시가 있으면 재시도 시 빠름).


## 빠른 시작(Quickstart)

1) 데이터 수집(핵심 시리즈만 예시)
- `python -m src.cli ingest --source fred --series DGS10,DGS3MO,VIXCLS,SP500 --start 2010-01-01`
- 산출물: `data/raw/fred/{DGS10,DGS3MO,VIXCLS,SP500}.parquet`

2) 피처 생성(정합+빈티지 지연+피처/타깃)
- `python -m src.cli features --calendar-start 2010-01-01`
- 산출물: `data/processed/master_timeseries.parquet`, `data/features/features.parquet`, `data/targets/targets.parquet`

3) 모델 학습(1/3/6/12M, 기본 분위)
- `python -m src.cli train --horizons 1M,3M,6M,12M --alpha 1.0`
- 산출물: `models/<YYYYMMDD_HHMMSS>/models.pkl`

4) 롤링 검증(최근 스냅샷 자동)
- `python -m src.cli validate --train-window 60 --test-window 12 --step-size 1`
- 산출물: `evaluations/<snapshot>/validation_report.json`

5) 대시보드 실행
- `python -m src.cli app --port 8501`
- 브라우저: http://localhost:8501


## CLI 명령어와 옵션

아래 모든 명령은 `python -m src.cli <command> [options]` 형태로 실행합니다.

### ingest — 공공 데이터 수집 및 캐시
- 설명: FRED API에서 월별/일별 시계열을 수집해 Parquet로 저장합니다.
- 기본 시리즈: `DGS10, DGS3MO, DGS2, VIXCLS, SP500, DCOILWTICO, BAA, BAMLH0A0HYM2, UNRATE, CPIAUCSL, CPILFESL, NFCI`
- 옵션
  - `--source`: 데이터 소스(기본: `fred`)
  - `--series`: 수집할 시리즈(쉼표). 생략 시 기본 셋 사용
  - `--start`: 시작일(YYYY-MM-DD, 기본: `2000-01-01`)
  - `--end`: 종료일(미지정 시 오늘)
  - `--api-key` 또는 환경변수 `FRED_API_KEY`
- 산출물: `data/raw/fred/*.parquet`

### features — 정합·빈티지 처리·피처/타깃 생성
- 설명: 월말(ME) 캘린더 정렬 → 발표 지연 규칙 적용 → 피처 엔지니어링 → 타깃(선행 수익률) 생성
- 옵션
  - `--calendar-start`: 캘린더 시작일(기본: `2000-01-01`)
  - `--calendar-end`: 캘린더 종료일(미지정 시 오늘)
- 산출물
  - `data/processed/master_timeseries.parquet`
  - `data/features/features.parquet`
  - `data/targets/targets.parquet`

### train — 분위 회귀 모델 학습/스냅샷 저장
- 설명: 각 기간(horizon)과 분위수별 QuantileRegressor 학습 및 저장
- 옵션
  - `--horizons`: 학습할 기간(기본: `1M,3M,6M,12M`)
  - `--quantiles`: 분위수 목록(예: `0.1,0.5,0.9`; 생략 시 기본 `[0.05,0.10,0.25,0.50,0.75,0.90,0.95]`)
  - `--alpha`: L1 규제 강도(기본: `1.0`)
  - `--snapshot`: 스냅샷 ID(미지정 시 현재 시각으로 자동 생성)
- 산출물: `models/<snapshot>/models.pkl`

### validate — 롤링 OOS 검증 및 리포트
- 설명: 지정 윈도로 재학습/예측하여 pinball loss, CRPS, coverage 등을 집계
- 옵션
  - `--snapshot`: 평가할 스냅샷(파일/디렉터리/ID 모두 지원, 미지정 시 최신)
  - `--start-date`, `--end-date`: 검증 구간 제한(선택)
  - `--train-window`: 학습 윈도 크기(개월, 기본: `60`)
  - `--test-window`: 테스트 윈도 크기(개월, 기본: `12`)
  - `--step-size`: 스텝 크기(개월, 기본: `1`)
- 동작 참고
  - 데이터 길이에 맞춰 최소치로 자동 보정해 실행합니다.
  - 리포트 출력 경로는 `evaluations/<snapshot_id>/validation_report.json` 입니다.

### app — Streamlit 대시보드 실행
- 설명: 탭 A(매크로 시황 요약) + 탭 B(팬 차트/게이지/시나리오/기여도) 제공
- 옵션
  - `--port`: 포트(기본: `8501`)
  - `--debug`: Streamlit 로거 레벨 디버그(기본: `False`)
- 사용 팁
  - 사이드바에서 스냅샷과 시나리오 쇼크를 선택할 수 있습니다.
  - 모델이 비어있는 스냅샷인 경우 경고를 표시하고 안전하게 중단합니다.

### status — 현재 시스템 상태 요약
- 설명: 원천/마스터/피처/타깃/모델 보유 현황을 표로 출력
- 옵션: 없음


## 대시보드 구성 요소

- 탭 A — 매크로 시황 요약
  - MacroScore/모멘텀/가속도 타일, Breadth 바, 레짐 배지, 핵심 지표 카드, 자동 요약
- 탭 B — S&P 500 팬 차트
  - 과거 가격 + 분위 밴드(50/80/90%), 중앙 경로
  - 불확실성 게이지: IQR 폭 vs. 역사적 평균
  - 시나리오 쇼크(슬라이더/프리셋), 기여도(계수×Δ특성, 단순화)


## 산출물 경로 요약

- 원천 데이터: `data/raw/fred/*.parquet`
- 마스터 빈티지 패널: `data/processed/master_timeseries.parquet`
- 피처: `data/features/features.parquet`
- 타깃: `data/targets/targets.parquet`
- 모델 스냅샷: `models/<snapshot>/models.pkl`
- 검증 리포트: `evaluations/<snapshot>/validation_report.json`


## 문제 해결 가이드

- API 키 누락/네트워크 오류: `.env`의 `FRED_API_KEY` 확인 후 재시도(캐시 사용)
- 학습 데이터 부족 경고: 캘린더 시작일을 앞당기거나 수집 기간을 늘리세요. 초기에는 `--horizons 1M,3M`으로 축소 학습을 권장합니다.
- NaN으로 인한 학습 실패: `features` 재생성 또는 결측률 높은 지표 제외/보강을 고려하세요.
- 검증 실행 에러: 스냅샷 경로는 파일/디렉터리/ID 모두 지원합니다. 데이터 길이가 부족하면 윈도를 줄이세요.
- 대시보드 빈 화면: 스냅샷에 모델이 없으면 경고 후 반환합니다. 먼저 `train`을 성공시켜 주세요.


## 개발자 섹션(선택)

- 테스트 실행
  - 전체: `pytest`
  - 조용히: `pytest -q`
  - 커버리지 HTML: `pytest --cov=src --cov=app --cov-report=html --cov-report=term-missing:skip-covered`
- 경로 보정: `pip install -e .`를 사용하지 않았다면 `PYTHONPATH=$PWD`로 임포트를 보정할 수 있습니다.
- 주요 모듈
  - `src/cli.py`: CLI 엔트리, 파이프라인 명령들
  - `src/models_quantile.py`: 분위 회귀, VIX 앵커링, 팬 차트 생성
  - `src/validation.py`: 롤링 검증 로직, 리포트 생성
  - `src/features.py`: 피처 엔지니어링
  - `src/align_vintage.py`: 월말 캘린더/발표 지연 규칙
  - `app/streamlit_app.py`, `src/viz.py`: 대시보드 UI/시각화

