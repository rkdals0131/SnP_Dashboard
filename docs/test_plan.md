SNP 대시보드 — 기능별 실행·검증 계획 v0.4.1

본 문서는 src 기반 CLI를 실제로 실행하여 데이터 수집→정합·피처→학습→검증→앱 구동까지 점검하는 테스트 계획입니다. 각 단계별로 실행 명령, 기대 산출물, 확인 포인트, 실패 원인과 대응을 정리했습니다.

## 사전 준비

- Python/가상환경: `python3 -m venv .venv && source .venv/bin/activate`
- 설치(개발옵션; zsh는 따옴표 필수): `uv pip install -e ".[dev]"`
  - 또는 최소 설치: `uv pip install -e .` (이 경우 `PYTHONPATH=$PWD`로 임포트 보정 가능)
- 환경변수: `.env`에 `FRED_API_KEY=...` 존재 확인
- 네트워크: FRED API 호출 필요(캐시 저장됨)

## 빠른 E2E 스모크 테스트 (현실 크기·저부하)

1) 데이터 수집(필수 핵심 시리즈만 수집)
- `python -m src.cli ingest --source fred --series DGS10,DGS3MO,VIXCLS,SP500 --start 2010-01-01`
- 기대 산출물: `data/raw/fred/{DGS10,DGS3MO,VIXCLS,SP500}.parquet`
- 콘솔 확인: “수집 결과” 표, 각 시리즈 관측치 개수/범위 표시, 완료 메시지

2) 피처 생성(정합+빈티지 지연 시뮬+피처·타깃 산출)
- `python -m src.cli features --calendar-start 2010-01-01`
- 기대 산출물:
  - `data/processed/master_timeseries.parquet`
  - `data/features/features.parquet`
  - `data/targets/targets.parquet`
- 콘솔 확인: “마스터 패널 구축: (행,열)”, “피처 생성: (행,열)”, “타깃 생성: (행,열)”

3) 모델 학습(1/3/6/12M, 기본 분위수)
- `python -m src.cli train --horizons 1M,3M,6M,12M --alpha 1.0`
- 기대 산출물: `models/<snapshot>/models.pkl`
- 콘솔 확인: 스냅샷 ID, 각 기간의 학습 완료 표

4) 롤링 검증(최근 스냅샷 자동 선택)
- `python -m src.cli validate --train-window 60 --test-window 12 --step-size 1`
- 기대 산출물: `evaluations/<snapshot>/validation_report.json`
- 콘솔 확인: pinball_loss, crps, coverage_50/80 표

5) 대시보드 실행(수동 확인)
- `python -m src.cli app --port 8501`
- 기대 동작: http://localhost:8501 접속 시 탭 A(신호 집계판), 탭 B(팬 차트) 표시

## 기능별 상세 테스트 시나리오

### 1. 데이터 수집 — `ingest`
- 명령 예시
  - 기본(전체 기본 시리즈): `python -m src.cli ingest`
  - 선택 시리즈: `python -m src.cli ingest --series DGS10,DGS3MO,VIXCLS,SP500 --start 2000-01-01`
- 확인 포인트
  - `data/raw/fred/*.parquet` 생성 및 비어있지 않음
  - 콘솔 “수집 결과” 표에 각 시리즈 시작/종료일이 요청 범위와 일치
- 실패·대응
  - API 키 누락: `.env`/`FRED_API_KEY` 설정
  - 네트워크 오류: 재실행 시 캐시가 저장되므로 부분 성공 후 반복 가능

### 2. 피처 생성 — `features`
- 명령 예시: `python -m src.cli features --calendar-start 2000-01-01`
- 처리 내용
  - 월말 캘린더(ME) 생성 → 마스터 패널(asof 정렬) → 발표 지연 규칙 적용 → 피처/브레드스/MacroScore/레짐 배지 → 타깃 생성
- 확인 포인트
  - `data/processed/master_timeseries.parquet` 존재, 행 인덱스가 월말(`ME`) 정렬
  - `data/features/features.parquet` 존재, 대표 컬럼
    - `*_d1`, `*_d2`, `*_pctscore`, `TERM_SPREAD`, `MacroBreadth_*`, `MacroScore`
    - 레짐 더미: `Regime_*_*` 컬럼이 0/1(float)이며, 원본 문자열 `Regime_*` 컬럼은 없음
  - `data/targets/targets.parquet`에 `return_1M/3M/6M/12M` 존재
- 실패·대응
  - 원천 데이터 없음: `ingest` 선행 필요
  - 빈티지/정합 에러: `data/raw/fred/*.parquet` 스키마 확인(index=date, columns=series_id/value/…)

### 3. 모델 학습 — `train`
- 명령 예시
  - 기본: `python -m src.cli train`
  - 커스텀 분위수: `python -m src.cli train --quantiles 0.1,0.5,0.9 --alpha 0.5`
- 확인 포인트
  - 콘솔: `학습 데이터: X (행,열), y (행,열)` 출력
  - 스냅샷 디렉터리: `models/<YYYYMMDD_HHMMSS>/models.pkl` 생성
  - 로그에 각 분위 학습 완료 메시지 존재
- 실패·대응
  - 학습 데이터 부족(경고): 캘린더 시작일 조정 또는 더 긴 기간 수집
  - NaN 에러: 피처 생성 단계에서 결측이 많을 수 있으니 `features` 재실행 확인

### 4. 롤링 검증 — `validate`
- 명령 예시
  - 최근 스냅샷 자동: `python -m src.cli validate --train-window 60 --test-window 12 --step-size 1`
  - 특정 스냅샷: `python -m src.cli validate --snapshot models/2025.../models.pkl`
  - 기간 제한: `--start-date 2015-01-01 --end-date 2022-12-31`
- 확인 포인트
  - `evaluations/<snapshot>/validation_report.json` 생성
  - 콘솔 표에 `pinball_loss, crps, coverage_50, coverage_80`가 기간별로 표시
  - 윈도 자동 보정(데이터 길이에 맞춰 최소치 적용) 동작
- 실패·대응
  - 데이터 길이 부족: `train-window/test-window` 축소 또는 수집 기간 확대
  - 스냅샷 경로 불일치: `--snapshot`에 파일/디렉터리/ID 모두 지원

### 5. 대시보드 — `app`
- 명령 예시: `python -m src.cli app --port 8501 --debug`
- 확인 포인트(수동)
  - 탭 A: 합성 점수, 브레드스, 레짐 배지, 자동요약
  - 탭 B: 가격+팬 차트, 폭 게이지, 비교 토글, 기여도 카드(단순화)
  - 시나리오 슬라이더 조작 시 팬 차트 재계산(2초 이내)
- 실패·대응
  - 포트 충돌: `--port` 변경
  - 데이터/모델 없음: `features`, `train` 선행

### 6. 상태 점검 — `status`
- 명령 예시: `python -m src.cli status`
- 확인 포인트
  - 원천/마스터/피처/타깃/모델 상태 요약 표
  - 최근 모델 스냅샷 5개까지 표시(크기/수정시각)

## 추가 점검 항목(권장)

- 예측 샘플링(파이썬 REPL)
  - 마지막 행 특성으로 팬차트 예측: 모델·피처 로드 후 `predict_fanchart` 호출
  - 예시
    - `python - <<'PY'
import pandas as pd
from pathlib import Path
from src.models_quantile import load_models, predict_fanchart
features = pd.read_parquet('data/features/features.parquet')
last = features.iloc[[-1]]
models = load_models(sorted(Path('models').glob('*/models.pkl'))[-1])
fc = predict_fanchart(models, last, vix_current=float(last.get('VIXCLS', pd.Series([20])).iloc[0]))
print({k: {q: float(v[0]) for q,v in v['quantiles'].items()} for k,v in fc.items()})
PY`
- 레짐 더미 검증
  - `Regime_*_*` 컬럼이 존재하고 0/1(float)인지 확인(문자열 컬럼 미포함)
- 빈티지 지연 검증
  - `align_vintage.PUBLICATION_RULES`에 따른 월간/분기 시프트가 적용됐는지(월말 기준)
- 퍼센타일 스코어 검증
  - 월별 데이터에서 10년 창 기준 `_pctscore ∈ [-1,1]` 범위, 최근 값 상승/하락 방향성 직관 점검

## 문제 해결 가이드

- zsh extras 설치 오류: `uv pip install -e ".[dev]"`처럼 따옴표로 감싸기
- 임포트 오류: `uv pip install -e .` 설치 또는 `PYTHONPATH=$PWD`로 보정
- pytest 플러그인 충돌: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q -o addopts=`
- 네트워크/키 오류: `.env` 키 확인, 재시도(캐시 저장)

## 산출물 체크리스트

- 데이터: `data/raw/fred/*.parquet`
- 정합/빈티지: `data/processed/master_timeseries.parquet`
- 피처: `data/features/features.parquet`
- 타깃: `data/targets/targets.parquet`
- 모델: `models/<snapshot>/models.pkl`
- 검증: `evaluations/<snapshot>/validation_report.json`

끝.

