# dev-architecture — 모듈화·검증 통합 v0.4.1

문서 상태: 확정 초안
문서 소유자: 사용자(강민 김)
연관 문서: prd.md — S\&P 리스크·신호 집계판 + 확률 콘 대시보드 v0.4.1
전제: 배포 계획 0% → 로컬 우선, 단일 앱 중심

## v0.4.1 Lean 조정(요약)

* **데이터 빈티지 간소화**: ALFRED 연동 보류. 1차 버전은 지표별 **발표 지연 규칙 기반 시뮬레이션**으로 선견편향을 억제
* **실행 구조 슬림화**: `scripts/` 제거 → `src/cli.py` 의 Typer CLI 단일 엔트리
* **테스트 전략 현실화**: E2E(UI) 자동화 제외, 유닛/통합 테스트 집중
* **기여도 계산 단순화**: `계수 × Δ특징량` 과 변화량 Top-N 중심. SHAP 등은 후속
* **판단보다 사실**: UI 기본값은 합성 판단이 아닌 브레드스·분위·변화량 중심

# 1. 아키텍처 원칙

* 모듈 경계 명확화(수집/정합/피처/모델/시나리오/시각화)
* 계약 기반 설계(타입·예외 명시), 불변·멱등 처리
* 관측성 기본 탑재(구조화 로깅/메트릭)
* AI 바이브코딩 친화(docstring 예제·TODO(ai) 규칙)

# 2. 기술 스택

* 언어: Python 3.11+
* 데이터: pandas, numpy, pyarrow/Parquet, DuckDB(로컬)
* 모델: scikit-learn QuantileRegressor, statsmodels QuantReg, (선택) MIDAS
* 프런트: Streamlit 단일 앱
* 스케줄러: APScheduler(로컬)
* **CLI**: Typer 기반 단일 엔트리(`src/cli.py`)
* 품질: ruff, black, isort, pytest, coverage, mypy(strict 선택)
* 문서: MkDocs(Material), Google-style docstring

# 3. 저장소 구조 v0.1 Lean

```
repo/
├─ app/
│  └─ streamlit_app.py            # 단일 진입점(대시보드)
├─ src/
│  ├─ cli.py                      # Typer CLI: ingest/features/train/validate/app
│  ├─ data_sources.py             # FRED/BLS/BEA/EIA/NFCI 어댑터
│  ├─ align_vintage.py            # asof 조인·리샘플·발표 지연 규칙
│  ├─ features.py                 # 레벨·Δ·Δ²·분위·레짐 배지
│  ├─ models_quantile.py          # 분위회귀 래퍼·블렌드 로직
│  ├─ scenarios.py                # 시나리오 쇼크 적용
│  └─ viz.py                      # 팬 차트·카드 컴포넌트
├─ data/{raw,processed,features}/
├─ tests/{unit,integration}/
├─ docs/{prd.md,dev-architecture.md}
└─ pyproject.toml
```

확장 시 apps/libs 구조로 점진 분리

# 4. 인터페이스와 스키마

## 4.1 데이터 소스 인터페이스

```python
from datetime import date
from typing import Protocol, Iterable
import pandas as pd

class DataSource(Protocol):
    name: str
    def fetch(self, series: Iterable[str], start: date, end: date, **kw) -> pd.DataFrame: ...
```

규약 스키마: index=date, columns=\["series\_id","value","realtime\_start","realtime\_end"]

## 4.2 피처 정의(요약)

* 1차차분: $\Delta x_t = x_t - x_{t-1}$를 생성
* 2차차분: $\Delta^2 x_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$를 생성
* 분위 스코어: 과거 10년 백분위로 \[−1,+1] 스케일
* 레짐 배지: 스프레드 임계·금융여건 임계 등을 기준으로 범주 부여

## 4.3 타깃·DTO

* 타깃 수익률: $r_{t \to t+h} = \log!\big( S_{t+h} / S_t \big)$를 사용, $h \in {1\mathrm{M},3\mathrm{M},6\mathrm{M},12\mathrm{M}}$를 지원
* 팬 차트 DTO 예시

```
{
  "asof": "YYYY-MM-DD",
  "horizon": "1M",
  "quantiles": {"q05": -0.04, "q10": -0.025, "q25": -0.01, "q50": 0.005, "q75": 0.015, "q90": 0.03, "q95": 0.045},
  "drivers": {
    "center": [{"name": "TERM", "contrib": 0.003}, {"name": "VIX_d1", "contrib": -0.002}],
    "width":  [{"name": "VIX", "contrib": 0.008}, {"name": "NFCI", "contrib": 0.004}]
  }
}
```

# 5. 파이프라인

* **수집**: 공공 API 호출 → 재시도·백오프 → Parquet 저장
* **정합**: 월말 asof 조인, 주/월 혼합 정렬, 결측은 명시적 처리(전진보간 금지)
* **피처**: 레벨·$\Delta$·$\Delta^2$ 생성, 분위 스케일링, 레짐 배지
* **모델**: 각 $h,q$ 별 분위회귀 적합 → 단기 폭은 VIX 앵커 → 거시축과 블렌드
* **시나리오**: 입력 피처벡터에 $\Delta x$ 쇼크 적용 → 중앙·폭 재계산
* **시각화**: 팬 차트, 게이지, 카드, 기여도 바

# 6. 검증 규격(Validation spec 통합)

목적: 예측 분포 품질을 일관 측정하고 과신/누수를 방지

## 6.1 데이터 빈티지

* v0.4.1: **발표 지연 규칙 기반 시뮬레이션** 적용(예: 월간 지표는 다음 달 3영업일부터 사용)
* 스냅샷 태깅(snapshot\_id), 시드/라이브러리 버전 기록

## 6.2 백테스트 프로토콜

* 롤링 OOS: 훈련 10년 → 테스트 1년, 월 단위 전진, 스텝마다 재학습
* 타깃: $r_{t \to t+h}$를 사용, $h \in {1\mathrm{M},3\mathrm{M},6\mathrm{M},12\mathrm{M}}$를 평가
* 리샘플: 월말 기준, forward-fill 금지(삭제 또는 결측 플래그)

## 6.3 기준선 모형

* 평균수익률: $q50 = \overline{r}$, 폭은 역사 분산 고정
* 랜덤워크: $q50 = 0$, 폭은 역사 분산
* 옵션단독(1M): 중앙 $q50 = 0$, 폭은 VIX 기반 분산 앵커

## 6.4 평가 지표

* 핀볼 손실: $L_q(y,\hat{y}) = \max{ q(y-\hat{y}),,(q-1)(y-\hat{y}) }$를 평균
* CRPS: 분포 전체 오차를 적분 근사해 산출
* 캘리브레이션: 예측구간 적중률(50%, 80%) – 목표 구간 대비 편차
* 안정성: 롤링 계수 표준편차, 콘 폭 변동성

## 6.5 리포트 산출물

```
evaluations/
  YYYYMM/
    metadata.json               # snapshot_id, seed, 버전
    metrics.csv                 # date,h,metric,value,model,baseline
    calibration.csv             # date,h,ci,hit
    backtest_fanchart.parquet   # t0 시점 분포·실현 경로
```

## 6.6 수용 기준

* 80% 구간 적중률 75~85%, 50% 구간 45~55%
* 기준선 대비 CRPS 5% 이상 개선 시 녹색 배지

# 7. CLI 및 실행 워크플로

* Typer CLI(`src/cli.py`)

  * `ingest` → 원천 수집(기간/시리즈 지정)
  * `features` → 정합·파생 빌드
  * `train` → 분위회귀 학습 저장
  * `validate` → 롤링 OOS 평가 실행
  * `app` → Streamlit 구동
* Makefile 예시

```
make setup     # venv, pre-commit 설치
make ingest    # 기간 지정 수집
make features  # 파생 생성
make train     # 학습
make validate  # 평가
make app       # 대시보드 실행
```

# 8. 테스트 전략

* 유닛: 어댑터, asof, $\Delta$·$\Delta^2$, 분위 계산, 블렌드
* 통합: 수집→정합→피처→모델 사이클(파일 픽스처)
* UI: 수동 확인(스크린샷 보관). **E2E 자동화 제외**

# 9. 품질·규약

* 타입 힌트 100%, mypy(선택 엄격)
* pre-commit: ruff → black → pytest
* 커밋 메시지: Conventional Commits
* Docstring: Google style + 예제

# 10. 확장 노트(선택)

* 구조: apps/libs 분리, FastAPI, Docker, TimescaleDB
* 모델: 옵션체인 기반 RND 복원, HAR/GARCH 폭 동태, SHAP 설명력

# 11. 오픈 이슈

* 발표 지연 규칙의 경험적 보정
* 시장축↔거시축 가중 전이 함수(선형/지수)
* CRPS 계산의 수치 안정화
