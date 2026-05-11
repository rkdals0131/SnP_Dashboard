"""명령줄 인터페이스 (CLI) 모듈

Typer를 사용하여 데이터 수집부터 대시보드 실행까지
전체 파이프라인을 관리하는 CLI 제공.

Example:
    $ python -m src.cli ingest --source fred --start 2020-01-01
    $ python -m src.cli features
    $ python -m src.cli train --horizons 1M,3M
    $ python -m src.cli validate
    $ python -m src.cli app
"""

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import pandas as pd
from structlog import get_logger

# .env 로드 (FRED_API_KEY 등)
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except Exception:
    # dotenv 미설치/에러 시 무시하고 환경변수만 사용
    pass

# 로컬 모듈 임포트
from .data_sources import FREDSource, FearGreedSource, persist_parquet
from .align_vintage import (
    create_month_end_calendar,
    build_master_panel,
    apply_publication_delays,
    PUBLICATION_RULES
)
from .features import build_feature_panel
from .targets import compute_forward_returns, save_targets
from .models_quantile import train_models, save_models, DEFAULT_QUANTILES
from .validation import run_rolling_validation, create_validation_report

logger = get_logger()
console = Console()
# Typer 앱 인스턴스 이름을 함수명과 충돌하지 않도록 분리
cli = typer.Typer(help="S&P 리스크·신호 집계판 CLI")

# 데이터 디렉토리 설정
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
TARGETS_DIR = DATA_DIR / "targets"
MODELS_DIR = Path("models")
EVALUATIONS_DIR = Path("evaluations")

# 기본 시리즈 목록
DEFAULT_SERIES = [
    "DGS10", "DGS3MO", "DGS2",  # 국채 수익률
    "VIXCLS",  # VIX
    "SP500",  # S&P 500
    "DCOILWTICO",  # WTI 유가
    "GOLDAMGBD228NLBM",  # 금 가격(LBMA AM, USD)
    "BAA", "BAMLH0A0HYM2",  # 크레딧 스프레드
    "UNRATE",  # 실업률
    "CPIAUCSL", "CPILFESL",  # 인플레이션
    "NFCI"  # 금융여건 지수
]


@cli.command()
def ingest(
    source: str = typer.Option("fred", help="데이터 소스 (fred, fear_greed, bls, bea)"),
    series: Optional[str] = typer.Option(None, help="수집할 시리즈 (쉼표 구분)"),
    start: str = typer.Option("2000-01-01", help="시작 날짜 (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, help="종료 날짜 (YYYY-MM-DD)"),
    api_key: Optional[str] = typer.Option(None, envvar="FRED_API_KEY", help="API 키")
):
    """데이터 수집 명령
    
    공공 API에서 시계열 데이터를 수집하여 Parquet 형식으로 저장합니다.
    """
    console.print(f"[bold blue]데이터 수집 시작[/bold blue]")
    console.print(f"소스: {source}")
    
    source_lc = source.lower()

    # 시리즈 파싱(소스별 기본값)
    if series:
        series_list = [s.strip() for s in series.split(",")]
    elif source_lc in {"fear_greed", "fng", "feargreed"}:
        series_list = ["CRYPTO_FNG"]
        console.print("기본 시리즈 사용: CRYPTO_FNG")
    else:
        series_list = DEFAULT_SERIES
        console.print(f"기본 시리즈 사용: {len(series_list)}개")
    
    # 날짜 파싱
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date() if end else date.today()
    
    console.print(f"기간: {start_date} ~ {end_date}")
    console.print(f"시리즈: {', '.join(series_list)}")
    
    # 소스별 처리
    if source_lc == "fred":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("FRED 데이터 수집 중...", total=len(series_list))
            
            try:
                # FRED 소스 초기화
                fred = FREDSource(api_key=api_key, cache_dir=RAW_DIR / "fred")
                
                # 데이터 수집
                df = fred.fetch(series_list, start_date, end_date)
                
                if df.empty:
                    console.print("[red]수집된 데이터가 없습니다.[/red]")
                    raise typer.Exit(1)
                
                # 통계 표시
                table = Table(title="수집 결과")
                table.add_column("시리즈", style="cyan")
                table.add_column("관측치", style="magenta")
                table.add_column("시작일", style="green")
                table.add_column("종료일", style="green")
                
                for series_id in series_list:
                    series_data = df[df["series_id"] == series_id]
                    if not series_data.empty:
                        table.add_row(
                            series_id,
                            str(len(series_data)),
                            str(series_data.index.min().date()),
                            str(series_data.index.max().date())
                        )
                
                console.print(table)
                console.print(f"[green]✓ 데이터 수집 완료![/green]")
                
            except Exception as e:
                console.print(f"[red]오류 발생: {e}[/red]")
                raise typer.Exit(1)

    elif source_lc in {"fear_greed", "fng", "feargreed"}:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("공포-탐욕 지수 수집 중...", total=len(series_list))

            try:
                fng = FearGreedSource(cache_dir=RAW_DIR / "fear_greed")
                df = fng.fetch(series_list, start_date, end_date)

                if df.empty:
                    console.print("[red]수집된 공포-탐욕 데이터가 없습니다.[/red]")
                    raise typer.Exit(1)

                table = Table(title="공포-탐욕 수집 결과")
                table.add_column("시리즈", style="cyan")
                table.add_column("관측치", style="magenta")
                table.add_column("시작일", style="green")
                table.add_column("종료일", style="green")
                table.add_column("최신값", style="yellow")

                for series_id in series_list:
                    series_data = df[df["series_id"] == series_id]
                    if series_data.empty:
                        continue
                    latest_value = series_data["value"].dropna().iloc[-1] if not series_data["value"].dropna().empty else float("nan")
                    table.add_row(
                        series_id,
                        str(len(series_data)),
                        str(series_data.index.min().date()),
                        str(series_data.index.max().date()),
                        f"{latest_value:.1f}" if pd.notna(latest_value) else "N/A",
                    )

                progress.update(task, completed=len(series_list))
                console.print(table)
                console.print("[green]✓ 공포-탐욕 데이터 수집 완료![/green]")

            except Exception as e:
                console.print(f"[red]오류 발생: {e}[/red]")
                raise typer.Exit(1)

    else:
        console.print(f"[yellow]'{source}' 소스는 아직 구현되지 않았습니다.[/yellow]")
        raise typer.Exit(1)


@cli.command()
def features(
    calendar_start: str = typer.Option("2000-01-01", help="캘린더 시작일"),
    calendar_end: Optional[str] = typer.Option(None, help="캘린더 종료일")
):
    """피처 생성 명령
    
    원천 데이터를 정합하고 피처 엔지니어링을 수행합니다.
    """
    console.print("[bold blue]피처 생성 시작[/bold blue]")
    
    # 캘린더 생성
    end_date = calendar_end or date.today().strftime("%Y-%m-%d")
    calendar = create_month_end_calendar(calendar_start, end_date)
    console.print(f"월말 캘린더 생성: {len(calendar)}개 시점")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        
        # 1. 원천 데이터 경로 수집
        task = progress.add_task("원천 데이터 확인 중...", total=1)
        
        raw_paths = {}
        for series_file in (RAW_DIR / "fred").glob("*.parquet"):
            series_id = series_file.stem
            raw_paths[series_id] = series_file
        
        console.print(f"발견된 시리즈: {len(raw_paths)}개")
        progress.update(task, completed=1)
        
        if not raw_paths:
            console.print("[red]원천 데이터를 찾을 수 없습니다. 먼저 'ingest' 명령을 실행하세요.[/red]")
            raise typer.Exit(1)
        
        # 2. 마스터 패널 구축
        task = progress.add_task("마스터 패널 구축 중...", total=1)
        
        try:
            master_panel = build_master_panel(raw_paths, calendar)
            console.print(f"마스터 패널 구축: {master_panel.shape}")
            
            # 빈티지 처리
            vintage_panel = apply_publication_delays(master_panel, PUBLICATION_RULES)
            
            # 저장
            vintage_path = PROCESSED_DIR / "master_timeseries.parquet"
            vintage_path.parent.mkdir(parents=True, exist_ok=True)
            vintage_panel.to_parquet(vintage_path)
            
            progress.update(task, completed=1)
            
        except Exception as e:
            console.print(f"[red]마스터 패널 구축 실패: {e}[/red]")
            raise typer.Exit(1)
        
        # 3. 피처 생성
        task = progress.add_task("피처 생성 중...", total=1)
        
        try:
            feature_panel = build_feature_panel(vintage_panel)
            console.print(f"피처 생성: {feature_panel.shape}")
            
            # 저장
            feature_path = FEATURES_DIR / "features.parquet"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            feature_panel.to_parquet(feature_path)
            
            progress.update(task, completed=1)
            
        except Exception as e:
            console.print(f"[red]피처 생성 실패: {e}[/red]")
            raise typer.Exit(1)
        
        # 4. 타깃 생성
        task = progress.add_task("타깃 변수 생성 중...", total=1)
        
        try:
            if "SP500" in vintage_panel.columns:
                price_df = vintage_panel[["SP500"]].copy()
                targets = compute_forward_returns(price_df, "SP500")
                
                # 저장
                target_path = TARGETS_DIR / "targets.parquet"
                save_targets(targets, target_path)
                
                console.print(f"타깃 생성: {targets.shape}")
            else:
                console.print("[yellow]S&P 500 데이터가 없어 타깃을 생성할 수 없습니다.[/yellow]")
            
            progress.update(task, completed=1)
            
        except Exception as e:
            console.print(f"[red]타깃 생성 실패: {e}[/red]")
            raise typer.Exit(1)
    
    console.print("[green]✓ 피처 생성 완료![/green]")


@cli.command()
def train(
    horizons: str = typer.Option("1M,3M,6M,12M", help="학습할 예측 기간 (쉼표 구분)"),
    quantiles: Optional[str] = typer.Option(None, help="학습할 분위수 (쉼표 구분)"),
    alpha: float = typer.Option(1.0, help="L1 규제 강도"),
    snapshot: Optional[str] = typer.Option(None, help="스냅샷 ID")
):
    """모델 학습 명령
    
    분위 회귀 모델을 학습하고 저장합니다.
    """
    console.print("[bold blue]모델 학습 시작[/bold blue]")
    
    # 파라미터 파싱
    horizon_list = [h.strip() for h in horizons.split(",")]
    
    if quantiles:
        quantile_list = [float(q.strip()) for q in quantiles.split(",")]
    else:
        quantile_list = DEFAULT_QUANTILES
    
    console.print(f"예측 기간: {horizon_list}")
    console.print(f"분위수: {quantile_list}")
    console.print(f"L1 규제: {alpha}")
    
    # 데이터 로드
    try:
        features = pd.read_parquet(FEATURES_DIR / "features.parquet")
        targets = pd.read_parquet(TARGETS_DIR / "targets.parquet")
        console.print(f"데이터 로드: 피처 {features.shape}, 타깃 {targets.shape}")
    except FileNotFoundError:
        console.print("[red]피처 또는 타깃 파일을 찾을 수 없습니다. 'features' 명령을 먼저 실행하세요.[/red]")
        raise typer.Exit(1)
    
    # 피처와 타깃 정렬
    from .targets import align_targets_with_features
    
    try:
        aligned_data = align_targets_with_features(targets, features)
        
        # 타깃이 있는 컬럼만 선택
        target_cols = [col for col in aligned_data.columns if col.startswith("return_")]
        feature_cols = [col for col in aligned_data.columns if not col.startswith("return_") and not col.endswith("_is_avail")]
        
        X = aligned_data[feature_cols]
        y = aligned_data[target_cols]
        
        console.print(f"학습 데이터: X {X.shape}, y {y.shape}")
        
    except Exception as e:
        console.print(f"[red]데이터 정렬 실패: {e}[/red]")
        raise typer.Exit(1)
    
    # 모델 학습
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("모델 학습 중...", total=len(horizon_list))
        
        try:
            models = train_models(
                X, y,
                horizons=horizon_list,
                quantiles=quantile_list,
                config={"alpha": alpha, "solver": "highs"}
            )
            
            # 스냅샷 ID 생성
            if not snapshot:
                snapshot = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 모델 저장
            model_path = MODELS_DIR / snapshot / "models.pkl"
            save_models(models, model_path)
            
            progress.update(task, completed=len(horizon_list))
            
            # 결과 요약
            table = Table(title=f"학습 완료 (스냅샷: {snapshot})")
            table.add_column("기간", style="cyan")
            table.add_column("분위수", style="magenta")
            
            for horizon in models.keys():
                table.add_row(horizon, str(len(quantile_list)))
            
            console.print(table)
            console.print(f"[green]✓ 모델 저장 완료: {model_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]모델 학습 실패: {e}[/red]")
            raise typer.Exit(1)


@cli.command()
def validate(
    snapshot: Optional[str] = typer.Option(None, help="검증할 모델 스냅샷(예: 20240101_120000 또는 경로)"),
    start_date: Optional[str] = typer.Option(None, help="검증 시작일"),
    end_date: Optional[str] = typer.Option(None, help="검증 종료일"),
    train_window: int = typer.Option(60, help="롤링 학습 윈도 크기(개월)"),
    test_window: int = typer.Option(12, help="롤링 테스트 윈도 크기(개월)"),
    step_size: int = typer.Option(1, help="롤링 스텝 크기(개월)")
):
    """모델 검증 명령
    
    롤링 아웃오브샘플 검증을 수행합니다.
    """
    console.print("[bold blue]모델 검증 시작[/bold blue]")
    
    # 스냅샷 선택
    model_path: Path
    snapshot_id: str
    if not snapshot:
        # 가장 최근 스냅샷 사용
        model_dirs = sorted(MODELS_DIR.glob("*/"))
        if not model_dirs:
            console.print("[red]학습된 모델을 찾을 수 없습니다.[/red]")
            raise typer.Exit(1)
        snapshot_id = model_dirs[-1].name
        model_path = MODELS_DIR / snapshot_id / "models.pkl"
    else:
        # 경로 또는 스냅샷 ID 모두 지원
        snap_path = Path(snapshot)
        if snap_path.is_file():
            model_path = snap_path
            snapshot_id = snap_path.parent.name
        elif snap_path.is_dir():
            model_path = snap_path / "models.pkl"
            snapshot_id = snap_path.name
        else:
            model_path = MODELS_DIR / snapshot / "models.pkl"
            snapshot_id = snapshot

    console.print(f"스냅샷 파일: {model_path}")
    
    # 데이터 로드
    try:
        features = pd.read_parquet(FEATURES_DIR / "features.parquet")
        targets = pd.read_parquet(TARGETS_DIR / "targets.parquet")
        
        from .targets import align_targets_with_features
        aligned_data = align_targets_with_features(targets, features)
        
        # 검증 기간 설정
        if start_date:
            aligned_data = aligned_data[aligned_data.index >= start_date]
        if end_date:
            aligned_data = aligned_data[aligned_data.index <= end_date]
        
        console.print(f"검증 데이터: {aligned_data.shape}")
        
    except Exception as e:
        console.print(f"[red]데이터 로드 실패: {e}[/red]")
        raise typer.Exit(1)
    
    # 검증 실행
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("롤링 검증 실행 중...", total=1)
        
        try:
            # 모델 로드
            from .models_quantile import load_models
            models = load_models(model_path)
            
            # 피처/타깃 분리
            target_cols = [col for col in aligned_data.columns if col.startswith("return_")]
            feature_cols = [col for col in aligned_data.columns if not col.startswith("return_") and not col.endswith("_is_avail")]
            
            X = aligned_data[feature_cols]
            y = aligned_data[target_cols]
            
            # 윈도 유효성 보정
            total_len = len(X)
            tw = min(train_window, max(12, total_len // 2))
            vw = min(test_window, max(6, total_len - tw))
            if tw + vw > total_len:
                vw = max(6, total_len - tw)
            if tw + vw > total_len:
                console.print("[red]검증에 필요한 데이터 길이가 부족합니다.[/red]")
                raise typer.Exit(1)

            # 검증 실행
            results = run_rolling_validation(
                models, X, y,
                train_window=tw,
                test_window=vw,
                step_size=step_size
            )
            
            progress.update(task, completed=1)
            
            # 결과 저장
            # snapshot이 None인 경우도 안전하게 디렉토리 구성
            eval_dir = EVALUATIONS_DIR / snapshot_id
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # 리포트 생성
            report = create_validation_report(results)
            report_path = eval_dir / "validation_report.json"
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # 결과 요약 표시
            table = Table(title="검증 결과 요약")
            table.add_column("지표", style="cyan")
            table.add_column("1M", style="magenta")
            table.add_column("3M", style="magenta")
            table.add_column("6M", style="magenta")
            table.add_column("12M", style="magenta")
            
            # 메트릭 표시
            metrics = ["pinball_loss", "crps", "coverage_50", "coverage_80"]
            for metric in metrics:
                row = [metric]
                for horizon in ["1M", "3M", "6M", "12M"]:
                    value = report.get("summary", {}).get(horizon, {}).get(metric, "N/A")
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                table.add_row(*row)
            
            console.print(table)
            console.print(f"[green]✓ 검증 완료! 리포트: {report_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]검증 실패: {e}[/red]")
            raise typer.Exit(1)


# 커맨드 함수명 'app'과 Typer 인스턴스명이 충돌하지 않도록 함수명을 변경
@cli.command(name="app")
def run_app(
    port: int = typer.Option(8501, help="포트 번호"),
    host: str = typer.Option("0.0.0.0", help="바인드 주소 (WSL/LAN 접근은 0.0.0.0 권장)"),
    debug: bool = typer.Option(False, help="디버그 모드")
):
    """대시보드 실행 명령
    
    Streamlit 웹 대시보드를 실행합니다.
    """
    console.print("[bold blue]대시보드 실행[/bold blue]")
    
    # Streamlit 앱 경로
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    
    if not app_path.exists():
        console.print(f"[red]앱 파일을 찾을 수 없습니다: {app_path}[/red]")
        raise typer.Exit(1)
    
    # Streamlit 실행
    import subprocess
    
    cmd = [
        "streamlit", "run",
        str(app_path),
        f"--server.port={port}",
        f"--server.address={host}",
        "--server.headless=true"
    ]
    
    if debug:
        cmd.append("--logger.level=debug")
    
    console.print(f"대시보드 시작 중... (http://{host}:{port})")
    if host == "0.0.0.0":
        console.print(f"같은 네트워크에서는 WSL/호스트 IP와 포트 {port}로 접속하세요.")
    console.print("종료하려면 Ctrl+C를 누르세요.")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]대시보드를 종료합니다.[/yellow]")
    except Exception as e:
        console.print(f"[red]대시보드 실행 실패: {e}[/red]")
        raise typer.Exit(1)


@cli.command()
def status():
    """시스템 상태 확인 명령
    
    데이터와 모델의 현재 상태를 확인합니다.
    """
    console.print("[bold blue]시스템 상태[/bold blue]\n")
    
    # 데이터 상태
    table = Table(title="데이터 상태")
    table.add_column("항목", style="cyan")
    table.add_column("상태", style="magenta")
    table.add_column("상세", style="green")
    
    # 원천 데이터
    raw_files = list((RAW_DIR / "fred").glob("*.parquet")) if (RAW_DIR / "fred").exists() else []
    table.add_row(
        "원천 데이터",
        "✓" if raw_files else "✗",
        f"{len(raw_files)}개 시리즈" if raw_files else "없음"
    )
    
    # 마스터 패널
    master_path = PROCESSED_DIR / "master_timeseries.parquet"
    if master_path.exists():
        master_df = pd.read_parquet(master_path)
        table.add_row(
            "마스터 패널",
            "✓",
            f"{master_df.shape[0]}행 × {master_df.shape[1]}열"
        )
    else:
        table.add_row("마스터 패널", "✗", "없음")
    
    # 피처
    feature_path = FEATURES_DIR / "features.parquet"
    if feature_path.exists():
        feature_df = pd.read_parquet(feature_path)
        table.add_row(
            "피처",
            "✓",
            f"{feature_df.shape[0]}행 × {feature_df.shape[1]}열"
        )
    else:
        table.add_row("피처", "✗", "없음")
    
    # 타깃
    target_path = TARGETS_DIR / "targets.parquet"
    if target_path.exists():
        target_df = pd.read_parquet(target_path)
        table.add_row(
            "타깃",
            "✓",
            f"{target_df.shape[0]}행 × {target_df.shape[1]}열"
        )
    else:
        table.add_row("타깃", "✗", "없음")
    
    console.print(table)
    
    # 모델 상태
    if MODELS_DIR.exists():
        model_dirs = sorted(MODELS_DIR.glob("*/"))
        if model_dirs:
            table = Table(title="모델 상태")
            table.add_column("스냅샷", style="cyan")
            table.add_column("생성일시", style="magenta")
            table.add_column("크기", style="green")
            
            for model_dir in model_dirs[-5:]:  # 최근 5개만
                model_file = model_dir / "models.pkl"
                if model_file.exists():
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    table.add_row(
                        model_dir.name,
                        datetime.fromtimestamp(model_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                        f"{size_mb:.1f} MB"
                    )
            
            console.print(table)
    
    console.print("\n실행 가능한 명령:")
    console.print("  • ingest   - 데이터 수집")
    console.print("  • features - 피처 생성")
    console.print("  • train    - 모델 학습")
    console.print("  • validate - 모델 검증")
    console.print("  • app      - 대시보드 실행")


if __name__ == "__main__":
    cli()
