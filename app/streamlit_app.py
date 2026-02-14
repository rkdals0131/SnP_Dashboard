"""Streamlit dashboard for macro overview and S&P 500 fan chart."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import structlog


def _configure_structlog_for_streamlit() -> None:
    """Avoid PrintLogger writes that can fail on some Windows/Streamlit sessions."""
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
from structlog import get_logger


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.align_vintage import (  # noqa: E402
    PUBLICATION_RULES,
    apply_publication_delays,
    build_master_panel,
    create_month_end_calendar,
)
from src.data_sources import FREDSource, FXSource, FearGreedSource  # noqa: E402
from src.features import build_feature_panel  # noqa: E402
from src.models_quantile import (  # noqa: E402
    DEFAULT_QUANTILES,
    load_models,
    predict_fanchart,
    save_models,
    train_models,
)
from src.scenarios import (  # noqa: E402
    DEFAULT_SCENARIOS,
    apply_shocks,
    compute_contributions,
    create_scenario_comparison,
)
from src.targets import (  # noqa: E402
    align_targets_with_features,
    compute_forward_returns,
    save_targets,
)
from src.validation import (  # noqa: E402
    ValidationConfig,
    create_validation_report,
    run_rolling_validation,
)
from src.viz import (  # noqa: E402
    create_dashboard_layout,
    render_ai_one_liner,
    render_auto_summary,
    render_bollinger_bands,
    render_breadth_bar,
    render_contribution_chart,
    render_fan_chart,
    render_indicator_card,
    render_indicator_narratives,
    render_macro_score_tile,
    render_regime_badges,
    render_scenario_comparison,
    render_uncertainty_gauge,
)


logger = get_logger()

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_PATH = DATA_DIR / "processed" / "master_timeseries.parquet"
FEATURES_PATH = DATA_DIR / "features" / "features.parquet"
TARGETS_PATH = DATA_DIR / "targets" / "targets.parquet"
MODELS_DIR = Path("models")
EVALUATIONS_DIR = Path("evaluations")

SHOCK_ALIASES = {
    "VIX": "VIXCLS",
    "CREDIT_SPREAD": "BAMLH0A0HYM2",
}


try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
except Exception:
    pass


def _safe_log(level: str, event: str, **kwargs: Any) -> None:
    try:
        getattr(logger, level)(event, **kwargs)
    except Exception:
        pass


def _parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_quantiles(text: str) -> List[float]:
    if not text.strip():
        return DEFAULT_QUANTILES
    values: List[float] = []
    for raw in _parse_csv_list(text):
        value = float(raw)
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"Invalid quantile '{raw}'. Must be between 0 and 1.")
        values.append(value)
    return sorted(set(values))


def _default_series_for_tab_a() -> List[str]:
    return [
        "DGS10",
        "DGS3MO",
        "DGS2",
        "VIXCLS",
        "SP500",
        "DCOILWTICO",
        "GOLDAMGBD228NLBM",
        "BAA",
        "BAMLH0A0HYM2",
        "UNRATE",
        "CPIAUCSL",
        "CPILFESL",
        "NFCI",
        "FEDFUNDS",
        "PAYEMS",
        "CIVPART",
        "JTSJOL",
        "DTWEXBGS",
    ]


def _collect_raw_paths(
    series_list: Optional[List[str]] = None,
    include_fx: bool = True,
    include_fng: bool = True,
) -> Dict[str, Path]:
    raw_paths: Dict[str, Path] = {}

    fred_dir = RAW_DIR / "fred"
    if series_list is None:
        for file in fred_dir.glob("*.parquet"):
            raw_paths[file.stem] = file
    else:
        for sid in series_list:
            file = fred_dir / f"{sid}.parquet"
            if file.exists():
                raw_paths[sid] = file

    if include_fx:
        fx_dir = RAW_DIR / "fx"
        for file in fx_dir.glob("*.parquet"):
            raw_paths[file.stem] = file

    if include_fng:
        fng_file = RAW_DIR / "fear_greed" / "CRYPTO_FNG.parquet"
        if fng_file.exists():
            raw_paths["CRYPTO_FNG"] = fng_file

    return raw_paths


def _load_features() -> Optional[pd.DataFrame]:
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    return None


def _load_prices() -> Optional[pd.DataFrame]:
    if PROCESSED_PATH.exists():
        df = pd.read_parquet(PROCESSED_PATH)
        if "SP500" in df.columns:
            return df[["SP500"]].dropna()
    return None


def _load_daily_sp500() -> Optional[pd.Series]:
    raw_path = RAW_DIR / "fred" / "SP500.parquet"
    if not raw_path.exists():
        return None

    try:
        df = pd.read_parquet(raw_path)
        if "series_id" in df.columns:
            df = df[df["series_id"] == "SP500"]

        if "value" in df.columns:
            values = pd.to_numeric(df["value"], errors="coerce")
        elif "SP500" in df.columns:
            values = pd.to_numeric(df["SP500"], errors="coerce")
        else:
            values = pd.to_numeric(df.iloc[:, 0], errors="coerce")

        values.index = pd.to_datetime(values.index)
        series = values.sort_index().dropna().rename("SP500")
        return series if not series.empty else None
    except Exception as exc:
        _safe_log("warning", "daily_sp500_load_failed", error=str(exc))
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _build_live_features_for_tab_a(
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    end_str = end or date.today().strftime("%Y-%m-%d")
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end_str).date()

    series_list = _default_series_for_tab_a()
    fx_pairs = ["USDKRW", "USDJPY", "USDEUR"]

    fred = FREDSource()
    _ = fred.fetch(series_list, start_dt, end_dt)

    fx = FXSource()
    _ = fx.fetch(fx_pairs, start_dt, end_dt)

    try:
        fng = FearGreedSource(provider="alternative")
        _ = fng.fetch(["CRYPTO_FNG"], start_dt, end_dt)
    except Exception as exc:
        _safe_log("warning", "fear_greed_fetch_optional_failed", error=str(exc))

    raw_paths = _collect_raw_paths(series_list=series_list, include_fx=True, include_fng=True)
    if not raw_paths:
        raise RuntimeError("No raw inputs found after live fetch.")

    calendar = create_month_end_calendar(start, end_str)
    master_panel = build_master_panel(raw_paths, calendar)
    vintage_panel = apply_publication_delays(master_panel, PUBLICATION_RULES)
    return build_feature_panel(vintage_panel)


def _select_feature_row(features: pd.DataFrame) -> pd.Series:
    if features.empty:
        return pd.Series(dtype=float)
    return features.ffill().bfill().iloc[-1]


def _list_model_snapshots() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.name for p in MODELS_DIR.iterdir() if p.is_dir()])


def _latest_snapshot() -> Optional[str]:
    snaps = _list_model_snapshots()
    return snaps[-1] if snaps else None


def _normalize_shocks(shocks: Dict[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in shocks.items():
        mapped = SHOCK_ALIASES.get(key, key)
        normalized[mapped] = normalized.get(mapped, 0.0) + float(value)
    return normalized


def _ingest_data(start: date, end: date, include_fx: bool, include_fng: bool) -> Dict[str, int]:
    series_list = _default_series_for_tab_a()
    result = {"fred_rows": 0, "fx_rows": 0, "fear_greed_rows": 0}

    fred = FREDSource()
    fred_df = fred.fetch(series_list, start, end)
    result["fred_rows"] = len(fred_df)

    if include_fx:
        fx = FXSource()
        fx_df = fx.fetch(["USDKRW", "USDJPY", "USDEUR"], start, end)
        result["fx_rows"] = len(fx_df)

    if include_fng:
        fng = FearGreedSource(provider="alternative")
        fng_df = fng.fetch(["CRYPTO_FNG"], start, end)
        result["fear_greed_rows"] = len(fng_df)

    return result


def _build_features_from_raw(calendar_start: str, calendar_end: str) -> Dict[str, Any]:
    calendar = create_month_end_calendar(calendar_start, calendar_end)
    raw_paths = _collect_raw_paths(series_list=None, include_fx=True, include_fng=True)
    if not raw_paths:
        raise RuntimeError("No raw inputs found. Run ingest first.")

    master_panel = build_master_panel(raw_paths, calendar)
    vintage_panel = apply_publication_delays(master_panel, PUBLICATION_RULES)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    vintage_panel.to_parquet(PROCESSED_PATH)

    features = build_feature_panel(vintage_panel)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH)

    targets_rows = 0
    if "SP500" in vintage_panel.columns:
        targets = compute_forward_returns(vintage_panel[["SP500"]], "SP500")
        TARGETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_targets(targets, TARGETS_PATH)
        targets_rows = len(targets)

    return {
        "master_shape": master_panel.shape,
        "feature_shape": features.shape,
        "targets_rows": targets_rows,
    }


def _train_from_files(
    horizons: List[str],
    quantiles: List[float],
    alpha: float,
    snapshot: Optional[str],
) -> Dict[str, Any]:
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise RuntimeError("features/targets artifacts are missing. Run feature build first.")

    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    aligned = align_targets_with_features(targets, features)

    target_cols = [f"return_{h}" for h in horizons if f"return_{h}" in aligned.columns]
    if not target_cols:
        raise RuntimeError("No matching target columns for selected horizons.")

    feature_cols = [
        col
        for col in aligned.columns
        if not col.startswith("return_") and not col.endswith("_is_avail")
    ]
    if not feature_cols:
        raise RuntimeError("No feature columns available for training.")

    x_data = aligned[feature_cols]
    y_data = aligned[target_cols]

    models = train_models(
        x_data,
        y_data,
        horizons=horizons,
        quantiles=quantiles,
        config={"alpha": alpha, "solver": "highs"},
    )
    if not models:
        raise RuntimeError("Training produced no models.")

    snapshot_id = snapshot or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / snapshot_id / "models.pkl"
    save_models(models, model_path)

    return {
        "snapshot": snapshot_id,
        "path": str(model_path),
        "trained_horizons": sorted(models.keys()),
        "n_rows": len(x_data),
    }


def _resolve_snapshot(snapshot: Optional[str]) -> Tuple[str, Path]:
    if snapshot and snapshot != "(none)":
        snapshot_id = snapshot
    else:
        latest = _latest_snapshot()
        if not latest:
            raise RuntimeError("No model snapshot found.")
        snapshot_id = latest

    model_path = MODELS_DIR / snapshot_id / "models.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return snapshot_id, model_path


def _validate_from_files(
    snapshot: Optional[str],
    train_window: int,
    test_window: int,
    step_size: int,
    retrain_each_step: bool,
) -> Dict[str, Any]:
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise RuntimeError("features/targets artifacts are missing.")

    snapshot_id, model_path = _resolve_snapshot(snapshot)
    models = load_models(model_path)
    if not models:
        raise RuntimeError("Loaded model snapshot is empty.")

    features = pd.read_parquet(FEATURES_PATH)
    targets = pd.read_parquet(TARGETS_PATH)
    aligned = align_targets_with_features(targets, features)

    target_cols = [f"return_{h}" for h in models.keys() if f"return_{h}" in aligned.columns]
    if not target_cols:
        raise RuntimeError("No matching targets for loaded model horizons.")

    feature_cols = [
        col
        for col in aligned.columns
        if not col.startswith("return_") and not col.endswith("_is_avail")
    ]

    x_data = aligned[feature_cols]
    y_data = aligned[target_cols]

    total_len = len(x_data)
    tw = min(train_window, max(12, total_len // 2))
    vw = min(test_window, max(6, total_len - tw))
    if tw + vw > total_len:
        vw = max(6, total_len - tw)
    if tw + vw > total_len:
        raise RuntimeError("Not enough aligned rows for validation windows.")

    any_model = next(iter(models.values()))
    cfg = ValidationConfig(
        horizons=list(models.keys()),
        quantiles=any_model.quantiles,
        retrain_each_step=retrain_each_step,
    )
    results = run_rolling_validation(
        models,
        x_data,
        y_data,
        train_window=tw,
        test_window=vw,
        step_size=step_size,
        config=cfg,
    )
    report = create_validation_report(results)

    eval_dir = EVALUATIONS_DIR / snapshot_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)

    return {"snapshot": snapshot_id, "report_path": str(report_path), "report": report}


def _run_full_pipeline(
    ingest_start: date,
    ingest_end: date,
    include_fx: bool,
    include_fng: bool,
    calendar_start: str,
    calendar_end: str,
    horizons: List[str],
    quantiles: List[float],
    alpha: float,
) -> Dict[str, Any]:
    ingest_stats = _ingest_data(ingest_start, ingest_end, include_fx=include_fx, include_fng=include_fng)
    feature_stats = _build_features_from_raw(calendar_start=calendar_start, calendar_end=calendar_end)
    train_stats = _train_from_files(horizons=horizons, quantiles=quantiles, alpha=alpha, snapshot=None)
    return {
        "ingest": ingest_stats,
        "features": feature_stats,
        "train": train_stats,
    }


def main() -> None:
    tab_a, tab_b = create_dashboard_layout()

    if "selected_snapshot" not in st.session_state:
        st.session_state["selected_snapshot"] = _latest_snapshot() or "(none)"

    with st.sidebar:
        st.header("Settings")
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.success("Cache cleared.")

        use_live_features = st.checkbox("Use live features for Tab A", value=False)
        live_start = st.text_input("Live start date", value="2000-01-01")
        live_end = st.text_input("Live end date (blank=today)", value="")

        with st.expander("GUI Pipeline Runner", expanded=False):
            ingest_start = st.date_input("Ingest start", value=date(2010, 1, 1), key="ingest_start")
            ingest_end = st.date_input("Ingest end", value=date.today(), key="ingest_end")
            include_fx = st.checkbox("Include FX", value=True, key="include_fx")
            include_fng = st.checkbox("Include Fear&Greed", value=True, key="include_fng")

            calendar_start = st.text_input("Calendar start", value="2010-01-01")
            calendar_end = st.text_input("Calendar end (blank=today)", value="")

            horizons_text = st.text_input("Train horizons", value="1M,3M,6M,12M")
            quantiles_text = st.text_input("Quantiles", value="0.05,0.10,0.25,0.50,0.75,0.90,0.95")
            alpha = st.number_input("L1 alpha", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            train_window = st.number_input("Validate train window", min_value=12, max_value=240, value=60, step=1)
            test_window = st.number_input("Validate test window", min_value=6, max_value=120, value=12, step=1)
            step_size = st.number_input("Validate step", min_value=1, max_value=24, value=1, step=1)
            retrain_each_step = st.checkbox("Retrain each validation step", value=True)

            col_ingest, col_feature = st.columns(2)
            col_train, col_validate = st.columns(2)
            run_ingest = col_ingest.button("1) Ingest", use_container_width=True)
            run_features = col_feature.button("2) Features", use_container_width=True)
            run_train = col_train.button("3) Train", use_container_width=True)
            run_validate = col_validate.button("4) Validate", use_container_width=True)
            run_all = st.button("Run full pipeline (1->3)", type="primary", use_container_width=True)

            calendar_end_effective = calendar_end.strip() or date.today().strftime("%Y-%m-%d")
            horizons = _parse_csv_list(horizons_text)

            if run_ingest:
                with st.spinner("Running ingest..."):
                    try:
                        stats = _ingest_data(
                            ingest_start,
                            ingest_end,
                            include_fx=include_fx,
                            include_fng=include_fng,
                        )
                        st.success(
                            f"Ingest done: FRED={stats['fred_rows']}, FX={stats['fx_rows']}, "
                            f"FearGreed={stats['fear_greed_rows']}"
                        )
                        st.cache_data.clear()
                    except Exception as exc:
                        _safe_log("warning", "gui_ingest_failed", error=str(exc))
                        st.error(f"Ingest failed: {exc}")

            if run_features:
                with st.spinner("Building feature artifacts..."):
                    try:
                        stats = _build_features_from_raw(
                            calendar_start=calendar_start,
                            calendar_end=calendar_end_effective,
                        )
                        st.success(
                            f"Features done: master={stats['master_shape']}, "
                            f"features={stats['feature_shape']}, targets_rows={stats['targets_rows']}"
                        )
                        st.cache_data.clear()
                    except Exception as exc:
                        _safe_log("warning", "gui_features_failed", error=str(exc))
                        st.error(f"Feature build failed: {exc}")

            if run_train:
                with st.spinner("Training models..."):
                    try:
                        quantiles = _parse_quantiles(quantiles_text)
                        stats = _train_from_files(
                            horizons=horizons,
                            quantiles=quantiles,
                            alpha=float(alpha),
                            snapshot=None,
                        )
                        st.session_state["selected_snapshot"] = stats["snapshot"]
                        st.success(
                            f"Train done: snapshot={stats['snapshot']}, horizons={stats['trained_horizons']}"
                        )
                    except Exception as exc:
                        _safe_log("warning", "gui_train_failed", error=str(exc))
                        st.error(f"Training failed: {exc}")

            if run_validate:
                with st.spinner("Running validation..."):
                    try:
                        stats = _validate_from_files(
                            snapshot=st.session_state.get("selected_snapshot"),
                            train_window=int(train_window),
                            test_window=int(test_window),
                            step_size=int(step_size),
                            retrain_each_step=retrain_each_step,
                        )
                        st.success(f"Validation done: {stats['report_path']}")
                    except Exception as exc:
                        _safe_log("warning", "gui_validate_failed", error=str(exc))
                        st.error(f"Validation failed: {exc}")

            if run_all:
                with st.spinner("Running full pipeline..."):
                    try:
                        quantiles = _parse_quantiles(quantiles_text)
                        stats = _run_full_pipeline(
                            ingest_start=ingest_start,
                            ingest_end=ingest_end,
                            include_fx=include_fx,
                            include_fng=include_fng,
                            calendar_start=calendar_start,
                            calendar_end=calendar_end_effective,
                            horizons=horizons,
                            quantiles=quantiles,
                            alpha=float(alpha),
                        )
                        snapshot = stats["train"]["snapshot"]
                        st.session_state["selected_snapshot"] = snapshot
                        st.success(
                            "Full pipeline done: "
                            f"snapshot={snapshot}, "
                            f"features={stats['features']['feature_shape']}"
                        )
                        st.cache_data.clear()
                    except Exception as exc:
                        _safe_log("warning", "gui_full_pipeline_failed", error=str(exc))
                        st.error(f"Full pipeline failed: {exc}")

        st.markdown("---")
        snaps = _list_model_snapshots()
        snapshot_options = ["(none)"] + snaps
        if st.session_state["selected_snapshot"] not in snapshot_options:
            st.session_state["selected_snapshot"] = "(none)"
        snapshot = st.selectbox("Model snapshot", options=snapshot_options, key="selected_snapshot")

        st.subheader("Scenario shocks")
        shock_defs = {
            "DGS10": st.slider("10Y UST (bp)", -100, 100, 0, key="shock_dgs10") / 100.0,
            "DGS3MO": st.slider("3M UST (bp)", -100, 100, 0, key="shock_dgs3mo") / 100.0,
            "VIXCLS": st.slider("VIX (pts)", -20, 20, 0, key="shock_vix"),
            "DCOILWTICO": st.slider("WTI (USD)", -20, 20, 0, key="shock_wti"),
            "GOLDAMGBD228NLBM": st.slider("Gold (USD/oz)", -200, 200, 0, key="shock_gold"),
            "NFCI": st.slider("NFCI", -1.0, 1.0, 0.0, step=0.05, key="shock_nfci"),
            "CPIAUCSL": st.slider("CPI surprise", -1.0, 1.0, 0.0, step=0.05, key="shock_cpi"),
        }
        preset = st.selectbox("Preset scenario", options=["(none)"] + list(DEFAULT_SCENARIOS.keys()))

        st.caption(f"features.parquet: {'yes' if FEATURES_PATH.exists() else 'no'}")
        st.caption(f"targets.parquet: {'yes' if TARGETS_PATH.exists() else 'no'}")
        st.caption(f"latest snapshot: {_latest_snapshot() or '(none)'}")

    features_file = _load_features()
    features_live: Optional[pd.DataFrame] = None
    if use_live_features:
        try:
            features_live = _build_live_features_for_tab_a(
                start=live_start.strip() or "2000-01-01",
                end=live_end.strip() or None,
            )
        except Exception as exc:
            _safe_log("warning", "live_feature_build_failed", error=str(exc))
            st.sidebar.warning(f"Live feature build failed: {exc}")

    features = features_live if features_live is not None else features_file
    prices = _load_prices()
    if prices is None and features is not None and "SP500" in features.columns:
        prices = features[["SP500"]].dropna()
    daily_sp500 = _load_daily_sp500()

    with tab_a:
        st.subheader("Macro condition summary")
        if features is None:
            st.warning("No feature data found. Run GUI pipeline step 2 (Features).")
        else:
            if features_live is not None:
                st.caption("Data source: live fetch (cached 5 min)")
            else:
                st.caption("Data source: local features.parquet")

            last_row = _select_feature_row(features)
            score = float(last_row.get("MacroScore", 0.0))
            momentum = (
                float(last_row.filter(like="_d1_z").mean())
                if any(last_row.index.str.endswith("_d1_z"))
                else 0.0
            )
            acceleration = (
                float(last_row.filter(like="_d2_z").mean())
                if any(last_row.index.str.endswith("_d2_z"))
                else 0.0
            )
            render_macro_score_tile(score, momentum, acceleration)

            breadth = (
                float(last_row.get("MacroBreadth_top50", np.nan))
                if "MacroBreadth_top50" in features.columns
                else np.nan
            )
            breadth_delta = (
                float(last_row.get("MacroBreadth_top50_delta", 0.0))
                if "MacroBreadth_top50_delta" in features.columns
                else 0.0
            )
            if not np.isnan(breadth):
                render_breadth_bar(breadth, breadth_delta, "top 50%")

            regime_cols = [col for col in features.columns if col.startswith("Regime_")]
            if regime_cols:
                regimes = {col.replace("Regime_", ""): str(last_row.get(col)) for col in regime_cols}
                render_regime_badges(regimes)

            st.markdown("---")
            st.subheader("Key indicators")
            key_inds = [
                "VIXCLS",
                "DGS10",
                "DGS3MO",
                "TERM_SPREAD",
                "NFCI",
                "DCOILWTICO",
                "GOLDAMGBD228NLBM",
                "USDKRW",
                "UNRATE",
                "CPIAUCSL",
                "CRYPTO_FNG",
            ]
            cols = st.columns(3)
            for idx, indicator in enumerate(key_inds):
                if indicator not in features.columns:
                    continue
                level = float(last_row.get(indicator, np.nan))
                level_score = float(last_row.get(f"{indicator}_pctscore", np.nan))
                mom = float(last_row.get(f"{indicator}_d1", np.nan))
                acc = float(last_row.get(f"{indicator}_d2", np.nan))
                pct = (level_score + 1.0) / 2.0 if not np.isnan(level_score) else np.nan
                with cols[idx % 3]:
                    render_indicator_card(indicator, level, level_score, mom, acc, pct)

            changes: List[Dict[str, Any]] = []
            if not np.isnan(breadth_delta) and abs(breadth_delta) >= 0.1:
                changes.append(
                    {
                        "indicator": "MacroBreadth(top50)",
                        "direction": "up" if breadth_delta > 0 else "down",
                        "magnitude": breadth_delta * 100.0,
                        "context": "breadth momentum",
                    }
                )
            render_auto_summary(changes)

            st.markdown("---")
            st.subheader("Indicator narratives")
            render_indicator_narratives(key_inds, last_row)

            with st.expander("AI one-liner (optional)"):
                use_ai = st.checkbox(
                    "Use Gemini summary",
                    value=False,
                    help="Requires GEMINI_API_KEY or GOOGLE_API_KEY.",
                )
                if use_ai:
                    render_ai_one_liner(last_row, key_inds)

    with tab_b:
        st.subheader("S&P 500 fan chart")
        if snapshot == "(none)":
            st.warning("No model snapshot selected. Run GUI pipeline step 3 (Train).")
            return

        model_path = MODELS_DIR / snapshot / "models.pkl"
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return

        if features is None or prices is None:
            st.warning("Missing features or prices. Run GUI pipeline step 2 (Features).")
            return

        try:
            models = load_models(model_path)
        except Exception as exc:
            st.error(f"Model load failed: {exc}")
            return

        if not models:
            st.warning("Snapshot contains no trained model.")
            return

        x_t = _select_feature_row(features)
        custom_shocks = {key: value for key, value in shock_defs.items() if value != 0}
        shocks = _normalize_shocks(custom_shocks)

        if preset != "(none)" and preset in DEFAULT_SCENARIOS:
            preset_shocks = DEFAULT_SCENARIOS[preset].get("shocks", {})
            shocks.update(_normalize_shocks(preset_shocks))

        x_shocked = apply_shocks(x_t, shocks) if shocks else x_t
        current_price = float(prices.iloc[-1]["SP500"]) if not prices.empty else 0.0
        vix_val = float(x_shocked.get("VIXCLS", x_t.get("VIXCLS", 20.0)))

        fancharts = predict_fanchart(models, x_shocked, vix_current=vix_val)
        if not fancharts:
            st.warning("No fan chart result. Check model and feature inputs.")
            return

        keys = list(fancharts.keys())
        default_index = keys.index("1M") if "1M" in keys else 0
        horizon = st.selectbox("Horizon", options=keys, index=default_index)
        render_fan_chart(fancharts[horizon], prices, current_price, show_history=True)

        if daily_sp500 is not None and len(daily_sp500) >= 30:
            st.markdown("---")
            st.subheader("Daily Bollinger Bands")
            render_bollinger_bands(daily_sp500, window=20, num_std=2.0)

        quantiles = fancharts[horizon]["quantiles"]
        if all(key in quantiles for key in ("q10", "q90")):
            iqr_width = float(quantiles["q90"][0] - quantiles["q10"][0])
            historical_avg = 0.0
            if len(prices) > 24:
                returns = np.log(prices["SP500"]).diff(21).dropna()
                historical_avg = float(returns.rolling(24).std().dropna().mean() * 2)
            historical_avg = historical_avg or max(iqr_width, 1e-3)
            render_uncertainty_gauge(iqr_width, historical_avg)

        if horizon in models:
            delta = (x_shocked - x_t).fillna(0.0)
            contrib = compute_contributions(models[horizon], delta, quantile=0.5)
            if not contrib.empty:
                render_contribution_chart(contrib, top_n=5)

        normalized_defaults = {
            name: {**cfg, "shocks": _normalize_shocks(cfg.get("shocks", {}))}
            for name, cfg in DEFAULT_SCENARIOS.items()
        }
        with st.expander("Preset scenario comparison", expanded=False):
            try:
                comparison_df = create_scenario_comparison(
                    models,
                    x_t,
                    normalized_defaults,
                    horizons=keys,
                )
                if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
                    render_scenario_comparison(comparison_df)
                else:
                    st.info("No scenario comparison result.")
            except Exception as exc:
                st.info(f"Scenario comparison failed: {exc}")


if __name__ == "__main__":
    main()
