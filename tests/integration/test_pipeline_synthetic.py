import numpy as np
import pandas as pd

from src.features import build_feature_panel
from src.targets import compute_forward_returns, align_targets_with_features
from src.models_quantile import train_models
from src.validation import run_rolling_validation, ValidationConfig


def test_end_to_end_pipeline_synthetic():
    # Synthetic monthly panel with key series: DGS10, DGS3MO, VIXCLS, SP500
    n = 132
    idx = pd.date_range("2010-01-31", periods=n, freq="M")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "DGS10": 2.0 + 0.2 * np.sin(np.linspace(0, 6, n)) + rng.normal(0, 0.05, n),
            "DGS3MO": 0.5 + 0.1 * np.cos(np.linspace(0, 6, n)) + rng.normal(0, 0.03, n),
            "VIXCLS": 15 + 3 * np.sin(np.linspace(0, 10, n)) + rng.normal(0, 1.0, n),
            "SP500": np.cumprod(1 + 0.005 + 0.02 * rng.normal(size=n)) * 1000,
        },
        index=idx,
    )

    features = build_feature_panel(panel)
    targets = compute_forward_returns(panel[["SP500"]], "SP500", ["1M"])  # keep small
    data = align_targets_with_features(targets, features)

    # Train a lightweight model
    feature_cols = [c for c in data.columns if not c.startswith("return_") and not c.endswith("_is_avail")]
    X = data[feature_cols]
    y = data[["return_1M"]]
    models = train_models(X, y, horizons=["1M"], quantiles=[0.1, 0.5, 0.9], config={"alpha": 0.1, "solver": "highs"})
    assert "1M" in models

    # Validation (no retrain for speed in test)
    cfg = ValidationConfig(horizons=["1M"], quantiles=[0.1, 0.5, 0.9], retrain_each_step=False)
    res = run_rolling_validation(models, X, y, train_window=60, test_window=6, step_size=6, config=cfg)
    assert "summary" in res and "1M" in res["summary"]

