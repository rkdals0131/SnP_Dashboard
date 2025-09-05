import numpy as np
import pandas as pd

from src.models_quantile import train_models, predict_fanchart


def test_train_and_predict_fanchart():
    # Synthetic monthly data: 150 samples, 5 features
    n = 150
    idx = pd.date_range("2010-01-31", periods=n, freq="M")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=[f"x{i}" for i in range(5)])
    # Target = linear combo + noise
    beta = np.array([0.02, -0.01, 0.0, 0.015, -0.005])
    y1m = X.values @ beta + rng.normal(scale=0.02, size=n)
    y = pd.DataFrame({"return_1M": y1m}, index=idx)

    models = train_models(X, y, horizons=["1M"], quantiles=[0.1, 0.5, 0.9], config={"alpha": 0.1, "solver": "highs"})
    assert "1M" in models

    x_t = X.iloc[-1]
    fc = predict_fanchart(models, x_t, vix_current=20)
    assert "1M" in fc and set(["quantiles", "horizon"]).issubset(fc["1M"].keys())

