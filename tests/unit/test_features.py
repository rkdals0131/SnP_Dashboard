import numpy as np
import pandas as pd

from src.features import make_deltas, make_delta_squared, percentile_score_10y, compute_z_scores


def test_make_deltas_and_delta2():
    idx = pd.date_range("2020-01-31", periods=5, freq="M")
    df = pd.DataFrame({"X": [1, 3, 6, 10, 15]}, index=idx)
    out = make_deltas(df, ["X"]).pipe(make_delta_squared, ["X"])  # Δ, Δ²
    assert np.allclose(out["X_d1"].tolist(), [np.nan, 2, 3, 4, 5], equal_nan=True)
    assert np.allclose(out["X_d2"].tolist(), [np.nan, np.nan, 1, 1, 1], equal_nan=True)


def test_percentile_score_basic():
    idx = pd.date_range("2010-01-31", periods=121, freq="M")
    # Linear ramp ensures percentiles increase
    df = pd.DataFrame({"X": np.linspace(0, 120, 121)}, index=idx)
    out = percentile_score_10y(df, ["X"], window_years=10, min_periods=12)
    scores = out["X_pctscore"].dropna()
    assert scores.min() >= -1 - 1e-9 and scores.max() <= 1 + 1e-9
    # Last value near +1
    assert scores.iloc[-1] > 0.8

