import pandas as pd
import pytest

from src.data_sources import FearGreedSource


def test_parse_alternative_payload_success():
    payload = {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": "42",
                "value_classification": "Fear",
                "timestamp": "1704067200",  # 2024-01-01 UTC
                "time_until_update": "0",
            },
            {
                "value": "55",
                "value_classification": "Greed",
                "timestamp": "1704153600",  # 2024-01-02 UTC
                "time_until_update": "0",
            },
        ],
        "metadata": {"error": None},
    }

    df = FearGreedSource._parse_alternative_payload(payload, "CRYPTO_FNG")
    assert not df.empty
    assert list(df["series_id"].unique()) == ["CRYPTO_FNG"]
    assert {"value", "value_classification", "realtime_start", "realtime_end", "source"}.issubset(df.columns)
    assert pd.Timestamp("2024-01-01") in df.index
    assert pd.Timestamp("2024-01-02") in df.index


def test_parse_alternative_payload_error():
    payload = {"data": [], "metadata": {"error": "limit reached"}}
    with pytest.raises(ValueError):
        FearGreedSource._parse_alternative_payload(payload, "CRYPTO_FNG")
