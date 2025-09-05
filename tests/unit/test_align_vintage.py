import pandas as pd
import numpy as np
from pathlib import Path

from src.align_vintage import create_month_end_calendar, build_master_panel, apply_publication_delays, PUBLICATION_RULES


def test_create_month_end_calendar():
    cal = create_month_end_calendar("2020-01-01", "2020-06-30")
    # Month-end dates
    assert len(cal) == 6
    assert cal[0].month == 1 and cal[-1].month == 6


def test_build_master_panel_and_vintage(tmp_path: Path):
    # Create two simple series with daily dates
    days = pd.date_range("2020-01-01", periods=10, freq="B")
    vix = pd.DataFrame({"series_id": "VIXCLS", "value": np.arange(10)}, index=days)
    dgs10 = pd.DataFrame({"series_id": "DGS10", "value": np.linspace(1, 2, 10)}, index=days)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    vix_path = raw_dir / "VIXCLS.parquet"
    dgs10_path = raw_dir / "DGS10.parquet"
    vix.to_parquet(vix_path)
    dgs10.to_parquet(dgs10_path)

    cal = create_month_end_calendar("2020-01-01", "2020-02-29")
    master = build_master_panel({"VIXCLS": vix_path, "DGS10": dgs10_path}, cal)
    assert set(["VIXCLS", "DGS10"]).issubset(set(master.columns))
    # availability flags
    assert "VIXCLS_is_avail" in master.columns

    # Apply publication delays
    vintage = apply_publication_delays(master, PUBLICATION_RULES)
    # For 1BD delay on month-end calendar, later months might be NA until delay date
    assert vintage.shape == master.shape

