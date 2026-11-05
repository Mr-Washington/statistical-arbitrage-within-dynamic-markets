import pandas as pd
import pytest

from stat_arb.data.schema import bars_to_close_panel, validate_bars


def test_validate_and_forbid_bfill():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "instrument_id": ["A", "A"],
            "close": [1.0, 2.0],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "volume": [10, 10],
        }
    )
    v = validate_bars(df)
    assert list(v.columns)
    with pytest.raises(ValueError, match="bfill"):
        bars_to_close_panel(v, allow_bfill=True)
