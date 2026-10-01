import pandas as pd

from src.metrics import compute_drawdown


def test_drawdown_calculation():
    series = pd.Series([100, 120, 90, 95])
    drawdown = compute_drawdown(series)
    assert drawdown.tolist() == [0.0, 0.0, -0.25, -0.20833333333333337]
