import pandas as pd
import pytest

from src.backtester import simulate_benchmark
from src.dataset_metadata import create_dataset_metadata, ensure_research_dataset
from src.validation import ValidationError, build_validation_summary, validate_prices_frame


def test_validate_prices_rejects_duplicate_symbol_dates():
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "symbol": ["AAA.NS", "AAA.NS"],
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
            "adj_close": [100, 100],
            "volume": [1000, 1000],
        }
    )
    with pytest.raises(ValidationError):
        validate_prices_frame(prices)


def test_simulate_benchmark_does_not_double_count_initial_capital():
    dates = pd.bdate_range("2024-01-01", periods=5)
    benchmark_prices = pd.Series([100, 101, 102, 103, 104], index=dates)
    values = simulate_benchmark(benchmark_prices, [dates[0], dates[3]], 1000, 100)
    assert values.iloc[0] == 1000


def test_research_label_rejects_synthetic_dataset():
    metadata = create_dataset_metadata(
        dataset_type="synthetic",
        price_source="demo",
        universe_source="demo",
        fundamentals_source="demo",
    )
    with pytest.raises(ValidationError):
        ensure_research_dataset(metadata, report_as_research=True)


def test_build_validation_summary_reports_full_coverage():
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "adj_close": 100.0,
                "volume": 1000,
            }
            for date in dates
            for symbol in ["AAA.NS", "^CRSLDX", "^NSEI"]
        ]
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA.NS"],
            "sector": ["Tech"],
            "start_date": [dates[0]],
            "end_date": [dates[-1]],
        }
    )
    summary = build_validation_summary(
        universe=universe,
        prices=prices,
        benchmark_symbols={"nifty_50": "^NSEI", "nifty_500": "^CRSLDX"},
        start_date="2024-01-15",
        end_date="2024-02-09",
    )
    assert summary["ticker_coverage"]["coverage_pct"] == 100.0
    assert summary["missing_data_percentage"] == 0.0
    assert summary["eligible_ticker_count_by_rebalance_month"]
