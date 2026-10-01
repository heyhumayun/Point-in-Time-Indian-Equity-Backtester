from __future__ import annotations

import argparse
from pathlib import Path

from .config import PathConfig, StrategyConfig
from .data_loader import load_fundamentals, load_price_frame, load_universe
from .dataset_metadata import load_dataset_metadata
from .validation import build_validation_summary, validate_backtest_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate backtest input CSV schemas and date coverage")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--prices-file", default=None)
    parser.add_argument("--no-download", action="store_true", help="Validate only local CSV data without fetching missing prices.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    paths = PathConfig.from_project_root(project_root)
    strategy = StrategyConfig()

    universe = load_universe(paths.universe_file)
    fundamentals = load_fundamentals(paths.fundamentals_file)
    dataset_metadata = load_dataset_metadata(paths.dataset_metadata_file, universe["symbol"].astype(str).tolist())
    prices = load_price_frame(
        paths=paths,
        symbols=universe["symbol"].tolist() + list(strategy.benchmark_symbols.values()),
        start_date=args.start_date,
        end_date=args.end_date,
        prices_file=args.prices_file,
        download_missing=not args.no_download,
    )
    warnings = validate_backtest_inputs(
        universe=universe,
        fundamentals=fundamentals,
        prices=prices,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark_symbols=strategy.benchmark_symbols,
    )
    validation_summary = build_validation_summary(
        universe=universe,
        prices=prices,
        benchmark_symbols=strategy.benchmark_symbols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print("Validation passed.")
    print(f"Dataset type: {dataset_metadata.dataset_type}")
    print(f"Universe mode: {dataset_metadata.universe_mode}")
    print(f"Universe source quality: {dataset_metadata.universe_source_quality}")
    print(f"Survivorship bias status: {dataset_metadata.survivorship_bias_status}")
    print(f"Ticker coverage: {validation_summary['ticker_coverage']['coverage_pct']:.2f}%")
    print(f"Missing data percentage: {validation_summary['missing_data_percentage']:.4f}%")
    print(f"Rebalance-day open availability: {validation_summary['rebalance_day_open_availability_pct']:.4f}%")
    monthly_coverage = validation_summary.get("eligible_ticker_count_by_rebalance_month", [])
    if monthly_coverage:
        counts = [row["eligible_ticker_count"] for row in monthly_coverage]
        print(f"Eligible tickers by rebalance month: min={min(counts)}, max={max(counts)}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
