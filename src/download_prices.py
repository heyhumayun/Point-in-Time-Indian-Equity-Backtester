from __future__ import annotations

import argparse
from pathlib import Path

from .config import PathConfig, StrategyConfig
from .data_loader import build_symbol_list, load_price_frame, load_universe
from .dataset_metadata import create_dataset_metadata, write_dataset_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or refresh OHLCV history into data/processed/prices.csv")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    paths = PathConfig.from_project_root(project_root)
    strategy = StrategyConfig()

    universe = load_universe(paths.universe_file)
    symbols = build_symbol_list(universe, strategy.benchmark_symbols)
    prices = load_price_frame(
        paths=paths,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh=args.refresh_cache,
        prices_file=None,
        download_missing=True,
    )
    output_path = Path(args.output_file) if args.output_file else paths.prices_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
    write_dataset_metadata(
        paths.dataset_metadata_file,
        create_dataset_metadata(
            dataset_type="downloaded",
            price_source="yfinance",
            universe_source=str(paths.universe_file),
            fundamentals_source=str(paths.fundamentals_file),
            universe_mode="unknown",
            universe_snapshot_count=0,
            universe_snapshot_frequency="unknown",
            survivorship_bias_status="unknown",
            notes=["Prices were refreshed from Yahoo Finance for the active universe and benchmark symbols."],
        ),
    )
    print(f"Saved price history to {output_path}")


if __name__ == "__main__":
    main()
