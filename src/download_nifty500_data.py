from __future__ import annotations

import argparse
from pathlib import Path

from .config import PathConfig, StrategyConfig
from .data_loader import YFinancePriceSource
from .real_data import (
    NIFTY_500_WIKIPEDIA_URL,
    build_dynamic_universe_from_prices,
    download_nifty_500_constituents,
    write_downloaded_dataset_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real OHLCV history for current NIFTY 500 constituents and benchmark indices."
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--universe-file", default=None)
    parser.add_argument("--prices-file", default=None)
    parser.add_argument("--metadata-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    paths = PathConfig.from_project_root(project_root)
    strategy = StrategyConfig()

    raw_universe = download_nifty_500_constituents()
    symbols = sorted(set(raw_universe["ticker"].tolist()) | set(strategy.benchmark_symbols.values()))
    downloader = YFinancePriceSource(cache_dir=paths.raw_data_dir / "prices")
    prices = downloader.download(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh=args.refresh_cache,
    )
    if prices.empty:
        raise ValueError("No price data was downloaded for the requested NIFTY 500 universe.")

    universe = build_dynamic_universe_from_prices(raw_universe, prices, args.end_date)
    universe_path = Path(args.universe_file) if args.universe_file else paths.universe_file
    prices_path = Path(args.prices_file) if args.prices_file else paths.prices_file
    metadata_path = Path(args.metadata_file) if args.metadata_file else paths.dataset_metadata_file

    universe_path.parent.mkdir(parents=True, exist_ok=True)
    prices_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(universe_path, index=False)
    prices.to_csv(prices_path, index=False)
    write_downloaded_dataset_metadata(
        metadata_path=metadata_path,
        universe_source=NIFTY_500_WIKIPEDIA_URL,
        price_source="yfinance",
        universe_start_date=str(universe["start_date"].min()),
        universe_end_date=None,
        notes=[
            "Universe membership is based on a current NIFTY 500 constituent snapshot, not a point-in-time historical membership series.",
            "This run remains exposed to survivorship bias and must not be described as survivorship-bias-reduced.",
            "Fundamentals are optional for Price-Only V1 when the market-cap filter is disabled.",
        ],
    )
    print(f"Saved universe to {universe_path}")
    print(f"Saved price history to {prices_path}")
    print(f"Saved dataset metadata to {metadata_path}")


if __name__ == "__main__":
    main()
