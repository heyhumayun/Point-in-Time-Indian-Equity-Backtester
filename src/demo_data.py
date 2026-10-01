from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PathConfig
from .dataset_metadata import create_dataset_metadata, write_dataset_metadata


DEMO_SYMBOLS = [
    ("ALPHA.NS", "Technology"),
    ("BETA.NS", "Technology"),
    ("GAMMA.NS", "Financials"),
    ("DELTA.NS", "Financials"),
    ("EPSILON.NS", "Industrials"),
    ("ZETA.NS", "Industrials"),
    ("ETA.NS", "Healthcare"),
    ("THETA.NS", "Healthcare"),
    ("IOTA.NS", "Consumer"),
    ("KAPPA.NS", "Consumer"),
    ("LAMBDA.NS", "Energy"),
    ("MU.NS", "Energy"),
    ("NU.NS", "Materials"),
    ("XI.NS", "Materials"),
    ("OMICRON.NS", "Utilities"),
    ("PI.NS", "Utilities"),
    ("RHO.NS", "Realty"),
    ("SIGMA.NS", "Telecom"),
]


def make_demo_prices(start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.bdate_range(start_date, end_date)
    records = []
    benchmark_map = {"^NSEI": 0.00045, "^CRSLDX": 0.00050}

    for i, (symbol, sector) in enumerate(DEMO_SYMBOLS):
        seed = 100 + i
        rng = np.random.default_rng(seed)
        drift = 0.0005 + (i % 6) * 0.00008
        vol = 0.012 + (i % 4) * 0.0015
        prices = [120 + i * 7]
        for _ in range(1, len(dates)):
            seasonal = 0.0015 * np.sin(len(prices) / 21.0 + i)
            shock = rng.normal(drift + seasonal, vol)
            prices.append(max(15.0, prices[-1] * (1.0 + shock)))

        series = pd.Series(prices, index=dates)
        opens = series.shift(1).fillna(series.iloc[0] * 0.995) * (1.0 + rng.normal(0.0002, 0.002, len(dates)))
        highs = pd.concat([opens, series], axis=1).max(axis=1) * (1.0 + 0.004)
        lows = pd.concat([opens, series], axis=1).min(axis=1) * (1.0 - 0.004)
        volume = (rng.normal(900_000 + i * 25_000, 50_000, len(dates))).clip(min=250_000)

        for date in dates:
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": round(float(opens.loc[date]), 2),
                    "high": round(float(highs.loc[date]), 2),
                    "low": round(float(lows.loc[date]), 2),
                    "close": round(float(series.loc[date]), 2),
                    "adj_close": round(float(series.loc[date]), 2),
                    "volume": int(volume[dates.get_loc(date)]),
                }
            )

    for symbol, drift in benchmark_map.items():
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        prices = [10_000 if symbol == "^NSEI" else 12_000]
        for _ in range(1, len(dates)):
            shock = rng.normal(drift, 0.008 if symbol == "^NSEI" else 0.007)
            prices.append(max(1_000.0, prices[-1] * (1.0 + shock)))
        series = pd.Series(prices, index=dates)
        opens = series.shift(1).fillna(series.iloc[0] * 0.996) * (1.0 + rng.normal(0.0001, 0.0015, len(dates)))
        highs = pd.concat([opens, series], axis=1).max(axis=1) * (1.0 + 0.003)
        lows = pd.concat([opens, series], axis=1).min(axis=1) * (1.0 - 0.003)
        volume = (rng.normal(0, 1, len(dates)) * 0 + 1).astype(int)
        for date in dates:
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": round(float(opens.loc[date]), 2),
                    "high": round(float(highs.loc[date]), 2),
                    "low": round(float(lows.loc[date]), 2),
                    "close": round(float(series.loc[date]), 2),
                    "adj_close": round(float(series.loc[date]), 2),
                    "volume": int(volume[dates.get_loc(date)]),
                }
            )

    return pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)


def make_demo_universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol for symbol, _ in DEMO_SYMBOLS],
            "sector": [sector for _, sector in DEMO_SYMBOLS],
            "start_date": ["2019-01-01"] * len(DEMO_SYMBOLS),
            "end_date": ["2025-12-31"] * len(DEMO_SYMBOLS),
        }
    )


def make_demo_fundamentals() -> pd.DataFrame:
    effective_dates = pd.date_range("2018-12-31", "2025-12-31", freq="Q")
    rows = []
    for i, (symbol, sector) in enumerate(DEMO_SYMBOLS):
        for j, effective_date in enumerate(effective_dates):
            rows.append(
                {
                    "effective_date": effective_date.date().isoformat(),
                    "symbol": symbol,
                    "market_cap": 15_000_000_000 + i * 2_000_000_000 + j * 150_000_000,
                    "roe": round(0.16 + (i % 5) * 0.015 + (j % 3) * 0.002, 4),
                    "debt_to_equity": round(0.08 + (i % 4) * 0.08, 4),
                    "revenue_growth_yoy": round(0.11 + (i % 6) * 0.02 + (j % 2) * 0.005, 4),
                    "eps_growth_yoy": round(0.12 + (i % 5) * 0.018 + (j % 2) * 0.004, 4),
                    "operating_cash_flow": 500_000_000 + i * 40_000_000 + j * 5_000_000,
                    "sector": sector,
                    "management_quality_score": round(0.45 + (i % 5) * 0.06, 2),
                    "earnings_sentiment_score": round(0.44 + (j % 4) * 0.04, 2),
                    "news_risk_score": round(0.54 - (i % 3) * 0.04, 2),
                }
            )
    return pd.DataFrame(rows)


def write_demo_dataset(project_root: Path, start_date: str, end_date: str) -> None:
    paths = PathConfig.from_project_root(project_root)
    paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    make_demo_universe().to_csv(paths.universe_file, index=False)
    make_demo_fundamentals().to_csv(paths.fundamentals_file, index=False)
    make_demo_prices(start_date=start_date, end_date=end_date).to_csv(paths.prices_file, index=False)
    write_dataset_metadata(
        paths.dataset_metadata_file,
        create_dataset_metadata(
            dataset_type="synthetic",
            price_source="src.demo_data",
            universe_source="src.demo_data",
            fundamentals_source="src.demo_data",
            universe_mode="synthetic",
            universe_snapshot_count=0,
            universe_snapshot_frequency="synthetic",
            survivorship_bias_status="unknown",
            universe_start_date=start_date,
            universe_end_date=end_date,
            notes=["Synthetic demo data is only for pipeline validation and must not be used for performance conclusions."],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a demo dataset for the Indian equity backtester")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    write_demo_dataset(project_root, args.start_date, args.end_date)
    print("Demo dataset written to data/processed/.")


if __name__ == "__main__":
    main()
