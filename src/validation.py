from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


class ValidationError(ValueError):
    """Raised when an input dataset fails schema or coverage checks."""


UNIVERSE_REQUIRED_COLUMNS = ["ticker", "company_name", "start_date", "end_date", "source", "notes"]
FUNDAMENTAL_REQUIRED_COLUMNS = [
    "effective_date",
    "symbol",
    "market_cap",
    "roe",
    "debt_to_equity",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "operating_cash_flow",
]
PRICE_REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]


def _missing_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> List[str]:
    return [column for column in required_columns if column not in frame.columns]


def validate_universe_frame(universe: pd.DataFrame) -> None:
    missing = _missing_columns(universe, UNIVERSE_REQUIRED_COLUMNS)
    if missing:
        raise ValidationError(f"Universe data is missing required columns: {missing}")
    if universe.empty:
        raise ValidationError("Universe data is empty. Add at least one ticker interval.")
    if universe["ticker"].isna().any() or (universe["ticker"].astype(str).str.strip() == "").any():
        raise ValidationError("Universe contains blank tickers.")
    if pd.to_datetime(universe["start_date"], errors="coerce").isna().any():
        raise ValidationError("Universe contains missing or invalid start_date values.")


def validate_fundamentals_frame(fundamentals: pd.DataFrame) -> None:
    missing = _missing_columns(fundamentals, FUNDAMENTAL_REQUIRED_COLUMNS)
    if missing:
        raise ValidationError(f"Fundamentals data is missing required columns: {missing}")
    if fundamentals.empty:
        raise ValidationError("Fundamentals data is empty. Point-in-time fundamentals are required for the quality filter.")
    if fundamentals[["effective_date", "symbol"]].duplicated().any():
        raise ValidationError("Fundamentals contain duplicate effective_date/symbol rows.")
    for column in FUNDAMENTAL_REQUIRED_COLUMNS[2:]:
        if pd.to_numeric(fundamentals[column], errors="coerce").isna().any():
            raise ValidationError(f"Fundamentals column '{column}' contains missing or non-numeric values.")


def validate_prices_frame(prices: pd.DataFrame) -> None:
    missing = _missing_columns(prices, PRICE_REQUIRED_COLUMNS)
    if missing:
        raise ValidationError(f"Price data is missing required columns: {missing}")
    if prices.empty:
        raise ValidationError("Price data is empty.")
    if prices[["date", "symbol"]].duplicated().any():
        raise ValidationError("Price data contains duplicate date/symbol rows.")
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        if pd.to_numeric(prices[column], errors="coerce").isna().any():
            raise ValidationError(f"Price column '{column}' contains missing or non-numeric values.")
    if (pd.to_numeric(prices["volume"], errors="coerce") < 0).any():
        raise ValidationError("Price data contains negative volume.")


def validate_backtest_inputs(
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    benchmark_symbols: Dict[str, str],
    require_fundamentals: bool = True,
) -> List[str]:
    validate_universe_frame(universe)
    if require_fundamentals:
        validate_fundamentals_frame(fundamentals)
    validate_prices_frame(prices)

    warnings: List[str] = []
    backtest_start = pd.Timestamp(start_date)
    backtest_end = pd.Timestamp(end_date)
    warmup_start = backtest_start - pd.Timedelta(days=400)

    price_dates = pd.to_datetime(prices["date"])
    if price_dates.min() > warmup_start:
        warnings.append(
            f"Price history starts at {price_dates.min().date()}, which may be too late to compute a full 12-month momentum signal by {backtest_start.date()}."
        )
    if price_dates.max() < backtest_end:
        raise ValidationError(
            f"Price history ends at {price_dates.max().date()}, before requested end date {backtest_end.date()}."
        )

    universe_end_dates = pd.to_datetime(universe["end_date"], errors="coerce")
    universe_window = universe[
        (pd.to_datetime(universe["start_date"]) <= backtest_end)
        & (universe_end_dates.isna() | (universe_end_dates >= backtest_start))
    ]
    if universe_window.empty:
        raise ValidationError("Universe has no symbols active in the requested backtest window.")

    price_symbols = set(prices["symbol"].astype(str).unique())
    missing_benchmarks = [symbol for symbol in benchmark_symbols.values() if symbol not in price_symbols]
    if missing_benchmarks:
        raise ValidationError(
            f"Benchmark price history is missing for symbols: {missing_benchmarks}. Include benchmark rows in prices.csv or download them."
        )

    symbol_column = "symbol" if "symbol" in universe_window.columns else "ticker"
    universe_symbols = set(universe_window[symbol_column].astype(str).unique())
    if require_fundamentals:
        fundamentals_symbols = set(fundamentals["symbol"].astype(str).unique())
        uncovered_fundamentals = sorted(universe_symbols - fundamentals_symbols)
        if uncovered_fundamentals:
            preview = uncovered_fundamentals[:10]
            warnings.append(
                f"{len(uncovered_fundamentals)} universe symbols have no fundamentals coverage and will fail quality filters. Example: {preview}"
            )

    uncovered_prices = sorted(universe_symbols - price_symbols)
    if uncovered_prices:
        preview = uncovered_prices[:10]
        warnings.append(
            f"{len(uncovered_prices)} universe symbols have no price history and will be skipped. Example: {preview}"
        )

    return warnings


def build_validation_summary(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_symbols: Dict[str, str],
    start_date: str,
    end_date: str,
) -> Dict[str, object]:
    backtest_start = pd.Timestamp(start_date)
    backtest_end = pd.Timestamp(end_date)
    warmup_start = backtest_start - pd.Timedelta(days=400)

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    universe = universe.copy()
    universe["start_date"] = pd.to_datetime(universe["start_date"])
    universe["end_date"] = pd.to_datetime(universe["end_date"])

    non_benchmark_prices = prices.loc[~prices["symbol"].astype(str).str.startswith("^")].copy()
    price_dates = prices["date"]
    universe_window = universe[
        (universe["start_date"] <= backtest_end)
        & (universe["end_date"].isna() | (universe["end_date"] >= warmup_start))
    ].copy()
    symbol_column = "symbol" if "symbol" in universe_window.columns else "ticker"
    universe_symbols = sorted(universe_window[symbol_column].astype(str).unique())
    price_symbols = set(non_benchmark_prices["symbol"].astype(str).unique())
    ticker_coverage_count = sum(symbol in price_symbols for symbol in universe_symbols)
    ticker_coverage_pct = 100.0 * ticker_coverage_count / len(universe_symbols) if universe_symbols else 0.0

    benchmark_coverage = {}
    for name, symbol in benchmark_symbols.items():
        benchmark_frame = prices.loc[prices["symbol"] == symbol, ["date", "open", "close", "adj_close"]].copy()
        benchmark_coverage[name] = {
            "symbol": symbol,
            "rows": int(len(benchmark_frame)),
            "start_date": str(benchmark_frame["date"].min().date()) if not benchmark_frame.empty else None,
            "end_date": str(benchmark_frame["date"].max().date()) if not benchmark_frame.empty else None,
            "has_full_window": bool(
                not benchmark_frame.empty
                and benchmark_frame["date"].min() <= backtest_start
                and benchmark_frame["date"].max() >= backtest_end
            ),
        }

    benchmark_calendar_symbol = benchmark_symbols.get("nifty_500") or next(iter(benchmark_symbols.values()))
    benchmark_calendar = (
        prices.loc[
            (prices["symbol"] == benchmark_calendar_symbol)
            & (prices["date"] >= warmup_start)
            & (prices["date"] <= backtest_end),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
    )
    close_table = (
        non_benchmark_prices.pivot(index="date", columns="symbol", values="adj_close")
        .reindex(index=benchmark_calendar.tolist(), columns=universe_symbols)
        .sort_index()
    )
    open_table = (
        non_benchmark_prices.pivot(index="date", columns="symbol", values="open")
        .reindex(index=benchmark_calendar.tolist(), columns=universe_symbols)
        .sort_index()
    )

    active_mask = pd.DataFrame(False, index=close_table.index, columns=close_table.columns)
    for row in universe_window.itertuples(index=False):
        active_start = max(pd.Timestamp(row.start_date), warmup_start)
        row_end = pd.Timestamp(row.end_date) if pd.notna(row.end_date) else backtest_end
        active_end = min(row_end, backtest_end)
        if row.symbol in active_mask.columns:
            active_mask.loc[(active_mask.index >= active_start) & (active_mask.index <= active_end), row.symbol] = True

    active_cells = int(active_mask.to_numpy().sum())
    missing_close_cells = int((close_table.isna() & active_mask).to_numpy().sum())
    missing_data_percentage = 100.0 * missing_close_cells / active_cells if active_cells else 0.0

    rebalance_calendar = pd.Series([date for date in benchmark_calendar.tolist() if pd.Timestamp(date) >= backtest_start])
    rebalance_candidates = (
        rebalance_calendar.groupby([rebalance_calendar.dt.year, rebalance_calendar.dt.month]).min().tolist()
        if not rebalance_calendar.empty
        else []
    )
    rebalance_open_active = open_table.reindex(index=rebalance_candidates)
    rebalance_active_mask = active_mask.reindex(index=rebalance_candidates).fillna(False)
    rebalance_active_cells = int(rebalance_active_mask.to_numpy().sum())
    rebalance_missing_open_cells = int((rebalance_open_active.isna() & rebalance_active_mask).to_numpy().sum())
    rebalance_open_availability_pct = (
        100.0 * (rebalance_active_cells - rebalance_missing_open_cells) / rebalance_active_cells if rebalance_active_cells else 0.0
    )
    rebalance_counts = active_mask.reindex(index=rebalance_candidates).sum(axis=1)
    rebalance_count_changes = rebalance_counts.diff().fillna(0.0)
    rebalance_change_pct = (
        ((rebalance_counts - rebalance_counts.shift(1)) / rebalance_counts.shift(1).replace(0, pd.NA)) * 100.0
    ).fillna(0.0)

    return {
        "price_coverage": {
            "requested_warmup_start": str(warmup_start.date()),
            "requested_end_date": str(backtest_end.date()),
            "available_start_date": str(price_dates.min().date()),
            "available_end_date": str(price_dates.max().date()),
            "has_required_warmup": bool(price_dates.min() <= warmup_start),
            "has_required_end_date": bool(price_dates.max() >= backtest_end),
        },
        "ticker_coverage": {
            "expected_universe_tickers": len(universe_symbols),
            "tickers_with_prices": ticker_coverage_count,
            "coverage_pct": round(ticker_coverage_pct, 2),
        },
        "benchmark_coverage": benchmark_coverage,
        "missing_data_percentage": round(missing_data_percentage, 4),
        "rebalance_day_open_availability_pct": round(rebalance_open_availability_pct, 4),
        "eligible_ticker_count_by_rebalance_month": [
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "eligible_ticker_count": int(count),
                "count_change": float(change),
                "count_change_pct": round(float(change_pct), 4),
            }
            for date, count, change, change_pct in zip(rebalance_candidates, rebalance_counts.tolist(), rebalance_count_changes.tolist(), rebalance_change_pct.tolist())
        ],
    }


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)
