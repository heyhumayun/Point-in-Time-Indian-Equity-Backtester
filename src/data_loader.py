from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

from .config import PathConfig, StrategyConfig
from .universe_history import get_universe_on_date, load_universe_history
from .validation import ValidationError, validate_fundamentals_frame, validate_prices_frame, validate_universe_frame


PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
FUNDAMENTAL_COLUMNS = [
    "effective_date",
    "symbol",
    "market_cap",
    "roe",
    "debt_to_equity",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "operating_cash_flow",
    "sector",
    "management_quality_score",
    "earnings_sentiment_score",
    "news_risk_score",
]
UNIVERSE_COLUMNS = ["ticker", "company_name", "start_date", "end_date", "source", "notes", "sector", "symbol"]


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def ensure_directories(paths: PathConfig) -> None:
    for directory in [
        paths.raw_data_dir,
        paths.processed_data_dir,
        paths.outputs_dir,
        paths.reports_dir,
        paths.tests_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_universe(path: Path) -> pd.DataFrame:
    universe = load_universe_history(path)
    validate_universe_frame(universe)
    return universe


def get_universe_as_of(universe: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if universe.empty:
        return universe.copy()
    return get_universe_on_date(universe, as_of_date).drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def load_fundamentals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fundamentals file not found: {path}. Populate data/processed/fundamentals_pti.csv with point-in-time data."
        )

    fundamentals = pd.read_csv(path)
    required = {"effective_date", "symbol"}
    missing = required - set(fundamentals.columns)
    if missing:
        raise ValueError(f"Fundamentals file is missing required columns: {sorted(missing)}")

    fundamentals = fundamentals.copy()
    for column in FUNDAMENTAL_COLUMNS:
        if column not in fundamentals.columns:
            fundamentals[column] = pd.NA

    fundamentals["effective_date"] = pd.to_datetime(fundamentals["effective_date"])
    fundamentals["symbol"] = fundamentals["symbol"].map(normalize_symbol)
    for score_column in [
        "management_quality_score",
        "earnings_sentiment_score",
        "news_risk_score",
    ]:
        fundamentals[score_column] = pd.to_numeric(fundamentals[score_column], errors="coerce").fillna(0.5)
    fundamentals = fundamentals[FUNDAMENTAL_COLUMNS].sort_values(["symbol", "effective_date"]).reset_index(drop=True)
    validate_fundamentals_frame(fundamentals)
    return fundamentals


def latest_fundamentals_as_of(fundamentals: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if fundamentals.empty:
        return fundamentals.copy()

    eligible = fundamentals.loc[fundamentals["effective_date"] <= as_of_date].copy()
    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(["symbol", "effective_date"])
    latest = eligible.groupby("symbol", as_index=False).tail(1)
    return latest.reset_index(drop=True)


@dataclass
class YFinancePriceSource:
    cache_dir: Path

    def download(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise ImportError("yfinance is required to download price data. Install requirements.txt first.") from exc

        if yf is None:
            raise ImportError("yfinance is required to download price data. Install requirements.txt first.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        all_frames: List[pd.DataFrame] = []
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=5)

        symbols_to_download: List[str] = []
        for raw_symbol in sorted(set(symbols)):
            symbol = normalize_symbol(raw_symbol) if not raw_symbol.startswith("^") else raw_symbol
            cache_file = self.cache_dir / f"{symbol.replace('^', '_')}.csv"
            if cache_file.exists() and not refresh:
                history = pd.read_csv(cache_file, parse_dates=["date"])
                history["symbol"] = symbol
                all_frames.append(history[PRICE_COLUMNS].copy())
            else:
                symbols_to_download.append(symbol)

        batch_size = 40
        for batch_start in range(0, len(symbols_to_download), batch_size):
            batch_symbols = symbols_to_download[batch_start : batch_start + batch_size]
            try:
                batch_history = yf.download(
                    tickers=batch_symbols,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
            except Exception as exc:  # pragma: no cover - network/runtime behavior
                warnings.warn(
                    f"Batch download failed for {batch_symbols[0]}..{batch_symbols[-1]} ({exc}). Falling back to single-symbol downloads.",
                    stacklevel=2,
                )
                batch_history = pd.DataFrame()

            if isinstance(batch_history.columns, pd.MultiIndex) and not batch_history.empty:
                for symbol in batch_symbols:
                    if symbol not in batch_history.columns.get_level_values(0):
                        warnings.warn(f"No price history returned for {symbol}.", stacklevel=2)
                        continue
                    symbol_history = batch_history[symbol].dropna(how="all")
                    if symbol_history.empty:
                        warnings.warn(f"No price history returned for {symbol}.", stacklevel=2)
                        continue
                    history = _format_yfinance_history(symbol_history.reset_index(), symbol)
                    cache_file = self.cache_dir / f"{symbol.replace('^', '_')}.csv"
                    history.to_csv(cache_file, index=False)
                    all_frames.append(history.copy())
                continue

            if not batch_history.empty and len(batch_symbols) == 1:
                symbol = batch_symbols[0]
                history = _format_yfinance_history(batch_history.reset_index(), symbol)
                cache_file = self.cache_dir / f"{symbol.replace('^', '_')}.csv"
                history.to_csv(cache_file, index=False)
                all_frames.append(history.copy())
                continue

            for symbol in batch_symbols:
                ticker = yf.Ticker(symbol)
                history = ticker.history(start=start, end=end, auto_adjust=False)
                if history.empty:
                    warnings.warn(f"No price history returned for {symbol}.", stacklevel=2)
                    continue
                formatted = _format_yfinance_history(history.reset_index(), symbol)
                cache_file = self.cache_dir / f"{symbol.replace('^', '_')}.csv"
                formatted.to_csv(cache_file, index=False)
                all_frames.append(formatted.copy())

        if not all_frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        prices = pd.concat(all_frames, ignore_index=True)
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
        for column in numeric_columns:
            prices[column] = pd.to_numeric(prices[column], errors="coerce")
        return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def _format_yfinance_history(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    history = history.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    history["symbol"] = symbol
    history = history[PRICE_COLUMNS].copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.replace([np.inf, -np.inf], pd.NA).dropna(subset=["date", "open", "high", "low", "close", "adj_close", "volume"])
    return history.sort_values("date").reset_index(drop=True)


def load_price_frame(
    paths: PathConfig,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    refresh: bool = False,
    prices_file: Optional[str] = None,
    download_missing: bool = True,
) -> pd.DataFrame:
    requested_symbols = sorted(set(symbols))
    frames: List[pd.DataFrame] = []
    local_file = Path(prices_file) if prices_file else paths.prices_file

    if local_file.exists():
        local_prices = pd.read_csv(local_file, parse_dates=["date"])
        local_prices["symbol"] = local_prices["symbol"].map(normalize_symbol)
        local_prices = local_prices[local_prices["symbol"].isin(requested_symbols)].copy()
        validate_prices_frame(local_prices)
        frames.append(local_prices)

    local_symbols = set()
    if frames:
        local_symbols = set(pd.concat(frames, ignore_index=True)["symbol"].unique())

    missing_symbols = [symbol for symbol in requested_symbols if symbol not in local_symbols]
    if missing_symbols and download_missing:
        source = YFinancePriceSource(cache_dir=paths.raw_data_dir / "prices")
        downloaded = source.download(symbols=missing_symbols, start_date=start_date, end_date=end_date, refresh=refresh)
        if not downloaded.empty:
            frames.append(downloaded)

    if not frames:
        missing_path = local_file if prices_file or local_file.exists() else paths.raw_data_dir / "prices"
        raise FileNotFoundError(
            "No usable price data found. Populate "
            f"{missing_path} with historical prices or enable yfinance downloads with internet access."
        )

    prices = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "symbol"], keep="last")
    validate_prices_frame(prices)
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def prepare_benchmark_prices(
    prices: pd.DataFrame,
    benchmark_symbols: Dict[str, str],
) -> pd.DataFrame:
    benchmark_values = set(benchmark_symbols.values())
    benchmarks = prices.loc[prices["symbol"].isin(benchmark_values)].copy()
    return benchmarks.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_symbol_list(
    universe: pd.DataFrame,
    benchmark_symbols: Dict[str, str],
    override_symbols: Optional[List[str]] = None,
) -> List[str]:
    if override_symbols:
        tickers = [normalize_symbol(symbol) for symbol in override_symbols]
    else:
        source_column = "symbol" if "symbol" in universe.columns else "ticker"
        tickers = universe[source_column].dropna().astype(str).tolist()
    tickers.extend(benchmark_symbols.values())
    return sorted(set(tickers))


def quality_mask(frame: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    return (
        frame["market_cap"].gt(config.min_market_cap_inr)
        & frame["roe"].gt(config.min_roe)
        & frame["debt_to_equity"].lt(config.max_debt_to_equity)
        & frame["revenue_growth_yoy"].gt(config.min_revenue_growth)
        & frame["eps_growth_yoy"].gt(config.min_eps_growth)
        & frame["operating_cash_flow"].gt(0)
    )
