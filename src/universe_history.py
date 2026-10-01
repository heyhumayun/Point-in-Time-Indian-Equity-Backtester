from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .validation import ValidationError


UNIVERSE_HISTORY_REQUIRED_COLUMNS = ["ticker", "company_name", "start_date", "end_date", "source", "notes"]


def load_universe_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}. Populate data/processed/universe.csv before running the backtest.")

    universe = pd.read_csv(path, keep_default_na=False).copy()
    if "ticker" not in universe.columns and "symbol" in universe.columns:
        universe["ticker"] = universe["symbol"]
    if "company_name" not in universe.columns:
        universe["company_name"] = universe.get("ticker", pd.Series(dtype=str)).astype(str)
    if "source" not in universe.columns:
        universe["source"] = "legacy_universe_file"
    if "notes" not in universe.columns:
        universe["notes"] = ""
    if "sector" not in universe.columns:
        universe["sector"] = "Unknown"

    missing = [column for column in UNIVERSE_HISTORY_REQUIRED_COLUMNS if column not in universe.columns]
    if missing:
        raise ValidationError(f"Universe history is missing required columns: {missing}")

    universe["ticker"] = universe["ticker"].astype(str).str.strip().map(_normalize_ticker)
    universe["symbol"] = universe["ticker"]
    universe["company_name"] = universe["company_name"].astype(str).str.strip()
    universe["source"] = universe["source"].astype(str).str.strip()
    universe["notes"] = universe["notes"].astype(str)
    universe["sector"] = universe["sector"].replace("", "Unknown").fillna("Unknown")
    universe["start_date"] = pd.to_datetime(universe["start_date"], errors="coerce")
    universe["end_date"] = pd.to_datetime(universe["end_date"].replace("", pd.NA), errors="coerce")
    validate_universe_intervals(universe)
    columns = UNIVERSE_HISTORY_REQUIRED_COLUMNS + ["sector", "symbol"]
    return universe[columns].sort_values(["ticker", "start_date", "end_date"], na_position="last").reset_index(drop=True)


def get_universe_on_date(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    universe_df = _coerce_universe_columns(universe_df.copy())
    as_of_date = pd.Timestamp(rebalance_date)
    mask = (universe_df["start_date"] <= as_of_date) & (universe_df["end_date"].isna() | (universe_df["end_date"] >= as_of_date))
    return universe_df.loc[mask].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def validate_universe_intervals(universe_df: pd.DataFrame) -> None:
    universe_df = _coerce_universe_columns(universe_df.copy())
    if universe_df.empty:
        raise ValidationError("Universe data is empty. Add at least one ticker interval.")
    if universe_df["ticker"].isna().any() or (universe_df["ticker"].astype(str).str.strip() == "").any():
        raise ValidationError("Universe contains blank tickers.")
    if universe_df["start_date"].isna().any():
        raise ValidationError("Universe contains missing or invalid start_date values.")
    if universe_df[["ticker", "start_date", "end_date"]].duplicated().any():
        raise ValidationError("Universe contains duplicate ticker/start_date/end_date intervals.")
    invalid_end = universe_df["end_date"].notna() & (universe_df["end_date"] < universe_df["start_date"])
    if invalid_end.any():
        raise ValidationError("Universe contains intervals where end_date is before start_date.")

    for ticker, frame in universe_df.sort_values(["ticker", "start_date", "end_date"], na_position="last").groupby("ticker", sort=False):
        rows = frame.reset_index(drop=True)
        for idx in range(1, len(rows)):
            prior_end = rows.loc[idx - 1, "end_date"]
            current_start = rows.loc[idx, "start_date"]
            if pd.isna(prior_end):
                raise ValidationError(f"Ticker {ticker} has an open-ended interval followed by another interval.")
            if current_start <= prior_end:
                raise ValidationError(f"Ticker {ticker} has overlapping intervals around {current_start.date()}.")


def summarize_universe_coverage(universe_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    universe_df = _coerce_universe_columns(universe_df.copy())
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    month_starts = pd.date_range(start, end, freq="MS")
    records: List[Dict[str, object]] = []
    for month_start in month_starts:
        eligible = get_universe_on_date(universe_df, month_start)
        records.append(
            {
                "date": month_start.date().isoformat(),
                "eligible_ticker_count": int(len(eligible)),
                "sample_tickers": ",".join(eligible["ticker"].head(10).tolist()),
            }
        )
    return pd.DataFrame(records)


def _coerce_universe_columns(universe_df: pd.DataFrame) -> pd.DataFrame:
    if "ticker" not in universe_df.columns and "symbol" in universe_df.columns:
        universe_df["ticker"] = universe_df["symbol"]
    if "symbol" not in universe_df.columns and "ticker" in universe_df.columns:
        universe_df["symbol"] = universe_df["ticker"].astype(str).map(_normalize_ticker)
    if "company_name" not in universe_df.columns:
        universe_df["company_name"] = universe_df["ticker"]
    if "source" not in universe_df.columns:
        universe_df["source"] = "in_memory"
    if "notes" not in universe_df.columns:
        universe_df["notes"] = ""
    if "sector" not in universe_df.columns:
        universe_df["sector"] = "Unknown"
    universe_df["start_date"] = pd.to_datetime(universe_df["start_date"], errors="coerce")
    universe_df["end_date"] = pd.to_datetime(universe_df["end_date"].replace("", pd.NA), errors="coerce")
    return universe_df


def _normalize_ticker(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"
