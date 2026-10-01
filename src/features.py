from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data_loader import get_universe_as_of, latest_fundamentals_as_of, quality_mask


TRADING_DAYS_3M = 63
TRADING_DAYS_6M = 126
TRADING_DAYS_12M = 252
TRADING_DAYS_200DMA = 200
TRADING_DAYS_LIQUIDITY = 60


@dataclass
class RebalanceSnapshot:
    date: pd.Timestamp
    prior_date: pd.Timestamp
    eligible_frame: pd.DataFrame
    ranked_frame: pd.DataFrame


def pivot_price_table(prices: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    table = prices.pivot(index="date", columns="symbol", values=field).sort_index()
    return table


def prepare_rebalance_dates(prices: pd.DataFrame, start_date: str, end_date: str) -> List[pd.Timestamp]:
    date_index = pd.Index(sorted(prices["date"].dropna().unique()))
    window = date_index[(date_index >= pd.Timestamp(start_date)) & (date_index <= pd.Timestamp(end_date))]
    if len(window) == 0:
        return []
    date_series = pd.Series(window)
    first_trading_days = date_series.groupby(date_series.dt.to_period("M")).min().sort_values()
    return [pd.Timestamp(value) for value in first_trading_days.tolist()]


def compute_returns(close_history: pd.DataFrame, lookback: int) -> pd.Series:
    if len(close_history) <= lookback:
        return pd.Series(index=close_history.columns, dtype=float)
    latest = close_history.iloc[-1]
    earlier = close_history.iloc[-lookback - 1]
    returns = latest / earlier - 1.0
    return returns.replace([np.inf, -np.inf], np.nan)


def average_daily_traded_value(close_history: pd.DataFrame, volume_history: pd.DataFrame, window: int) -> pd.Series:
    traded_value = close_history * volume_history
    return traded_value.tail(window).mean()


def compute_sma(close_history: pd.DataFrame, window: int) -> pd.Series:
    return close_history.tail(window).mean()


def build_rebalance_snapshot(
    rebalance_date: pd.Timestamp,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_symbol: str,
    config: StrategyConfig,
) -> RebalanceSnapshot:
    close_table = pivot_price_table(prices, "adj_close")
    volume_table = pivot_price_table(prices, "volume")

    if rebalance_date not in close_table.index:
        raise ValueError(f"Rebalance date {rebalance_date.date()} is not present in price history.")

    date_position = close_table.index.get_loc(rebalance_date)
    if date_position == 0:
        raise ValueError("At least one prior trading day is required before the first rebalance.")

    prior_date = pd.Timestamp(close_table.index[date_position - 1])
    close_history = close_table.loc[:prior_date]
    volume_history = volume_table.loc[:prior_date]

    if len(close_history) < TRADING_DAYS_12M + 1:
        empty = pd.DataFrame()
        return RebalanceSnapshot(date=rebalance_date, prior_date=prior_date, eligible_frame=empty, ranked_frame=empty)

    universe_as_of = get_universe_as_of(universe, prior_date)
    current_fundamentals = latest_fundamentals_as_of(fundamentals, prior_date)
    if current_fundamentals.empty:
        empty = pd.DataFrame()
        return RebalanceSnapshot(date=rebalance_date, prior_date=prior_date, eligible_frame=empty, ranked_frame=empty)
    benchmark_returns_12m = compute_returns(close_history[[benchmark_symbol]].dropna(), TRADING_DAYS_12M)
    benchmark_12m = float(benchmark_returns_12m.iloc[0]) if not benchmark_returns_12m.empty else np.nan

    feature_frame = pd.DataFrame({"symbol": close_history.columns})
    feature_frame["momentum_12m"] = compute_returns(close_history, TRADING_DAYS_12M).reindex(feature_frame["symbol"]).values
    feature_frame["momentum_6m"] = compute_returns(close_history, TRADING_DAYS_6M).reindex(feature_frame["symbol"]).values
    feature_frame["momentum_3m"] = compute_returns(close_history, TRADING_DAYS_3M).reindex(feature_frame["symbol"]).values
    feature_frame["adv60_inr"] = average_daily_traded_value(close_history, volume_history, TRADING_DAYS_LIQUIDITY).reindex(
        feature_frame["symbol"]
    ).values
    feature_frame["sma_200"] = compute_sma(close_history, TRADING_DAYS_200DMA).reindex(feature_frame["symbol"]).values
    latest_close = close_history.iloc[-1]
    feature_frame["last_close"] = latest_close.reindex(feature_frame["symbol"]).values
    feature_frame["above_200dma"] = feature_frame["last_close"] > feature_frame["sma_200"]
    feature_frame["relative_strength"] = feature_frame["momentum_12m"] - benchmark_12m
    feature_frame = feature_frame.loc[~feature_frame["symbol"].str.startswith("^")].copy()

    merged = feature_frame.merge(universe_as_of[["symbol", "sector"]], on="symbol", how="inner")
    merged = merged.merge(current_fundamentals, on="symbol", how="left", suffixes=("", "_fund"))
    for column in [
        "market_cap",
        "roe",
        "debt_to_equity",
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "operating_cash_flow",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    for score_column in [
        "management_quality_score",
        "earnings_sentiment_score",
        "news_risk_score",
    ]:
        merged[score_column] = pd.to_numeric(merged[score_column], errors="coerce").fillna(0.5)

    merged["has_complete_prices"] = merged[["momentum_12m", "momentum_6m", "momentum_3m", "adv60_inr", "sma_200"]].notna().all(axis=1)
    merged["passes_liquidity"] = merged["adv60_inr"].gt(config.min_adv60_inr) & merged["has_complete_prices"]
    merged["passes_quality"] = quality_mask(merged, config)

    eligible = merged.loc[merged["passes_liquidity"] & merged["passes_quality"]].copy()
    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "momentum_12m",
            "momentum_6m",
            "momentum_3m",
            "relative_strength",
            "roe",
            "revenue_growth_yoy",
            "eps_growth_yoy",
            "debt_to_equity",
        ]
    )

    return RebalanceSnapshot(
        date=rebalance_date,
        prior_date=prior_date,
        eligible_frame=eligible.reset_index(drop=True),
        ranked_frame=merged.reset_index(drop=True),
    )
