from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    side: str
    shares: int
    price: float
    gross_value: float
    transaction_cost: float
    net_cash_flow: float
    reason: str


@dataclass
class Position:
    symbol: str
    shares: int
    sector: str


def _weights_for_count(count: int, config: StrategyConfig) -> float:
    if count <= 0:
        return 0.0
    return min(1.0 / count, config.max_single_weight)


def select_target_portfolio(
    scored: pd.DataFrame,
    current_positions: Dict[str, Position],
    config: StrategyConfig,
) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    ranked = scored.sort_values(["score_rank", "symbol"]).copy()
    keep_mask = ranked["symbol"].isin(current_positions.keys()) & ranked["score_rank"].le(config.hold_rank_buffer) & ranked["above_200dma"]
    keepers = ranked.loc[keep_mask].copy()
    additions = ranked.loc[~ranked["symbol"].isin(keepers["symbol"]) & ranked["above_200dma"]].copy()

    selected = pd.concat([keepers, additions], ignore_index=True).head(config.rebalance_target_count).copy()
    if selected.empty:
        return selected

    while True:
        weight = _weights_for_count(len(selected), config)
        sector_exposure = selected.groupby("sector").size() * weight
        offending = sector_exposure[sector_exposure > config.max_sector_weight]
        if offending.empty:
            break
        worst_sector = offending.sort_values(ascending=False).index[0]
        drop_idx = selected.loc[selected["sector"] == worst_sector].sort_values("composite_score").index[0]
        selected = selected.drop(index=drop_idx).reset_index(drop=True)
        if selected.empty:
            break

    if selected.empty:
        return selected

    target_weight = _weights_for_count(len(selected), config)
    selected["target_weight"] = target_weight
    return selected.reset_index(drop=True)


def calculate_trade_cost(gross_value: float, transaction_cost_rate: float) -> float:
    return abs(gross_value) * transaction_cost_rate


def execute_rebalance(
    rebalance_date: pd.Timestamp,
    target_portfolio: pd.DataFrame,
    current_positions: Dict[str, Position],
    current_cash: float,
    prices_on_date: pd.Series,
    config: StrategyConfig,
) -> Tuple[Dict[str, Position], float, List[Trade], pd.DataFrame]:
    trades: List[Trade] = []
    updated_positions = {symbol: Position(symbol=pos.symbol, shares=pos.shares, sector=pos.sector) for symbol, pos in current_positions.items()}

    target_symbols = set(target_portfolio["symbol"].tolist())

    for symbol, position in list(updated_positions.items()):
        should_sell = symbol not in target_symbols
        if should_sell:
            price = float(prices_on_date.get(symbol, np.nan))
            if np.isnan(price) or position.shares <= 0:
                continue
            gross_value = position.shares * price
            transaction_cost = calculate_trade_cost(gross_value, config.transaction_cost_rate)
            net_cash_flow = gross_value - transaction_cost
            current_cash += net_cash_flow
            trades.append(
                Trade(
                    date=rebalance_date,
                    symbol=symbol,
                    side="SELL",
                    shares=position.shares,
                    price=price,
                    gross_value=gross_value,
                    transaction_cost=transaction_cost,
                    net_cash_flow=net_cash_flow,
                    reason="rank_or_trend_exit",
                )
            )
            del updated_positions[symbol]

    investable_equity = current_cash + sum(
        pos.shares * float(prices_on_date.get(symbol, 0.0))
        for symbol, pos in updated_positions.items()
        if not np.isnan(float(prices_on_date.get(symbol, np.nan)))
    )

    if target_portfolio.empty or investable_equity <= 0:
        holdings_frame = positions_to_frame(updated_positions, prices_on_date)
        return updated_positions, current_cash, trades, holdings_frame

    target_portfolio = target_portfolio.copy()
    target_portfolio["price"] = target_portfolio["symbol"].map(prices_on_date)
    target_portfolio = target_portfolio.dropna(subset=["price"])
    target_portfolio["target_value"] = target_portfolio["target_weight"] * investable_equity

    for _, row in target_portfolio.iterrows():
        symbol = row["symbol"]
        price = float(row["price"])
        if price <= 0:
            continue
        target_value = float(row["target_value"])
        current_shares = updated_positions.get(symbol, Position(symbol=symbol, shares=0, sector=row["sector"])).shares
        desired_shares = int(target_value // (price * (1.0 + config.transaction_cost_rate)))
        share_delta = desired_shares - current_shares
        if share_delta == 0:
            if symbol not in updated_positions:
                updated_positions[symbol] = Position(symbol=symbol, shares=current_shares, sector=row["sector"])
            continue

        gross_value = share_delta * price
        transaction_cost = calculate_trade_cost(gross_value, config.transaction_cost_rate)
        net_cash_flow = -(gross_value + transaction_cost)

        if share_delta > 0 and current_cash + net_cash_flow < -1e-9:
            affordable_shares = int(current_cash // (price * (1.0 + config.transaction_cost_rate)))
            share_delta = max(0, affordable_shares)
            gross_value = share_delta * price
            transaction_cost = calculate_trade_cost(gross_value, config.transaction_cost_rate)
            net_cash_flow = -(gross_value + transaction_cost)
        if share_delta == 0:
            continue

        side = "BUY" if share_delta > 0 else "SELL"
        if side == "SELL":
            net_cash_flow = abs(gross_value) - transaction_cost
            updated_shares = current_shares + share_delta
        else:
            updated_shares = current_shares + share_delta

        current_cash += net_cash_flow

        trades.append(
            Trade(
                date=rebalance_date,
                symbol=symbol,
                side=side,
                shares=abs(int(share_delta)),
                price=price,
                gross_value=abs(gross_value),
                transaction_cost=transaction_cost,
                net_cash_flow=net_cash_flow,
                reason="target_rebalance",
            )
        )

        if updated_shares <= 0:
            updated_positions.pop(symbol, None)
        else:
            updated_positions[symbol] = Position(symbol=symbol, shares=int(updated_shares), sector=row["sector"])

    holdings_frame = positions_to_frame(updated_positions, prices_on_date)
    return updated_positions, current_cash, trades, holdings_frame


def positions_to_frame(positions: Dict[str, Position], prices_on_date: pd.Series) -> pd.DataFrame:
    records = []
    for symbol, position in positions.items():
        price = float(prices_on_date.get(symbol, np.nan))
        records.append(
            {
                "symbol": symbol,
                "sector": position.sector,
                "shares": position.shares,
                "price": price,
                "market_value": position.shares * price if not np.isnan(price) else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values("symbol").reset_index(drop=True) if records else pd.DataFrame(
        columns=["symbol", "sector", "shares", "price", "market_value"]
    )
