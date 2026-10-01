import pandas as pd

from src.config import StrategyConfig
from src.portfolio import Position, calculate_trade_cost, execute_rebalance, select_target_portfolio


def test_transaction_cost_calculation():
    assert calculate_trade_cost(100000, 0.002) == 200.0


def test_rebalance_logic_sells_exits_and_buys_targets():
    config = StrategyConfig()
    current_positions = {"OLD.NS": Position(symbol="OLD.NS", shares=10, sector="Tech")}
    target = pd.DataFrame(
        {
            "symbol": ["NEW.NS"],
            "sector": ["Banks"],
            "target_weight": [0.1],
            "score_rank": [1],
            "composite_score": [1.0],
            "above_200dma": [True],
        }
    )
    prices = pd.Series({"OLD.NS": 100.0, "NEW.NS": 50.0})
    updated_positions, cash, trades, holdings = execute_rebalance(
        pd.Timestamp("2024-01-01"),
        target,
        current_positions,
        current_cash=10000.0,
        prices_on_date=prices,
        config=config,
    )
    assert "OLD.NS" not in updated_positions
    assert "NEW.NS" in updated_positions
    assert any(trade.side == "SELL" for trade in trades)
    assert any(trade.side == "BUY" for trade in trades)
    assert not holdings.empty


def test_sector_cap_keeps_selection_within_limit():
    config = StrategyConfig(max_sector_weight=0.25)
    scored = pd.DataFrame(
        {
            "symbol": [f"A{i}.NS" for i in range(6)],
            "sector": ["Tech", "Tech", "Tech", "Banks", "Banks", "Energy"],
            "score_rank": [1, 2, 3, 4, 5, 6],
            "composite_score": [0.9, 0.85, 0.8, 0.79, 0.78, 0.77],
            "above_200dma": [True] * 6,
        }
    )
    selected = select_target_portfolio(scored, {}, config)
    weight = selected["target_weight"].iloc[0]
    assert (selected.groupby("sector").size() * weight).max() <= 0.25


def test_rank_buffer_keeps_current_holding_beyond_top_target_count():
    config = StrategyConfig(rebalance_target_count=2, hold_rank_buffer=4, max_sector_weight=1.0)
    current_positions = {"KEEP.NS": Position(symbol="KEEP.NS", shares=10, sector="Tech")}
    scored = pd.DataFrame(
        {
            "symbol": ["A.NS", "B.NS", "KEEP.NS", "C.NS"],
            "sector": ["Tech", "Banks", "Tech", "Energy"],
            "score_rank": [1, 2, 3, 4],
            "composite_score": [0.95, 0.90, 0.89, 0.80],
            "above_200dma": [True, True, True, True],
        }
    )
    selected = select_target_portfolio(scored, current_positions, config)
    assert "KEEP.NS" in selected["symbol"].tolist()
