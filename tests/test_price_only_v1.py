import pandas as pd

from src.config import StrategyConfig
from src.price_only_v1 import build_price_only_snapshot, determine_actual_start_date


def make_price_frame(symbols, dates, base_prices):
    records = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            price = base_prices[symbol] + i
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(records)


def test_price_only_snapshot_filters_by_market_cap_liquidity_and_dma():
    dates = pd.bdate_range("2020-01-01", periods=280)
    prices = make_price_frame(["AAA.NS", "BBB.NS", "^CRSLDX"], dates, {"AAA.NS": 100, "BBB.NS": 100, "^CRSLDX": 100})
    prices.loc[prices["symbol"] == "BBB.NS", "volume"] = 10
    universe = pd.DataFrame(
        {
            "symbol": ["AAA.NS", "BBB.NS"],
            "sector": ["Tech", "Banks"],
            "start_date": [dates[0], dates[0]],
            "end_date": [dates[-1], dates[-1]],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "effective_date": [dates[200], dates[200]],
            "symbol": ["AAA.NS", "BBB.NS"],
            "market_cap": [20_000_000_000, 20_000_000_000],
            "roe": [0.1, 0.1],
            "debt_to_equity": [0.1, 0.1],
            "revenue_growth_yoy": [0.1, 0.1],
            "eps_growth_yoy": [0.1, 0.1],
            "operating_cash_flow": [1.0, 1.0],
            "sector": ["Tech", "Banks"],
            "management_quality_score": [0.5, 0.5],
            "earnings_sentiment_score": [0.5, 0.5],
            "news_risk_score": [0.5, 0.5],
        }
    )
    snapshot = build_price_only_snapshot(dates[260], prices, universe, fundamentals, "^CRSLDX", StrategyConfig())
    assert snapshot["symbol"].tolist() == ["AAA.NS"]


def test_actual_start_date_respects_warmup_and_universe_start():
    universe = pd.DataFrame(
        {"symbol": ["AAA.NS"], "sector": ["Tech"], "start_date": ["2019-01-01"], "end_date": ["2025-12-31"]}
    )
    prices = pd.DataFrame(
        {
            "date": pd.bdate_range("2018-01-01", periods=300),
            "symbol": ["AAA.NS"] * 300,
            "open": [1] * 300,
            "high": [1] * 300,
            "low": [1] * 300,
            "close": [1] * 300,
            "adj_close": [1] * 300,
            "volume": [1] * 300,
        }
    )
    actual = determine_actual_start_date("2015-01-01", universe, prices)
    assert actual >= pd.Timestamp("2019-01-01")
