import pandas as pd

from src.config import StrategyConfig
from src.features import build_rebalance_snapshot, compute_returns


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


def test_momentum_calculation_uses_lookback_window():
    dates = pd.bdate_range("2020-01-01", periods=260)
    frame = pd.DataFrame({"AAA.NS": range(1, 261)}, index=dates)
    result = compute_returns(frame, 252)
    expected = 260 / 8 - 1
    assert round(result["AAA.NS"], 6) == round(expected, 6)


def test_liquidity_filter_excludes_low_adv_names():
    dates = pd.bdate_range("2020-01-01", periods=280)
    prices = make_price_frame(["AAA.NS", "^CRSLDX"], dates, {"AAA.NS": 100, "^CRSLDX": 100})
    prices.loc[prices["symbol"] == "AAA.NS", "volume"] = 10
    universe = pd.DataFrame(
        {"symbol": ["AAA.NS"], "sector": ["Tech"], "start_date": [dates[0]], "end_date": [dates[-1]]}
    )
    fundamentals = pd.DataFrame(
        {
            "effective_date": [dates[200]],
            "symbol": ["AAA.NS"],
            "market_cap": [20_000_000_000],
            "roe": [0.2],
            "debt_to_equity": [0.1],
            "revenue_growth_yoy": [0.2],
            "eps_growth_yoy": [0.2],
            "operating_cash_flow": [1.0],
            "sector": ["Tech"],
            "management_quality_score": [0.5],
            "earnings_sentiment_score": [0.5],
            "news_risk_score": [0.5],
        }
    )
    snapshot = build_rebalance_snapshot(dates[260], prices, universe, fundamentals, "^CRSLDX", StrategyConfig())
    assert snapshot.eligible_frame.empty


def test_snapshot_uses_prior_day_fundamentals_without_future_leakage():
    dates = pd.bdate_range("2020-01-01", periods=280)
    prices = make_price_frame(["AAA.NS", "^CRSLDX"], dates, {"AAA.NS": 100, "^CRSLDX": 100})
    universe = pd.DataFrame(
        {"symbol": ["AAA.NS"], "sector": ["Tech"], "start_date": [dates[0]], "end_date": [dates[-1]]}
    )
    fundamentals = pd.DataFrame(
        {
            "effective_date": [dates[250], dates[261]],
            "symbol": ["AAA.NS", "AAA.NS"],
            "market_cap": [20_000_000_000, 20_000_000_000],
            "roe": [0.16, 0.30],
            "debt_to_equity": [0.1, 0.1],
            "revenue_growth_yoy": [0.11, 0.30],
            "eps_growth_yoy": [0.11, 0.30],
            "operating_cash_flow": [1.0, 1.0],
            "sector": ["Tech", "Tech"],
            "management_quality_score": [0.5, 0.5],
            "earnings_sentiment_score": [0.5, 0.5],
            "news_risk_score": [0.5, 0.5],
        }
    )
    snapshot = build_rebalance_snapshot(dates[261], prices, universe, fundamentals, "^CRSLDX", StrategyConfig())
    assert snapshot.eligible_frame.iloc[0]["roe"] == 0.16


def test_snapshot_uses_adjusted_close_for_momentum():
    dates = pd.bdate_range("2020-01-01", periods=280)
    prices = make_price_frame(["AAA.NS", "^CRSLDX"], dates, {"AAA.NS": 100, "^CRSLDX": 100})
    prices.loc[prices["symbol"] == "AAA.NS", "adj_close"] = prices.loc[prices["symbol"] == "AAA.NS", "close"] * 2
    prices.loc[prices["symbol"] == "^CRSLDX", "adj_close"] = prices.loc[prices["symbol"] == "^CRSLDX", "close"]
    universe = pd.DataFrame(
        {"symbol": ["AAA.NS"], "sector": ["Tech"], "start_date": [dates[0]], "end_date": [dates[-1]]}
    )
    fundamentals = pd.DataFrame(
        {
            "effective_date": [dates[200]],
            "symbol": ["AAA.NS"],
            "market_cap": [20_000_000_000],
            "roe": [0.2],
            "debt_to_equity": [0.1],
            "revenue_growth_yoy": [0.2],
            "eps_growth_yoy": [0.2],
            "operating_cash_flow": [1.0],
            "sector": ["Tech"],
            "management_quality_score": [0.5],
            "earnings_sentiment_score": [0.5],
            "news_risk_score": [0.5],
        }
    )
    snapshot = build_rebalance_snapshot(dates[260], prices, universe, fundamentals, "^CRSLDX", StrategyConfig())
    assert snapshot.eligible_frame.iloc[0]["momentum_12m"] > 0
