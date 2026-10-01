import pandas as pd

from src.config import StrategyConfig
from src.scoring import apply_composite_score


def test_composite_scoring_ranks_stronger_name_first():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA.NS", "BBB.NS"],
            "momentum_12m": [0.5, 0.2],
            "momentum_6m": [0.3, 0.1],
            "momentum_3m": [0.2, 0.05],
            "relative_strength": [0.2, 0.01],
            "roe": [0.25, 0.18],
            "revenue_growth_yoy": [0.2, 0.12],
            "eps_growth_yoy": [0.22, 0.11],
            "debt_to_equity": [0.1, 0.4],
            "management_quality_score": [0.5, 0.5],
            "earnings_sentiment_score": [0.5, 0.5],
            "news_risk_score": [0.5, 0.5],
        }
    )
    scored = apply_composite_score(frame, StrategyConfig())
    assert scored.iloc[0]["symbol"] == "AAA.NS"
    assert scored.iloc[0]["score_rank"] == 1
