from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series.rank(pct=True, ascending=ascending, method="average")
    return ranked.fillna(0.0)


def apply_composite_score(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        result["composite_score"] = []
        result["score_rank"] = []
        return result

    scored = frame.copy()
    scored["debt_score"] = 1.0 - percentile_rank(scored["debt_to_equity"], ascending=True)

    score_components = {}
    for column, weight in config.composite_weights.items():
        if column == "debt_score":
            component = scored["debt_score"]
        else:
            component = percentile_rank(scored[column], ascending=True)
        score_components[column] = component * weight

    composite = sum(score_components.values())
    base_total = float(sum(config.composite_weights.values()))
    scored["base_score"] = composite / base_total

    ai_signal = (
        (scored["management_quality_score"] - 0.5)
        + (scored["earnings_sentiment_score"] - 0.5)
        + ((1.0 - scored["news_risk_score"]) - 0.5)
    ) / 3.0
    adjustment = np.clip(ai_signal * 0.2, -config.ai_adjustment_cap, config.ai_adjustment_cap)
    scored["ai_adjustment"] = adjustment
    scored["composite_score"] = scored["base_score"] * (1.0 + adjustment)
    scored["score_rank"] = scored["composite_score"].rank(ascending=False, method="first").astype(int)
    return scored.sort_values(["composite_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
