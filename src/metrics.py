from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def compute_drawdown(series: pd.Series) -> pd.Series:
    running_max = series.cummax()
    return series / running_max - 1.0


def annualized_return(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return float("nan")
    start_date = pd.Timestamp(series.index[0]) if isinstance(series.index, pd.Index) else None
    end_date = pd.Timestamp(series.index[-1]) if isinstance(series.index, pd.Index) else None
    if start_date is not None and end_date is not None and end_date > start_date:
        years = (end_date - start_date).days / 365.25
    else:
        years = (len(series) - 1) / 252.0
    if years <= 0:
        return float("nan")
    total_return = series.iloc[-1] / series.iloc[0]
    return total_return ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    if returns.dropna().empty:
        return float("nan")
    return returns.std(ddof=0) * np.sqrt(252.0)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    excess = clean - risk_free_rate / 252.0
    denominator = excess.std(ddof=0)
    if denominator == 0:
        return float("nan")
    return excess.mean() / denominator * np.sqrt(252.0)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    downside = clean[clean < risk_free_rate / 252.0] - risk_free_rate / 252.0
    downside_std = downside.std(ddof=0)
    if pd.isna(downside_std) or downside_std == 0:
        return float("nan")
    excess = clean.mean() - risk_free_rate / 252.0
    return excess / downside_std * np.sqrt(252.0)


def calmar_ratio(cagr: float, max_drawdown: float) -> float:
    if pd.isna(cagr) or pd.isna(max_drawdown) or max_drawdown == 0:
        return float("nan")
    return cagr / abs(max_drawdown)


def xnpv(rate: float, cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
    if rate <= -0.999999:
        return np.inf
    origin = cash_flows[0][0]
    return sum(value / ((1 + rate) ** ((date - origin).days / 365.0)) for date, value in cash_flows)


def xirr(cash_flows: List[Tuple[pd.Timestamp, float]]) -> float:
    if len(cash_flows) < 2:
        return float("nan")

    low, high = -0.9999, 10.0
    npv_low = xnpv(low, cash_flows)
    npv_high = xnpv(high, cash_flows)
    expansions = 0
    while npv_low * npv_high > 0 and expansions < 25:
        high *= 2
        npv_high = xnpv(high, cash_flows)
        expansions += 1
    if npv_low * npv_high > 0:
        return float("nan")

    for _ in range(200):
        mid = (low + high) / 2.0
        npv_mid = xnpv(mid, cash_flows)
        if abs(npv_mid) < 1e-8:
            return mid
        if npv_low * npv_mid <= 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
    return (low + high) / 2.0


def summarize_performance(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    monthly_performance: pd.DataFrame,
) -> Dict[str, float]:
    indexed_equity = equity_curve.copy()
    if "date" in indexed_equity.columns:
        indexed_equity = indexed_equity.set_index("date")

    daily_returns = indexed_equity["portfolio_value"].pct_change().fillna(0.0)
    cagr = annualized_return(indexed_equity["portfolio_value"])
    drawdown = compute_drawdown(indexed_equity["portfolio_value"])
    max_dd = float(drawdown.min()) if not drawdown.empty else float("nan")

    win_trades = trades.loc[trades["side"] == "SELL", "pnl"]
    monthly_returns = monthly_performance["monthly_return"].dropna()

    cash_flows = [
        (pd.Timestamp(row["date"]), -float(row["net_contribution"]))
        for _, row in equity_curve.loc[equity_curve["net_contribution"] > 0, ["date", "net_contribution"]].iterrows()
    ]
    if not equity_curve.empty:
        cash_flows.append((pd.Timestamp(equity_curve.iloc[-1]["date"]), float(equity_curve.iloc[-1]["portfolio_value"])))

    total_return = equity_curve["portfolio_value"].iloc[-1] / equity_curve["net_contribution"].sum() - 1.0 if equity_curve[
        "net_contribution"
    ].sum() else float("nan")
    turnover = trades["gross_value"].sum() / equity_curve["portfolio_value"].mean() if not trades.empty else 0.0

    return {
        "final_portfolio_value": float(equity_curve["portfolio_value"].iloc[-1]) if not equity_curve.empty else float("nan"),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "xirr": float(xirr(cash_flows)),
        "max_drawdown": max_dd,
        "volatility": float(annualized_volatility(daily_returns)),
        "sharpe_ratio": float(sharpe_ratio(daily_returns)),
        "sortino_ratio": float(sortino_ratio(daily_returns)),
        "calmar_ratio": float(calmar_ratio(cagr, max_dd)),
        "turnover": float(turnover),
        "number_of_trades": int(len(trades)),
        "win_rate": float((win_trades > 0).mean()) if not win_trades.empty else float("nan"),
        "best_month": float(monthly_returns.max()) if not monthly_returns.empty else float("nan"),
        "worst_month": float(monthly_returns.min()) if not monthly_returns.empty else float("nan"),
    }
