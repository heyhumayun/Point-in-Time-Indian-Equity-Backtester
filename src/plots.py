from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8")


def plot_equity_curve(equity_curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in ["portfolio_value", "benchmark_nifty_50", "benchmark_nifty_500"]:
        if column in equity_curve.columns:
            ax.plot(equity_curve["date"], equity_curve[column], label=column.replace("_", " ").title(), linewidth=2)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown(equity_curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(equity_curve["date"], equity_curve["drawdown"], 0, color="#c44e52", alpha=0.35)
    ax.plot(equity_curve["date"], equity_curve["drawdown"], color="#c44e52", linewidth=1.5)
    ax.set_title("Drawdown Curve")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_monthly_returns(monthly_performance: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(monthly_performance["date"], monthly_performance["monthly_return"], color="#4c72b0")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Monthly Returns")
    ax.set_ylabel("Return")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sector_allocation(holdings_history: pd.DataFrame, output_path: Path) -> None:
    if holdings_history.empty or "sector" not in holdings_history.columns:
        return

    sector_frame = holdings_history.pivot_table(
        index="date",
        columns="sector",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )
    if sector_frame.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sector_frame.plot.area(ax=ax, stacked=True)
    ax.set_title("Sector Allocation Over Time")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
