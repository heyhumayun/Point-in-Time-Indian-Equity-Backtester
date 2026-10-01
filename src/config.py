from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StrategyConfig:
    min_market_cap_inr: float = 10_000_000_000.0
    min_adv60_inr: float = 50_000_000.0
    min_roe: float = 0.15
    max_debt_to_equity: float = 0.5
    min_revenue_growth: float = 0.10
    min_eps_growth: float = 0.10
    transaction_cost_rate: float = 0.002
    rebalance_target_count: int = 15
    hold_rank_buffer: int = 30
    max_single_weight: float = 0.10
    max_sector_weight: float = 0.25
    ai_adjustment_cap: float = 0.10
    benchmark_symbols: Dict[str, str] = field(
        default_factory=lambda: {
            "nifty_50": "^NSEI",
            "nifty_500": "^CRSLDX",
        }
    )
    composite_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "momentum_12m": 0.25,
            "momentum_6m": 0.15,
            "momentum_3m": 0.10,
            "relative_strength": 0.15,
            "roe": 0.10,
            "revenue_growth_yoy": 0.10,
            "eps_growth_yoy": 0.10,
            "debt_score": 0.05,
        }
    )


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    universe_snapshots_dir: Path
    outputs_dir: Path
    reports_dir: Path
    tests_dir: Path
    universe_file: Path
    fundamentals_file: Path
    prices_file: Path
    dataset_metadata_file: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PathConfig":
        data_dir = project_root / "data"
        processed_dir = data_dir / "processed"
        return cls(
            project_root=project_root,
            raw_data_dir=data_dir / "raw",
            processed_data_dir=processed_dir,
            universe_snapshots_dir=data_dir / "universe_snapshots",
            outputs_dir=project_root / "outputs",
            reports_dir=project_root / "reports",
            tests_dir=project_root / "tests",
            universe_file=processed_dir / "universe.csv",
            fundamentals_file=processed_dir / "fundamentals_pti.csv",
            prices_file=processed_dir / "prices.csv",
            dataset_metadata_file=processed_dir / "dataset_metadata.json",
        )


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float
    monthly_contribution: float
    universe_symbols: Optional[List[str]] = None
    benchmark_symbols: Optional[Dict[str, str]] = None
    output_name: Optional[str] = None
    output_dir: Optional[str] = None
    download_prices: bool = True
    refresh_cache: bool = False
    prices_file: Optional[str] = None
    report_as_research: bool = False
    use_fundamentals: bool = True


DEFAULT_REPORT_COLUMNS = [
    "date",
    "portfolio_value",
    "cash",
    "holdings_value",
    "net_contribution",
    "daily_return",
    "drawdown",
]
