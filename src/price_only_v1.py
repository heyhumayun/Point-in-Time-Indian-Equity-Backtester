from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .backtester import build_output_directory, create_markdown_report, simulate_benchmark
from .config import BacktestConfig, PathConfig, StrategyConfig
from .data_loader import build_symbol_list, ensure_directories, load_fundamentals, load_price_frame, load_universe
from .dataset_metadata import ensure_research_dataset, load_dataset_metadata
from .features import (
    TRADING_DAYS_12M,
    TRADING_DAYS_200DMA,
    TRADING_DAYS_6M,
    TRADING_DAYS_LIQUIDITY,
    average_daily_traded_value,
    compute_returns,
    compute_sma,
    pivot_price_table,
    prepare_rebalance_dates,
)
from .metrics import compute_drawdown, summarize_performance
from .plots import plot_drawdown, plot_equity_curve, plot_monthly_returns
from .portfolio import Position, execute_rebalance
from .universe_history import get_universe_on_date, summarize_universe_coverage
from .validation import build_validation_summary, validate_backtest_inputs


PRICE_ONLY_WEIGHTS = {
    "momentum_12m": 0.5,
    "momentum_6m": 0.25,
    "relative_strength": 0.25,
}

PRICE_ONLY_COLUMNS = [
    "symbol",
    "sector",
    "market_cap",
    "adv60_inr",
    "sma_200",
    "last_close",
    "above_200dma",
    "momentum_12m",
    "momentum_6m",
    "relative_strength",
    "passes_market_cap",
    "passes_liquidity",
    "passes_trend",
    "has_complete_prices",
    "composite_score",
    "score_rank",
]


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, ascending=True, method="average").fillna(0.0)


def determine_actual_start_date(
    requested_start: str,
    universe: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Timestamp:
    requested = pd.Timestamp(requested_start)
    universe_start = pd.to_datetime(universe["start_date"]).min()
    first_price = pd.to_datetime(prices["date"]).min()
    warmup_ready = first_price + pd.tseries.offsets.BDay(TRADING_DAYS_12M + 1)
    return max(requested, pd.Timestamp(universe_start), pd.Timestamp(warmup_ready))


def build_price_only_snapshot(
    rebalance_date: pd.Timestamp,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_symbol: str,
    config: StrategyConfig,
) -> pd.DataFrame:
    eligible_frame, _ = build_price_only_snapshot_with_diagnostics(
        rebalance_date=rebalance_date,
        prices=prices,
        universe=universe,
        fundamentals=fundamentals,
        benchmark_symbol=benchmark_symbol,
        config=config,
    )
    return eligible_frame


def build_price_only_snapshot_with_diagnostics(
    rebalance_date: pd.Timestamp,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark_symbol: str,
    config: StrategyConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame(columns=PRICE_ONLY_COLUMNS)
    empty_rejected = pd.DataFrame(columns=["date", "symbol", "company_name", "sector", "rejection_reason"])
    close_table = pivot_price_table(prices, "adj_close")
    volume_table = pivot_price_table(prices, "volume")
    if rebalance_date not in close_table.index:
        raise ValueError(f"Rebalance date {rebalance_date.date()} is not present in price history.")

    date_position = close_table.index.get_loc(rebalance_date)
    if date_position == 0:
        return empty, empty_rejected

    prior_date = pd.Timestamp(close_table.index[date_position - 1])
    close_history = close_table.loc[:prior_date]
    volume_history = volume_table.loc[:prior_date]
    if len(close_history) < TRADING_DAYS_12M + 1:
        return empty, empty_rejected

    universe_as_of = get_universe_on_date(universe, prior_date)
    if universe_as_of.empty:
        return empty, empty_rejected

    fundamentals_latest = pd.DataFrame(columns=["symbol", "market_cap"])
    if not fundamentals.empty:
        fundamentals_eligible = fundamentals[pd.to_datetime(fundamentals["effective_date"]) <= prior_date].copy()
        if not fundamentals_eligible.empty:
            fundamentals_latest = fundamentals_eligible.sort_values(["symbol", "effective_date"]).groupby("symbol", as_index=False).tail(1)

    benchmark_return = compute_returns(close_history[[benchmark_symbol]].dropna(), TRADING_DAYS_12M)
    benchmark_12m = float(benchmark_return.iloc[0]) if not benchmark_return.empty else 0.0

    frame = pd.DataFrame({"symbol": close_history.columns})
    frame["momentum_12m"] = compute_returns(close_history, TRADING_DAYS_12M).reindex(frame["symbol"]).values
    frame["momentum_6m"] = compute_returns(close_history, TRADING_DAYS_6M).reindex(frame["symbol"]).values
    frame["adv60_inr"] = average_daily_traded_value(close_history, volume_history, TRADING_DAYS_LIQUIDITY).reindex(frame["symbol"]).values
    frame["sma_200"] = compute_sma(close_history, TRADING_DAYS_200DMA).reindex(frame["symbol"]).values
    latest_close = close_history.iloc[-1]
    frame["last_close"] = latest_close.reindex(frame["symbol"]).values
    frame["above_200dma"] = frame["last_close"] > frame["sma_200"]
    frame["relative_strength"] = frame["momentum_12m"] - benchmark_12m
    frame = frame.loc[~frame["symbol"].str.startswith("^")].copy()
    frame = frame.merge(universe_as_of[["symbol", "company_name", "sector"]], on="symbol", how="inner")
    if not fundamentals_latest.empty:
        frame = frame.merge(fundamentals_latest[["symbol", "market_cap"]], on="symbol", how="left")
    else:
        frame["market_cap"] = pd.NA
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    frame["passes_market_cap"] = True if config.min_market_cap_inr <= 0 else frame["market_cap"] > config.min_market_cap_inr
    frame["passes_liquidity"] = frame["adv60_inr"] > config.min_adv60_inr
    frame["passes_trend"] = frame["above_200dma"].fillna(False)
    frame["has_complete_prices"] = frame[["momentum_12m", "momentum_6m", "relative_strength", "adv60_inr", "sma_200"]].notna().all(axis=1)
    frame["rejection_reason"] = ""
    frame.loc[~frame["passes_market_cap"], "rejection_reason"] += "market_cap;"
    frame.loc[~frame["passes_liquidity"], "rejection_reason"] += "liquidity;"
    frame.loc[~frame["passes_trend"], "rejection_reason"] += "trend;"
    frame.loc[~frame["has_complete_prices"], "rejection_reason"] += "incomplete_prices;"
    rejected = frame.loc[frame["rejection_reason"] != "", ["symbol", "company_name", "sector", "rejection_reason"]].copy()
    if not rejected.empty:
        rejected["rejection_reason"] = rejected["rejection_reason"].str.rstrip(";")
        rejected["date"] = rebalance_date
    eligible = frame.loc[
        frame["passes_market_cap"] & frame["passes_liquidity"] & frame["passes_trend"] & frame["has_complete_prices"]
    ].copy()
    if eligible.empty:
        return empty, rejected if not rejected.empty else empty_rejected

    composite = sum(percentile_rank(eligible[column]) * weight for column, weight in PRICE_ONLY_WEIGHTS.items())
    eligible["composite_score"] = composite / sum(PRICE_ONLY_WEIGHTS.values())
    eligible["score_rank"] = eligible["composite_score"].rank(ascending=False, method="first").astype(int)
    eligible = eligible.sort_values(["score_rank", "symbol"]).reset_index(drop=True)
    return eligible[PRICE_ONLY_COLUMNS], rejected if not rejected.empty else empty_rejected


def run_price_only_v1(backtest_config: BacktestConfig, strategy_config: StrategyConfig, path_config: PathConfig) -> Path:
    ensure_directories(path_config)
    if backtest_config.output_name is None and backtest_config.output_dir is None:
        backtest_config = BacktestConfig(**{**asdict(backtest_config), "output_name": "price_only_v1"})

    warnings_list: List[str] = []
    universe = load_universe(path_config.universe_file)
    dataset_metadata = load_dataset_metadata(path_config.dataset_metadata_file, universe["symbol"].astype(str).tolist())
    ensure_research_dataset(dataset_metadata, backtest_config.report_as_research)
    if backtest_config.output_dir is None and dataset_metadata.universe_mode != "current_snapshot_only":
        backtest_config = BacktestConfig(**{**asdict(backtest_config), "output_dir": str(path_config.reports_dir / "price_only_v1_survivorship_reduced")})
    output_dir = build_output_directory(path_config, backtest_config)
    if dataset_metadata.universe_mode == "current_snapshot_only":
        warnings_list.append(
            "Universe mode is current_snapshot_only. Results remain exposed to survivorship bias and must not be described as survivorship-bias-reduced."
        )
    if dataset_metadata.universe_snapshot_min_date and pd.Timestamp(backtest_config.start_date) < pd.Timestamp(dataset_metadata.universe_snapshot_min_date):
        warnings_list.append("Backtest before first historical universe snapshot is not point-in-time reliable.")
    try:
        fundamentals = (
            load_fundamentals(path_config.fundamentals_file)
            if backtest_config.use_fundamentals
            else pd.DataFrame(columns=["effective_date", "symbol", "market_cap"])
        )
    except FileNotFoundError:
        fundamentals = pd.DataFrame(columns=["effective_date", "symbol", "market_cap"])
        if strategy_config.min_market_cap_inr > 0:
            raise
        warnings_list.append("Fundamentals file not found. Price-Only V1 is proceeding because the market-cap filter is disabled.")
    if not backtest_config.use_fundamentals:
        warnings_list.append("Fundamentals were intentionally ignored for this Price-Only V1 run.")
    benchmark_symbols = backtest_config.benchmark_symbols or strategy_config.benchmark_symbols
    symbols = build_symbol_list(universe, benchmark_symbols, backtest_config.universe_symbols)
    requested_history_start = str((pd.Timestamp(backtest_config.start_date) - pd.Timedelta(days=400)).date())
    prices = load_price_frame(
        paths=path_config,
        symbols=symbols,
        start_date=requested_history_start,
        end_date=backtest_config.end_date,
        refresh=backtest_config.refresh_cache,
        prices_file=backtest_config.prices_file,
        download_missing=backtest_config.download_prices,
    )
    warnings_list.extend(
        validate_backtest_inputs(
            universe=universe,
            fundamentals=fundamentals,
            prices=prices,
            start_date=backtest_config.start_date,
            end_date=backtest_config.end_date,
            benchmark_symbols=benchmark_symbols,
            require_fundamentals=backtest_config.use_fundamentals,
        )
    )
    validation_summary = build_validation_summary(
        universe=universe,
        prices=prices,
        benchmark_symbols=benchmark_symbols,
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
    )
    if validation_summary["ticker_coverage"]["coverage_pct"] < 95:
        warnings_list.append(
            f"Ticker coverage is only {validation_summary['ticker_coverage']['coverage_pct']:.2f}% of the requested universe."
        )
    if validation_summary["missing_data_percentage"] > 5:
        warnings_list.append(
            f"Missing adjusted-close coverage across active symbol-days is {validation_summary['missing_data_percentage']:.2f}%."
        )
    if validation_summary["rebalance_day_open_availability_pct"] < 95:
        warnings_list.append("Rebalance-day open availability is below 95%, so execution coverage may be incomplete.")
    coverage_rows = pd.DataFrame(validation_summary.get("eligible_ticker_count_by_rebalance_month", []))
    if not coverage_rows.empty:
        if (coverage_rows["eligible_ticker_count"] < 100).any():
            first_low = coverage_rows.loc[coverage_rows["eligible_ticker_count"] < 100].iloc[0]["date"]
            warnings_list.append(f"Eligible universe falls below 100 tickers in at least one rebalance month. First month: {first_low}")
        if (coverage_rows["count_change_pct"].abs() >= 20).any():
            examples = ", ".join(coverage_rows.loc[coverage_rows["count_change_pct"].abs() >= 20, "date"].head(5).tolist())
            warnings_list.append(f"Large eligible-universe drops/spikes detected between rebalance months. Example months: {examples}")

    actual_start = determine_actual_start_date(backtest_config.start_date, universe, prices)
    if actual_start > pd.Timestamp(backtest_config.start_date):
        warnings_list.append(
            f"Requested start date {pd.Timestamp(backtest_config.start_date).date()} was shifted to {actual_start.date()} because local data coverage starts later."
        )

    close_table = pivot_price_table(prices, "close")
    open_table = pivot_price_table(prices, "open")
    benchmark_calendar_prices = prices.loc[prices["symbol"] == benchmark_symbols["nifty_500"], ["date"]].copy()
    rebalance_dates = prepare_rebalance_dates(benchmark_calendar_prices, str(actual_start.date()), backtest_config.end_date)
    if len(rebalance_dates) < 2:
        raise ValueError("Not enough monthly rebalance dates found for the price-only strategy.")

    positions: Dict[str, Position] = {}
    cash = 0.0
    trades_records: List[Dict[str, object]] = []
    holdings_records: List[Dict[str, object]] = []
    daily_records: List[Dict[str, object]] = []
    eligible_universe_records: List[Dict[str, object]] = []
    selected_holdings_records: List[Dict[str, object]] = []
    rejected_tickers_records: List[Dict[str, object]] = []
    price_dates = close_table.index

    for idx, rebalance_date in enumerate(rebalance_dates):
        contribution = backtest_config.initial_capital if idx == 0 else backtest_config.monthly_contribution
        cash += contribution

        universe_as_of = get_universe_on_date(universe, rebalance_date)
        eligible_universe_records.append(
            {
                "date": rebalance_date.date().isoformat(),
                "eligible_ticker_count": int(len(universe_as_of)),
                "universe_mode": dataset_metadata.universe_mode,
                "survivorship_bias_status": dataset_metadata.survivorship_bias_status,
            }
        )

        ranked, rejected_frame = build_price_only_snapshot_with_diagnostics(
            rebalance_date=rebalance_date,
            prices=prices,
            universe=universe,
            fundamentals=fundamentals,
            benchmark_symbol=benchmark_symbols["nifty_500"],
            config=strategy_config,
        )
        if not rejected_frame.empty:
            rejected_tickers_records.extend(rejected_frame.to_dict("records"))
        target_portfolio = ranked.head(strategy_config.rebalance_target_count).copy()
        if not target_portfolio.empty:
            target_portfolio["target_weight"] = 1.0 / len(target_portfolio)
            selected_frame = target_portfolio.copy()
            selected_frame["date"] = rebalance_date
            selected_holdings_records.extend(selected_frame.to_dict("records"))

        positions, cash, new_trades, holdings_frame = execute_rebalance(
            rebalance_date=rebalance_date,
            target_portfolio=target_portfolio,
            current_positions=positions,
            current_cash=cash,
            prices_on_date=open_table.loc[rebalance_date],
            config=strategy_config,
        )

        for trade in new_trades:
            trades_records.append(
                {
                    "date": trade.date,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "shares": trade.shares,
                    "price": trade.price,
                    "gross_value": trade.gross_value,
                    "transaction_cost": trade.transaction_cost,
                    "net_cash_flow": trade.net_cash_flow,
                    "reason": trade.reason,
                    "pnl": 0.0,
                }
            )

        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else price_dates[-1] + pd.Timedelta(days=1)
        period_dates = [date for date in price_dates if rebalance_date <= date < next_rebalance]
        for date in period_dates:
            day_prices = close_table.loc[date]
            holdings_value = sum(
                position.shares * float(day_prices.get(symbol, 0.0))
                for symbol, position in positions.items()
                if pd.notna(day_prices.get(symbol))
            )
            daily_records.append(
                {
                    "date": pd.Timestamp(date),
                    "portfolio_value": cash + holdings_value,
                    "cash": cash,
                    "holdings_value": holdings_value,
                    "net_contribution": contribution if date == rebalance_date else 0.0,
                }
            )

        if not holdings_frame.empty:
            close_prices = close_table.loc[rebalance_date]
            holdings_frame = holdings_frame.copy()
            holdings_frame["price"] = holdings_frame["symbol"].map(close_prices)
            holdings_frame["market_value"] = holdings_frame["shares"] * holdings_frame["price"]
            total_value = holdings_frame["market_value"].sum() + cash
            holdings_frame["weight"] = holdings_frame["market_value"] / total_value if total_value else 0.0
            holdings_frame["date"] = rebalance_date
            holdings_records.extend(holdings_frame.to_dict("records"))

        if ranked.empty:
            warnings_list.append(
                f"No eligible securities on {rebalance_date.date()} for the price-only strategy."
            )

    equity_curve = pd.DataFrame(daily_records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    equity_curve["daily_return"] = equity_curve["portfolio_value"].pct_change().fillna(0.0)
    equity_curve["drawdown"] = compute_drawdown(equity_curve["portfolio_value"])

    benchmark_summaries: Dict[str, Dict[str, float]] = {}
    for benchmark_name, symbol in benchmark_symbols.items():
        benchmark_values = simulate_benchmark(
            benchmark_prices=close_table[symbol].dropna(),
            rebalance_dates=rebalance_dates,
            initial_capital=backtest_config.initial_capital,
            monthly_contribution=backtest_config.monthly_contribution,
        )
        equity_curve[f"benchmark_{benchmark_name}"] = equity_curve["date"].map(benchmark_values)
        benchmark_equity = equity_curve[["date", f"benchmark_{benchmark_name}"]].dropna().rename(
            columns={f"benchmark_{benchmark_name}": "portfolio_value"}
        )
        benchmark_equity["net_contribution"] = equity_curve["net_contribution"]
        monthly_benchmark = benchmark_equity.set_index("date")["portfolio_value"].resample("M").last().pct_change().rename("monthly_return").reset_index()
        benchmark_summaries[benchmark_name] = summarize_performance(
            benchmark_equity,
            trades=pd.DataFrame(columns=["side", "gross_value", "pnl"]),
            monthly_performance=monthly_benchmark,
        )

    monthly_performance = equity_curve.set_index("date")[["portfolio_value", "cash", "holdings_value", "net_contribution"]].resample("M").last()
    monthly_performance["monthly_return"] = monthly_performance["portfolio_value"].pct_change()
    monthly_performance["drawdown"] = compute_drawdown(monthly_performance["portfolio_value"])
    monthly_performance.index = monthly_performance.index.to_period("M").to_timestamp("M")
    monthly_performance = monthly_performance.reset_index()

    trades_df = pd.DataFrame(trades_records)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=["date", "symbol", "side", "shares", "price", "gross_value", "transaction_cost", "net_cash_flow", "reason", "pnl"])
    holdings_history = pd.DataFrame(holdings_records)
    eligible_universe_history = pd.DataFrame(eligible_universe_records)
    selected_holdings_history = pd.DataFrame(selected_holdings_records)
    rejected_tickers_history = pd.DataFrame(rejected_tickers_records)
    universe_coverage_summary = summarize_universe_coverage(universe, str(actual_start.date()), backtest_config.end_date)
    if not eligible_universe_history.empty:
        eligible_universe_history["count_change"] = eligible_universe_history["eligible_ticker_count"].diff().fillna(0.0)
        prior_counts = eligible_universe_history["eligible_ticker_count"].shift(1)
        eligible_universe_history["count_change_pct"] = (
            ((eligible_universe_history["eligible_ticker_count"] - prior_counts) / prior_counts.replace(0, pd.NA)) * 100.0
        ).fillna(0.0)
    summary = summarize_performance(equity_curve, trades_df, monthly_performance)

    equity_curve.to_csv(output_dir / "daily_equity_curve.csv", index=False)
    monthly_performance.to_csv(output_dir / "monthly_performance.csv", index=False)
    monthly_performance[["date", "monthly_return"]].to_csv(output_dir / "monthly_returns.csv", index=False)
    holdings_history.to_csv(output_dir / "monthly_holdings.csv", index=False)
    eligible_universe_history.to_csv(output_dir / "eligible_universe_by_month.csv", index=False)
    selected_holdings_history.to_csv(output_dir / "selected_holdings_by_month.csv", index=False)
    rejected_tickers_history.to_csv(output_dir / "rejected_tickers_by_month.csv", index=False)
    universe_coverage_summary.to_csv(output_dir / "universe_coverage_summary.csv", index=False)
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(benchmark_summaries).T.reset_index().rename(columns={"index": "benchmark"}).to_csv(
        output_dir / "benchmark_summary.csv", index=False
    )
    pd.DataFrame({"warning": warnings_list}).to_csv(output_dir / "warnings.csv", index=False)

    plot_equity_curve(equity_curve, output_dir / "equity_curve.png")
    plot_drawdown(equity_curve, output_dir / "drawdown_curve.png")
    plot_monthly_returns(monthly_performance, output_dir / "monthly_returns.png")
    create_markdown_report(
        output_dir=output_dir,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        summary=summary,
        benchmark_summaries=benchmark_summaries,
        warnings_list=warnings_list,
        dataset_type=dataset_metadata.dataset_type,
        report_label="Research" if backtest_config.report_as_research else "Backtest",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "backtest_config": asdict(backtest_config),
                "strategy_config": asdict(strategy_config),
                "strategy_name": "price_only_v1",
                "actual_start_date": str(actual_start.date()),
                "ranking_weights": PRICE_ONLY_WEIGHTS,
                "dataset_metadata": asdict(dataset_metadata),
                "validation_summary": validation_summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (output_dir / "dataset_metadata.json").write_text(json.dumps(asdict(dataset_metadata), indent=2), encoding="utf-8")
    (output_dir / "validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Price-only Version 1 Indian equity backtester")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--monthly-contribution", type=float, default=20000.0)
    parser.add_argument("--output-name", default="price_only_v1")
    parser.add_argument("--transaction-cost-rate", type=float, default=0.002)
    parser.add_argument("--min-market-cap", type=float, default=10_000_000_000.0)
    parser.add_argument("--min-adv", type=float, default=50_000_000.0)
    parser.add_argument("--prices-file", default=None)
    parser.add_argument("--no-download-prices", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--as-research", action="store_true", help="Label outputs as research if dataset metadata is non-synthetic.")
    parser.add_argument(
        "--ignore-fundamentals-file",
        action="store_true",
        help="Run without loading fundamentals. Pair with --min-market-cap 0 to avoid synthetic fundamentals in price-only research.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    path_config = PathConfig.from_project_root(project_root)
    strategy_config = StrategyConfig(
        min_market_cap_inr=args.min_market_cap,
        min_adv60_inr=args.min_adv,
        max_sector_weight=1.0,
        transaction_cost_rate=args.transaction_cost_rate,
    )
    backtest_config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        output_name=args.output_name,
        output_dir=args.output_dir,
        prices_file=args.prices_file,
        download_prices=not args.no_download_prices,
        report_as_research=args.as_research,
        use_fundamentals=not args.ignore_fundamentals_file,
    )
    output_dir = run_price_only_v1(backtest_config, strategy_config, path_config)
    print(f"Price-only backtest complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
