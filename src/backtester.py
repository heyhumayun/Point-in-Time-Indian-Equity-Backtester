from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import BacktestConfig, PathConfig, StrategyConfig
from .data_loader import build_symbol_list, ensure_directories, load_fundamentals, load_price_frame, load_universe
from .dataset_metadata import ensure_research_dataset, load_dataset_metadata
from .features import build_rebalance_snapshot, pivot_price_table, prepare_rebalance_dates
from .metrics import compute_drawdown, summarize_performance
from .plots import plot_drawdown, plot_equity_curve, plot_monthly_returns, plot_sector_allocation
from .portfolio import Position, execute_rebalance, select_target_portfolio
from .scoring import apply_composite_score
from .validation import build_validation_summary, validate_backtest_inputs


def simulate_benchmark(
    benchmark_prices: pd.Series,
    rebalance_dates: List[pd.Timestamp],
    initial_capital: float,
    monthly_contribution: float,
) -> pd.Series:
    shares = 0.0
    cash = 0.0
    values = []
    contributions_by_date = {rebalance_dates[0]: initial_capital} if rebalance_dates else {}
    for date in rebalance_dates[1:]:
        contributions_by_date[date] = monthly_contribution

    for date, price in benchmark_prices.dropna().items():
        contribution = contributions_by_date.get(pd.Timestamp(date), 0.0)
        if contribution:
            cash += contribution
        if pd.Timestamp(date) in contributions_by_date and price > 0:
            shares += cash / price
            cash = 0.0
        values.append((pd.Timestamp(date), shares * price + cash))

    result = pd.Series(dict(values)).sort_index()
    result.index.name = "date"
    return result


def build_output_directory(paths: PathConfig, config: BacktestConfig) -> Path:
    if config.output_dir:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    output_name = config.output_name or (
        f"backtest_{pd.Timestamp(config.start_date).date()}_{pd.Timestamp(config.end_date).date()}_"
        f"{int(config.initial_capital)}_{int(config.monthly_contribution)}"
    )
    output_dir = paths.outputs_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_markdown_report(
    output_dir: Path,
    strategy_config: StrategyConfig,
    backtest_config: BacktestConfig,
    summary: Dict[str, float],
    benchmark_summaries: Dict[str, Dict[str, float]],
    warnings_list: List[str],
    dataset_type: str,
    report_label: str,
) -> Path:
    report_path = output_dir / "report.md"
    lines = [
        f"# Indian Equity Portfolio {report_label} Report",
        "",
        "## Backtest Configuration",
        "",
        f"- Start date: {backtest_config.start_date}",
        f"- End date: {backtest_config.end_date}",
        f"- Initial capital: INR {backtest_config.initial_capital:,.0f}",
        f"- Monthly contribution: INR {backtest_config.monthly_contribution:,.0f}",
        f"- Transaction cost rate: {strategy_config.transaction_cost_rate:.2%}",
        f"- Rebalance target count: {strategy_config.rebalance_target_count}",
        f"- Dataset type: {dataset_type}",
        f"- Result label: {report_label}",
        "",
        "## Strategy Summary",
        "",
        "- Universe: Nifty 500 symbols supplied through `data/processed/universe.csv`",
        "- Rebalance timing: first trading day of each month",
        "- Signal timestamp: previous trading day close",
        "- Execution assumption: rebalance-day open with configurable transaction costs",
        "- Exit rules: leave top 30 ranking or close below 200-day moving average",
        "",
        "## Performance Summary",
        "",
    ]

    for key, value in summary.items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value:,.4f}" if isinstance(value, float) else f"- {key}: {value}")

    lines.extend(["", "## Benchmarks", ""])
    for name, benchmark_summary in benchmark_summaries.items():
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        for key, value in benchmark_summary.items():
            lines.append(
                f"- {key.replace('_', ' ').title()}: {value:,.4f}" if isinstance(value, float) else f"- {key}: {value}"
            )
        lines.append("")

    lines.extend(
        [
            "## Assumptions And Limitations",
            "",
            "- Point-in-time fundamentals are required to avoid look-ahead bias.",
            "- A static universe file introduces survivorship bias if historical membership dates are missing.",
            "- Free Yahoo Finance data can contain symbol mapping issues, missing corporate actions, and benchmark inconsistencies.",
            "- AI-style research fields are placeholders and default to neutral values unless supplied in the fundamentals file.",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings_list:
        lines.extend(f"- {item}" for item in warnings_list)
    else:
        lines.append("- No runtime warnings captured.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_backtest(
    backtest_config: BacktestConfig,
    strategy_config: StrategyConfig,
    path_config: PathConfig,
) -> Path:
    ensure_directories(path_config)
    output_dir = build_output_directory(path_config, backtest_config)

    warnings_list: List[str] = []

    def capture_warning(message: str) -> None:
        warnings_list.append(message)

    universe = load_universe(path_config.universe_file)
    fundamentals = load_fundamentals(path_config.fundamentals_file)
    dataset_metadata = load_dataset_metadata(path_config.dataset_metadata_file, universe["symbol"].astype(str).tolist())
    ensure_research_dataset(dataset_metadata, backtest_config.report_as_research)
    if dataset_metadata.universe_mode == "current_snapshot_only":
        warnings_list.append(
            "Universe mode is current_snapshot_only. Results remain exposed to survivorship bias and must not be described as survivorship-bias-reduced."
        )

    benchmark_symbols = backtest_config.benchmark_symbols or strategy_config.benchmark_symbols
    symbols = build_symbol_list(universe, benchmark_symbols, backtest_config.universe_symbols)
    history_start = str((pd.Timestamp(backtest_config.start_date) - pd.Timedelta(days=400)).date())
    prices = load_price_frame(
        paths=path_config,
        symbols=symbols,
        start_date=history_start,
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
        warnings_list.append(
            "Rebalance-day open availability is below 95%, so execution coverage may be incomplete."
        )
    coverage_rows = pd.DataFrame(validation_summary.get("eligible_ticker_count_by_rebalance_month", []))
    if not coverage_rows.empty:
        if (coverage_rows["eligible_ticker_count"] < 100).any():
            first_low = coverage_rows.loc[coverage_rows["eligible_ticker_count"] < 100].iloc[0]["date"]
            warnings_list.append(f"Eligible universe falls below 100 tickers in at least one rebalance month. First month: {first_low}")
        if (coverage_rows["count_change_pct"].abs() >= 20).any():
            examples = ", ".join(coverage_rows.loc[coverage_rows["count_change_pct"].abs() >= 20, "date"].head(5).tolist())
            warnings_list.append(f"Large eligible-universe drops/spikes detected between rebalance months. Example months: {examples}")

    close_table = pivot_price_table(prices, "close")
    open_table = pivot_price_table(prices, "open")
    benchmark_calendar_prices = prices.loc[prices["symbol"] == benchmark_symbols["nifty_500"], ["date"]].copy()
    rebalance_dates = prepare_rebalance_dates(benchmark_calendar_prices, backtest_config.start_date, backtest_config.end_date)
    if len(rebalance_dates) < 2:
        raise ValueError("Not enough monthly rebalance dates found in the requested window.")

    price_dates = close_table.index
    positions: Dict[str, Position] = {}
    cash = 0.0
    trades_records = []
    holdings_records = []
    monthly_records = []
    daily_records = []
    for idx, rebalance_date in enumerate(rebalance_dates):
        contribution = backtest_config.initial_capital if idx == 0 else backtest_config.monthly_contribution
        cash += contribution

        snapshot = build_rebalance_snapshot(
            rebalance_date=rebalance_date,
            prices=prices,
            universe=universe,
            fundamentals=fundamentals,
            benchmark_symbol=benchmark_symbols["nifty_500"],
            config=strategy_config,
        )
        ranked = apply_composite_score(snapshot.eligible_frame, strategy_config)

        prices_on_date = open_table.loc[rebalance_date]
        target_portfolio = select_target_portfolio(ranked, positions, strategy_config) if not ranked.empty else ranked

        positions, cash, new_trades, holdings_frame = execute_rebalance(
            rebalance_date=rebalance_date,
            target_portfolio=target_portfolio,
            current_positions=positions,
            current_cash=cash,
            prices_on_date=prices_on_date,
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
            prices_for_day = close_table.loc[date]
            holdings_value = sum(
                position.shares * float(prices_for_day.get(symbol, 0.0))
                for symbol, position in positions.items()
                if pd.notna(prices_for_day.get(symbol))
            )
            portfolio_value = cash + holdings_value
            daily_records.append(
                {
                    "date": pd.Timestamp(date),
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "holdings_value": holdings_value,
                    "net_contribution": contribution if date == rebalance_date else 0.0,
                }
            )

        if not holdings_frame.empty:
            closing_values = holdings_frame.copy()
            closing_values["price"] = closing_values["symbol"].map(close_table.loc[rebalance_date])
            closing_values["market_value"] = closing_values["shares"] * closing_values["price"]
            total_value = closing_values["market_value"].sum() + cash
            closing_values["date"] = rebalance_date
            closing_values["weight"] = closing_values["market_value"] / total_value if total_value else 0.0
            holdings_records.extend(closing_values.to_dict("records"))

        monthly_records.append(
            {
                "date": rebalance_date,
                "contribution": contribution,
                "cash_after_rebalance": cash,
                "holdings_count": len(positions),
                "selected_symbols": ",".join(sorted(positions.keys())),
            }
        )

        if snapshot.eligible_frame.empty:
            capture_warning(
                f"No eligible securities on {rebalance_date.date()}. Check fundamentals coverage or strict filters."
            )

    equity_curve = pd.DataFrame(daily_records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    equity_curve["daily_return"] = equity_curve["portfolio_value"].pct_change().fillna(0.0)
    equity_curve["drawdown"] = compute_drawdown(equity_curve["portfolio_value"])

    benchmark_series = {}
    for benchmark_name, symbol in benchmark_symbols.items():
        benchmark_prices = close_table[symbol].dropna()
        benchmark_values = simulate_benchmark(
            benchmark_prices=benchmark_prices,
            rebalance_dates=rebalance_dates,
            initial_capital=backtest_config.initial_capital,
            monthly_contribution=backtest_config.monthly_contribution,
        )
        benchmark_series[benchmark_name] = benchmark_values
        equity_curve[f"benchmark_{benchmark_name}"] = equity_curve["date"].map(benchmark_values)

    monthly_performance = equity_curve.set_index("date")[["portfolio_value", "cash", "holdings_value", "net_contribution"]].resample("M").last()
    monthly_performance["monthly_return"] = monthly_performance["portfolio_value"].pct_change()
    monthly_performance["drawdown"] = compute_drawdown(monthly_performance["portfolio_value"])
    monthly_performance.index = monthly_performance.index.to_period("M").to_timestamp("M")
    monthly_performance = monthly_performance.reset_index()

    trades_df = pd.DataFrame(trades_records)
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=["date", "symbol", "side", "shares", "price", "gross_value", "transaction_cost", "net_cash_flow", "reason", "pnl"]
        )

    if not trades_df.empty:
        buy_reference: Dict[str, List[Tuple[int, float]]] = {}
        pnl_values = []
        for _, row in trades_df.sort_values(["date", "symbol"]).iterrows():
            symbol = row["symbol"]
            shares = int(row["shares"])
            price = float(row["price"])
            if row["side"] == "BUY":
                if shares > 0:
                    lot_cost_per_share = (row["gross_value"] + row["transaction_cost"]) / shares
                    buy_reference.setdefault(symbol, []).append((shares, lot_cost_per_share))
                pnl_values.append(0.0)
            else:
                remaining = shares
                cost_basis = 0.0
                lots = buy_reference.setdefault(symbol, [])
                while remaining > 0 and lots:
                    lot_shares, lot_price = lots[0]
                    used = min(remaining, lot_shares)
                    cost_basis += used * lot_price
                    remaining -= used
                    lot_shares -= used
                    if lot_shares == 0:
                        lots.pop(0)
                    else:
                        lots[0] = (lot_shares, lot_price)
                pnl_values.append(row["gross_value"] - cost_basis - row["transaction_cost"])
        trades_df["pnl"] = pnl_values

    holdings_history = pd.DataFrame(holdings_records)
    summary = summarize_performance(equity_curve, trades_df, monthly_performance)

    benchmark_summaries = {}
    for benchmark_name in benchmark_symbols:
        benchmark_equity = equity_curve[["date", f"benchmark_{benchmark_name}"]].dropna().rename(
            columns={f"benchmark_{benchmark_name}": "portfolio_value"}
        )
        benchmark_equity["net_contribution"] = equity_curve["net_contribution"]
        benchmark_equity["daily_return"] = benchmark_equity["portfolio_value"].pct_change().fillna(0.0)
        benchmark_equity["drawdown"] = compute_drawdown(benchmark_equity["portfolio_value"])
        benchmark_summaries[benchmark_name] = summarize_performance(
            benchmark_equity,
            trades=pd.DataFrame(columns=["side", "gross_value", "pnl"]),
            monthly_performance=benchmark_equity.set_index("date")["portfolio_value"].resample("M").last().pct_change().rename("monthly_return").reset_index(),
        )

    equity_curve.to_csv(output_dir / "daily_equity_curve.csv", index=False)
    pd.DataFrame(monthly_records).to_csv(output_dir / "monthly_portfolio_state.csv", index=False)
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    holdings_history.to_csv(output_dir / "monthly_holdings.csv", index=False)
    monthly_performance.to_csv(output_dir / "monthly_performance.csv", index=False)
    monthly_performance[["date", "monthly_return"]].to_csv(output_dir / "monthly_returns.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(benchmark_summaries).T.reset_index().rename(columns={"index": "benchmark"}).to_csv(
        output_dir / "benchmark_summary.csv", index=False
    )
    pd.DataFrame({"warning": warnings_list}).to_csv(output_dir / "warnings.csv", index=False)

    plot_equity_curve(equity_curve, output_dir / "equity_curve.png")
    plot_drawdown(equity_curve, output_dir / "drawdown_curve.png")
    plot_monthly_returns(monthly_performance, output_dir / "monthly_returns.png")
    plot_sector_allocation(holdings_history, output_dir / "sector_allocation.png")

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
    parser = argparse.ArgumentParser(description="Indian equity portfolio backtester")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--initial-capital", required=True, type=float)
    parser.add_argument("--monthly-contribution", required=True, type=float)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--transaction-cost-rate", type=float, default=0.002)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-download-prices", action="store_true", help="Use only local CSV price data.")
    parser.add_argument("--prices-file", default=None, help="Optional local CSV with OHLCV history.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional override universe symbols.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--as-research", action="store_true", help="Label outputs as research if the dataset metadata is non-synthetic.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    path_config = PathConfig.from_project_root(project_root)
    strategy_config = StrategyConfig(transaction_cost_rate=args.transaction_cost_rate)
    backtest_config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        universe_symbols=args.symbols,
        output_name=args.output_name,
        output_dir=args.output_dir,
        download_prices=not args.no_download_prices,
        refresh_cache=args.refresh_cache,
        prices_file=args.prices_file,
        report_as_research=args.as_research,
    )
    output_dir = run_backtest(backtest_config, strategy_config, path_config)
    print(f"Backtest complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
