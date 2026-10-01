# Indian Equity Portfolio Backtester

A Python backtesting project for building and testing a monthly rebalanced Indian equity portfolio based on momentum, quality, and liquidity rules.

This project is for research and historical simulation only.

Synthetic demo outputs are only for pipeline validation and must not be used for performance conclusions.

- No live trading
- No broker integration
- No real orders

## What It Does

- Builds a portfolio from a Nifty 500-style universe
- Supports point-in-time universe intervals when historical snapshots are available
- Rebalances monthly
- Adds a fixed monthly contribution
- Applies liquidity and quality filters
- Ranks stocks using a composite score
- Compares results against Nifty 50 and Nifty 500 benchmarks
- Exports holdings, trades, performance tables, charts, and a Markdown report

## Strategy Summary

- Universe: stocks in `data/processed/universe.csv`
- Rebalance: first trading day of each month
- Contribution: added before each rebalance
- Signals: computed using information available through the previous trading day
- Execution model: rebalance-day open
- Portfolio: top 15 names, equal weight, with stock and sector caps
- Costs: configurable transaction cost, default `0.20%`

## Project Structure

```text
data/
  raw/
  universe_snapshots/
  processed/
src/
tests/
reports/
outputs/
README.md
requirements.txt
LICENSE
```

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate the demo dataset:

```bash
python -m src.demo_data
```

Validate the input files:

```bash
python -m src.validate_data --start-date 2020-01-01 --end-date 2025-12-31 --no-download
```

Run the demo backtest:

```bash
python -m src.backtester --start-date 2020-01-01 --end-date 2025-12-31 --initial-capital 500000 --monthly-contribution 20000 --output-name demo_backtest --no-download-prices
```

## Input Files

The backtester uses these main files:

- `universe.csv`
  Contains point-in-time universe intervals with:
  `ticker`, `company_name`, `start_date`, `end_date`, `source`, `notes`
- `fundamentals_pti.csv`
  Contains point-in-time fundamentals such as ROE, debt/equity, growth, and cash flow
- `prices.csv`
  Contains daily OHLCV price history for stocks and benchmarks
- `dataset_metadata.json`
  Records whether the dataset is `synthetic`, `downloaded`, or `manual`, plus universe mode and survivorship-bias metadata

Historical universe snapshots live in `data/universe_snapshots/` and should be named like:

```text
data/universe_snapshots/nifty500_2015-01-01.csv
data/universe_snapshots/nifty500_2016-01-01.csv
data/universe_snapshots/nifty500_2017-01-01.csv
```

Each snapshot file should contain:

```text
ticker,company_name
```

The importer also maintains:

- `data/universe_snapshots/sources_manifest.csv`
  Tracks snapshot date, filename, source URL, source name, downloaded date, source type, and notes

## Main Commands

Generate demo data:

```bash
python -m src.demo_data
```

Validate local CSV data:

```bash
python -m src.validate_data --start-date 2020-01-01 --end-date 2025-12-31 --no-download
```

Download the current NIFTY 500 universe and real OHLCV history:

```bash
python -m src.download_nifty500_data --start-date 1990-01-01 --end-date 2025-12-31
```

Build historical universe intervals from dated snapshots:

```bash
python -m src.build_universe_history
```

Import manually downloaded snapshot files:

```bash
python -m src.import_universe_snapshots --input-dir raw_snapshots/ --source-type official_niftyindices
```

Download price history for the existing universe file:

```bash
python -m src.download_prices --start-date 2014-01-01 --end-date 2025-12-31
```

Run a backtest:

```bash
python -m src.backtester --start-date 2020-01-01 --end-date 2025-12-31 --initial-capital 500000 --monthly-contribution 20000 --no-download-prices
```

Run Price-Only V1 on real data without loading synthetic fundamentals:

```bash
python -m src.price_only_v1 --start-date 1990-01-01 --end-date 2025-12-31 --initial-capital 500000 --monthly-contribution 20000 --min-market-cap 0 --ignore-fundamentals-file --no-download-prices --as-research --output-dir reports/price_only_v1_real_data
```

Run Price-Only V1 using the best available universe mode:

```bash
python -m src.price_only_v1 --start-date 2007-09-17 --end-date 2025-12-31 --initial-capital 500000 --monthly-contribution 20000 --min-market-cap 0 --ignore-fundamentals-file --no-download-prices --as-research --output-dir reports/price_only_v1_historical_universe
```

## Outputs

Each run writes a folder under `outputs/` with files such as:

- `daily_equity_curve.csv`
- `monthly_holdings.csv`
- `monthly_performance.csv`
- `trades.csv`
- `summary.csv`
- `eligible_universe_by_month.csv`
- `selected_holdings_by_month.csv`
- `rejected_tickers_by_month.csv`
- `universe_coverage_summary.csv`
- `report.md`
- performance plots

A demo run is also available in `reports/demo_backtest/`.

## Survivorship Bias

Survivorship bias happens when you test today's winners against the past as if they were always investable. If you project the current NIFTY 500 membership backward, the strategy can end up trading stocks that were not actually in the eligible universe at the time.

Historical universe snapshots reduce that bias by letting the backtest use only the tickers known to be eligible at each rebalance date. This project now supports dated snapshot files and builds interval-based universe membership from them.

Important:

- The project does not fabricate historical NIFTY 500 membership.
- If only the current constituent list is available, the universe mode is `current_snapshot_only`.
- `current_snapshot_only` is still useful for pipeline validation and exploratory research, but it is not survivorship-bias-reduced.
- Only historical snapshots should be used to argue that survivorship bias has been reduced.

## How To Reduce Survivorship Bias

1. Download real dated NIFTY 500 constituent snapshots manually from reliable sources.
2. Put the raw files in a staging folder such as `raw_snapshots/`.
3. Import them with:

```bash
python -m src.import_universe_snapshots --input-dir raw_snapshots/ --source-type official_niftyindices --source-name "Nifty Indices"
```

4. This normalizes each file to:

```text
ticker,company_name
```

and saves it as:

```text
data/universe_snapshots/nifty500_YYYY-MM-DD.csv
```

5. Build interval-based universe history:

```bash
python -m src.build_universe_history --snapshots-dir data/universe_snapshots --output data/processed/universe.csv
```

6. Re-run validation and the backtest:

```bash
python -m src.validate_data --start-date 2007-09-17 --end-date 2025-12-31 --no-download
python -m src.price_only_v1 --start-date 2007-09-17 --end-date 2025-12-31 --initial-capital 500000 --monthly-contribution 20000 --min-market-cap 0 --ignore-fundamentals-file --no-download-prices --as-research
```

Accepted raw snapshot formats:

- `.csv`
- `.xlsx`
- `.xls`
- `.zip` containing one supported snapshot file

Source-type values in the manifest:

- `official_nse`
- `official_niftyindices`
- `broker_archive`
- `data_vendor`
- `manual`
- `unknown`

Interpretation:

- `survivorship_bias_status=high`: current snapshot only
- `survivorship_bias_status=medium`: annual snapshots
- `survivorship_bias_status=medium_low`: quarterly snapshots
- `survivorship_bias_status=low`: monthly snapshots or better
- `universe_source_quality` reflects source provenance, not time granularity by itself

## Universe Modes

- `synthetic`: demo-only universe for pipeline validation
- `current_snapshot_only`: current constituent list projected backward; survivorship bias remains high
- `annual_snapshots`: reduced survivorship bias, but still coarse
- `quarterly_snapshots`: lower survivorship bias than annual snapshots
- `monthly_snapshots`: lower survivorship bias than quarterly snapshots
- `exact_point_in_time`: best available mode when exact historical membership intervals are known

`dataset_metadata.json` records:

- `universe_mode`
- `universe_snapshot_count`
- `universe_snapshot_frequency`
- `survivorship_bias_status`
- `universe_start_date`
- `universe_end_date`

## Bias And Limitations

- A static universe can create survivorship bias
- Fundamentals must be point-in-time to avoid look-ahead bias
- Free price data can contain mapping or corporate action issues
- The included demo dataset is synthetic and only meant to verify the pipeline
- A current NIFTY 500 constituent snapshot is not the same as point-in-time historical index membership

## Snapshot Workflow

1. Add dated snapshot CSV files to `data/universe_snapshots/`.
2. Or import raw snapshot files with `python -m src.import_universe_snapshots --input-dir raw_snapshots/ --source-type ...`.
3. Run `python -m src.build_universe_history`.
4. Validate inputs with `python -m src.validate_data --start-date YYYY-MM-DD --end-date YYYY-MM-DD --no-download`.
5. Run the backtest with `python -m src.price_only_v1 ...`.

Interpretation notes:

- `universe_mode=current_snapshot_only` means the backtest should not be described as survivorship-bias-reduced.
- `survivorship_bias_status=high` means current constituents were projected backward.
- `survivorship_bias_status=medium` usually means annual snapshots.
- `survivorship_bias_status=low` usually means quarterly or monthly snapshots.

## Tests

Run the tests with:

```bash
python -m pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
