# Indian Equity Portfolio Backtest Report

## Backtest Configuration

- Start date: 2015-01-01
- End date: 2025-12-31
- Initial capital: INR 100,000
- Monthly contribution: INR 20,000
- Transaction cost rate: 0.20%
- Rebalance target count: 15

## Strategy Summary

- Universe: Nifty 500 symbols supplied through `data/processed/universe.csv`
- Rebalance timing: first trading day of each month
- Signal timestamp: previous trading day close
- Execution assumption: rebalance-day open with configurable transaction costs
- Exit rules: leave top 30 ranking or close below 200-day moving average

## Performance Summary

- Final Portfolio Value: 3,621,828.3924
- Total Return: 1.0579
- Cagr: 0.6702
- Xirr: 0.1933
- Max Drawdown: -0.0608
- Volatility: 0.1713
- Sharpe Ratio: 2.9735
- Sortino Ratio: 13.7956
- Calmar Ratio: 11.0226
- Turnover: 29.0853
- number_of_trades: 1138
- Win Rate: 0.0000
- Best Month: 0.2081
- Worst Month: -0.0256

## Benchmarks

### Nifty 50

- Final Portfolio Value: 2,692,890.6003
- Total Return: 0.5301
- Cagr: 0.6009
- Xirr: 0.1141
- Max Drawdown: -0.1293
- Volatility: 0.1974
- Sharpe Ratio: 2.3976
- Sortino Ratio: 6.6151
- Calmar Ratio: 4.6478
- Turnover: 0.0000
- number_of_trades: 0
- Win Rate: nan
- Best Month: 0.2299
- Worst Month: -0.0618

### Nifty 500

- Final Portfolio Value: 2,868,164.2205
- Total Return: 0.6296
- Cagr: 0.6154
- Xirr: 0.1309
- Max Drawdown: -0.0753
- Volatility: 0.1919
- Sharpe Ratio: 2.5057
- Sortino Ratio: 7.5402
- Calmar Ratio: 8.1682
- Turnover: 0.0000
- number_of_trades: 0
- Win Rate: nan
- Best Month: 0.2284
- Worst Month: -0.0412

## Assumptions And Limitations

- Point-in-time fundamentals are required to avoid look-ahead bias.
- A static universe file introduces survivorship bias if historical membership dates are missing.
- Free Yahoo Finance data can contain symbol mapping issues, missing corporate actions, and benchmark inconsistencies.
- AI-style research fields are placeholders and default to neutral values unless supplied in the fundamentals file.

## Warnings

- Price history starts at 2018-01-01, which may be too late to compute a full 12-month momentum signal by 2015-01-01.
- Requested start date 2015-01-01 was shifted to 2019-01-01 because local data coverage starts later.
- No eligible securities on 2019-01-01 for the price-only strategy.