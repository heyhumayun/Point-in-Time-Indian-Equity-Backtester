# Indian Equity Portfolio Research Report

## Backtest Configuration

- Start date: 2007-09-17
- End date: 2025-12-31
- Initial capital: INR 500,000
- Monthly contribution: INR 20,000
- Transaction cost rate: 0.20%
- Rebalance target count: 15
- Dataset type: downloaded
- Result label: Research

## Strategy Summary

- Universe: Nifty 500 symbols supplied through `data/processed/universe.csv`
- Rebalance timing: first trading day of each month
- Signal timestamp: previous trading day close
- Execution assumption: rebalance-day open with configurable transaction costs
- Exit rules: leave top 30 ranking or close below 200-day moving average

## Performance Summary

- Final Portfolio Value: 235,863,267.9480
- Total Return: 47.3326
- Cagr: 0.3984
- Xirr: 0.3261
- Max Drawdown: -0.9972
- Volatility: 116.3023
- Sharpe Ratio: 0.4023
- Sortino Ratio: 64.3054
- Calmar Ratio: 0.3995
- Turnover: 161.2603
- number_of_trades: 4318
- Win Rate: 0.0000
- Best Month: 0.2759
- Worst Month: -0.2713

## Benchmarks

### Nifty 50

- Final Portfolio Value: 16,902,854.2666
- Total Return: 2.5660
- Cagr: 0.2122
- Xirr: 0.1138
- Max Drawdown: -0.4723
- Volatility: 0.2121
- Sharpe Ratio: 1.0380
- Sortino Ratio: 1.3861
- Calmar Ratio: 0.4493
- Turnover: 0.0000
- number_of_trades: 0
- Win Rate: nan
- Best Month: 0.3146
- Worst Month: -0.2408

### Nifty 500

- Final Portfolio Value: 19,383,281.5341
- Total Return: 2.9720
- Cagr: 0.2213
- Xirr: 0.1230
- Max Drawdown: -0.5233
- Volatility: 0.2073
- Sharpe Ratio: 1.0928
- Sortino Ratio: 1.4049
- Calmar Ratio: 0.4229
- Turnover: 0.0000
- number_of_trades: 0
- Win Rate: nan
- Best Month: 0.3826
- Worst Month: -0.2476

## Assumptions And Limitations

- Point-in-time fundamentals are required to avoid look-ahead bias.
- A static universe file introduces survivorship bias if historical membership dates are missing.
- Free Yahoo Finance data can contain symbol mapping issues, missing corporate actions, and benchmark inconsistencies.
- AI-style research fields are placeholders and default to neutral values unless supplied in the fundamentals file.

## Warnings

- Fundamentals were intentionally ignored for this Price-Only V1 run.