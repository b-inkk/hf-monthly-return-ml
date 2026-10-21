# Data

Monthly panels used in the paper. Raw minute bars are not included.

| File | Role |
|------|------|
| `monthly_features_1998_2017.csv` | Training panel: 26,523 stock-months, 126 tickers, columns `YearMonth`, `ticker`, `x1`–`x13`, `y` |
| `monthly_features_2017_2023.csv` | Evaluation panel: 9,491 stock-months |

`y` is next-month log return. Feature names match the paper Table 1 (`x1_mom1m`, …, `x13_indprem`).
