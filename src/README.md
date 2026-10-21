# Source

Run all scripts from the **repository root**.

| Script | Purpose |
|--------|---------|
| `extract_daily_eod_1998_2017.py` | Minute files → daily EOD CSVs (needs local minute data) |
| `fixed_monthly_feature_extractor.py` | Daily EOD → monthly `x1`–`x13` + `y` |
| `stock_industry_mapping.py` | Ticker → sector ETF |
| `smart_feature_imputation.py` | Fill remaining industry-feature gaps |
| `train_elastic_net_model.py` | Fit Elastic Net on `data/monthly_features_1998_2017.csv` |
| `run_corrected_backtesting.py` | Score `data/monthly_features_2017_2023.csv` |

A fitted pipeline is already stored in `results/models/elastic_net_pipeline.pkl`.
