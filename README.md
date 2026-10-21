# High-frequency data, monthly return prediction

Companion code and monthly feature files for:

**Machine learning applications with large high-frequency financial data**  
Ray Cai

Repository: https://github.com/b-inkk/hf-monthly-return-ml

This is an empirical learning project in the spirit of Gu, Kelly, and Xiu (2020). Minute bars are compressed to daily aggregates, then to a pooled monthly panel of 13 predictors. Models forecast next-month log returns for S&P 100 names. Out-of-sample return R² is negative; directional accuracy is about 54.6%.

Raw minute files are **not** in this repository.

## Layout

```
README.md              # this file
requirements.txt
LICENSE
paper/                 # LaTeX + figure PNGs
  main.tex
  references.bib
  Plots/
src/                   # extraction, training, backtest
data/                  # monthly feature CSVs only (no raw minutes)
results/
  models/              # fitted Elastic Net pipeline
  metrics/             # coefficients, predictions, tables
  plots/               # backtest and importance figures
```

Each of `paper/`, `src/`, `data/`, and `results/` has its own short README.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Train Elastic Net (optional; a fitted pipeline is already in `results/models/`):

```bash
python src/train_elastic_net_model.py
```

Backtest:

```bash
python src/run_corrected_backtesting.py
```

Default paths assume you run from the repository root.

## Feature construction (what the code actually does)

1. Minute file → last close of each day, daily volume, daily dollar volume.
2. Daily close-to-close log returns (overnight included).
3. Monthly panel: next-month log return `y`, predictors `x1`–`x13` (see the paper Table 1).

The saved Elastic Net uses `alpha=0.01`, `l1_ratio=0.1`, trained on 26,523 rows / 126 stocks.

## Compile the paper

The Overleaf free plan often times out on TikZ plus six large PNGs. This `paper/main.tex` drops TikZ. Upload `paper/main.tex`, `paper/references.bib`, and `paper/Plots/*.png` as a small project.

Locally (TeX Live / MiKTeX):

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## License

Code is MIT (see `LICENSE`). Monthly feature files are derived from vendor minute data that is **not** redistributed here. Do not treat this repository as a license to the original market data.
