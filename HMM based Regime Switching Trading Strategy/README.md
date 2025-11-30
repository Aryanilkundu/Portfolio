Markov Regime Mixing Backtests
==============================

Overview
--------
This repository contains a small backtesting research framework that:
- Builds an equal-weight index (EQ) from an NSE500-like universe.
- Implements multiple Top-K strategies (Binomial, Poisson, WVF-LR).
- Fits an HMM (Gaussian) to EQ-derived features and classifies regimes.
- Mixes strategy weight matrices according to inferred regimes and runs backtests using `bt`.

Files of interest
-----------------
- `run_markov.py`  : Main script to run training, infer regimes and run test-period backtests. Supports CLI args for dates.
- `robustness.py`  : Walk-forward / robustness analysis across multiple split dates; produces `results/` plots and CSV summary.
- `markov_regime_switcher.py` : HMM fitting, state classification, mixing logic and `bt` helper for weighting.
- `binomial_strategy.py`, `wvf_lr_strategy.py` : Individual strategy implementations returning precomputed weight DataFrames.
- `data_loader.py`  : (helper) data handling utilities.
- `data/`           : Input datasets (feather files). Do NOT commit large data files to git.

Quick start
-----------

1. Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies (create `requirements.txt` as needed):

```powershell
pip install -r requirements.txt
```

3. First, download the price data required by the examples by running:

```powershell
python data_loader.py --symbols_csv ind_nifty500list.csv --years 20
```
- The repository expects two feather files under `data/`:
  - `nse500_prices.feather`
  - `nse500_prices_low.feather`


4. Run the main script (defaults are embedded in the script):

```powershell
python run_markov.py
```

5. Run `run_markov` with custom date ranges using CLI flags:

```powershell
python run_markov.py --start-date 2007-01-01 --split-date 2015-01-01 --end-date 2019-10-01
```

Running robustness / walk-forward
---------------------------------
`robustness.py` runs the analysis across a list of split dates and writes outputs to the `results/` folder. Run it from the repo root:

```powershell
python robustness.py
```

Outputs will be written to `results/walkforward_summary.csv`, and PNG plots.

