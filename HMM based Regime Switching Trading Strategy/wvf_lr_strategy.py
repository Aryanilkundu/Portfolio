"""
wvf_lr_strategy.py

Williams Vix Fix + Linear Regression Strategy (WVF-LR)

Definitions (per stock):

1. Williams Vix Fix (WVF):
   WVF_t = ((max_{i in [t-M+1, t]} Close_i - Low_t) /
            max_{i in [t-M+1, t]} Close_i) * 100

   where M is the WVF lookback (default: 5).

   Bollinger-style bands on WVF:
       UpperBand = SMA_bbl(WVF) + k * stdev_bbl(WVF)
       LowerBand = SMA_bbl(WVF) - k * stdev_bbl(WVF)

   Also define percentile range level:
       rangeHigh_t = max_{i in [t-lb+1, t]} WVF_i * ph

   Defaults:
       bbl = 20, k = 2, ph = 0.95, lb = 10.

2. Linear Regression on WVF:
   Based on past pd (default: 5) days of WVF, fit a linear regression and
   compute the predicted WVF at time t, denoted w_hat_t.

   Colour rules:
       if w_hat_t > w_hat_{t-1} and w_hat_t > 0:      'green'
       if w_hat_t >= w_hat_{t-1} and w_hat_t < 0:     'orange'
       if w_hat_t < w_hat_{t-1} and w_hat_t > 0:      'lime'
       if w_hat_t < w_hat_{t-1} and w_hat_t < 0:      'red'

3. Signal Logic (per stock):
   Entry:
       If (WVF_t >= UpperBand_t OR WVF_t >= rangeHigh_t)
       AND colour_t == 'lime'  -> go long.

   Exit:
       If colour_t in {'orange', 'red'}               -> exit (flat).

Portfolio:
    - Long-only
    - Equal-weight across all stocks that are long on a given day.

"""

from typing import Optional

import numpy as np
import pandas as pd
import bt

def compute_wvf(
    close: pd.DataFrame,
    low: Optional[pd.DataFrame] = None,
    M: int = 5,
) -> pd.DataFrame:
    if low is None:
        low = close
    low = low.reindex_like(close)
    rolling_max = close.rolling(window=M, min_periods=M).max()
    wvf = (rolling_max - low) / rolling_max * 100.0

    return wvf


def compute_wvf_bands_and_range(
    wvf: pd.DataFrame,
    bbl: int = 20,
    k: float = 2.0,
    ph: float = 0.95,
    lb: int = 10,
):
    mean_wvf = wvf.rolling(window=bbl, min_periods=bbl).mean()
    std_wvf = wvf.rolling(window=bbl, min_periods=bbl).std()
    upper_band = mean_wvf + k * std_wvf
    lower_band = mean_wvf - k * std_wvf
    range_high = wvf.rolling(window=lb, min_periods=lb).max() * ph
    return upper_band, lower_band, range_high


def _linreg_predict_last(y: np.ndarray) -> float:
    if np.all(np.isnan(y)):
        return np.nan
    x = np.arange(len(y), dtype=float)
    a, b = np.polyfit(x, y, 1)
    return a * x[-1] + b


def compute_wvf_regression(
    wvf: pd.DataFrame,
    pd_window: int = 5,
) -> pd.DataFrame:
    w_hat = wvf.rolling(
        window=pd_window,
        min_periods=pd_window
    ).apply(_linreg_predict_last, raw=True)

    return w_hat


def compute_colour_masks(
    w_hat: pd.DataFrame,
) -> dict:
    """
    Compute boolean masks for regression colours: green, orange, lime, red.

    Returns a dict with keys 'green', 'orange', 'lime', 'red'.
    """
    w_prev = w_hat.shift(1)

    green = (w_hat > w_prev) & (w_hat > 0)
    orange = (w_hat >= w_prev) & (w_hat < 0)
    lime = (w_hat < w_prev) & (w_hat > 0)
    red = (w_hat < w_prev) & (w_hat < 0)

    return {
        "green": green,
        "orange": orange,
        "lime": lime,
        "red": red,
    }


def compute_wvf_lr_weights(
    close: pd.DataFrame,
    low: Optional[pd.DataFrame] = None,
    M: int = 5,
    bbl: int = 20,
    k: float = 2.0,
    ph: float = 0.95,
    lb: int = 10,
    pd_window: int = 5,
    top_k: int = 50,
) -> pd.DataFrame:
    
    wvf = compute_wvf(close, low, M=M)
    upper_band, lower_band, range_high = compute_wvf_bands_and_range(
        wvf, bbl=bbl, k=k, ph=ph, lb=lb
    )
    w_hat = compute_wvf_regression(wvf, pd_window=pd_window)
    colours = compute_colour_masks(w_hat)
    lime = colours["lime"]
    orange = colours["orange"]
    red = colours["red"]

    trigger_wvf = (wvf >= upper_band) | (wvf >= range_high)
    entry = trigger_wvf & lime
    exit_cond = orange | red

    index = close.index
    cols = close.columns
    positions = pd.DataFrame(0.0, index=index, columns=cols)

    n_dates = len(index)

    entry_np = entry.to_numpy(dtype=bool)
    exit_np = exit_cond.to_numpy(dtype=bool)

    for j, col in enumerate(cols):
        pos = 0.0
        entry_col = entry_np[:, j]
        exit_col = exit_np[:, j]
        col_pos = np.zeros(n_dates, dtype=float)

        for i in range(n_dates):
            if entry_col[i]:
                pos = 1.0
            elif exit_col[i]:
                pos = 0.0
            col_pos[i] = pos

        positions.iloc[:, j] = col_pos
    valid = close.notna()
    positions = positions.where(valid, 0.0)
    signal_strength = wvf * positions
    signal_strength = signal_strength.fillna(0.0)

    ranks = signal_strength.rank(axis=1, ascending=False, method="first")
    mask_top = (ranks <= top_k) & (signal_strength > 0)

    positions_topk = mask_top.astype(float)

    row_sum = positions_topk.sum(axis=1).replace(0, np.nan)
    weights = positions_topk.div(row_sum, axis=0).fillna(0.0)

    weights = weights.reindex_like(close).fillna(0.0)
    weights = weights.div(weights.abs().sum(axis=1), axis=0).fillna(0.0)

    return weights

class WeighFromDF(bt.Algo):
    """
    Custom bt.Algo to set target.temp['weights'] from a precomputed
    weights DataFrame (DateTimeIndex x tickers).
    """

    def __init__(self, weights: pd.DataFrame):
        self.weights = weights

    def __call__(self, target) -> bool:
        dt = target.now
        if dt not in self.weights.index:
            return False

        w = self.weights.loc[dt].dropna().to_dict()
        target.temp["weights"] = w
        return True


def create_wvf_lr_backtest(
    close: pd.DataFrame,
    low: Optional[pd.DataFrame] = None,
    name: str = "WVF_LR_NSE500",
    M: int = 5,
    bbl: int = 20,
    k: float = 2.0,
    ph: float = 0.95,
    lb: int = 10,
    pd_window: int = 5,
    top_k: int = 50,
) -> bt.Backtest:

    weights = compute_wvf_lr_weights(
        close,
        low=low,
        M=M,
        bbl=bbl,
        k=k,
        ph=ph,
        lb=lb,
        pd_window=pd_window,
        top_k=top_k
    )

    strat = bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(),
            bt.algos.SelectAll(),
            WeighFromDF(weights),
            bt.algos.Rebalance(),
        ],
    )

    return bt.Backtest(strat, close,commissions=lambda q, p: abs(q) * p * 0.0006)
