"""
Implements the Binomial Strategy described as:

1. Define an up-bar if close_t > close_{t-1}.
2. For past M days (default 100), compute up-bar probability
       p = (# up-bars in last M days) / M
3. For past n days (default 20), let k = # up-bars.
   Assume X ~ Binomial(n, p) and compute
       p_n = P(X <= k)
4. Compute EMA10 and EMA20 of p_n.
5. Go long when EMA10 > EMA20, exit when EMA10 < EMA20.
   Portfolio: equal-weight all longs each day.

"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binom
import bt


def compute_binomial_signal(
    prices: pd.DataFrame,
    M: int = 100,
    n: int = 20,
) -> pd.DataFrame:
    """
    Compute the binomial CDF signal p_n for each date and ticker.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame (DateTimeIndex x tickers), e.g. Adj Close.
    M : int
        Lookback window for estimating p (default 100).
    n : int
        Lookback window for k and binomial(n, p) (default 20).

    Returns
    -------
    pd.DataFrame
        DataFrame of p_n values (same shape as prices).
    """
    up = prices.diff() > 0 
    up_M = up.rolling(window=M, min_periods=M).sum()
    p_hat = up_M / float(M)
    k = up.rolling(window=n, min_periods=n).sum()
    p_vals = p_hat.to_numpy(dtype=float)
    k_vals = k.to_numpy(dtype=float)
    cdf_vals = binom.cdf(k_vals, n, p_vals)
    p_n = pd.DataFrame(cdf_vals, index=prices.index, columns=prices.columns)
    return p_n


def compute_binomial_emas(
    prices: pd.DataFrame,
    M: int = 100,
    n: int = 20,
    ema_fast: int = 10,
    ema_slow: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute EMA_fast and EMA_slow of the binomial CDF signal.

    Returns
    -------
    (ema_fast_df, ema_slow_df)
    """
    p_n = compute_binomial_signal(prices, M=M, n=n)

    ema_fast_df = p_n.ewm(span=ema_fast, adjust=False, min_periods=ema_fast).mean()
    ema_slow_df = p_n.ewm(span=ema_slow, adjust=False, min_periods=ema_slow).mean()

    return ema_fast_df, ema_slow_df

def compute_binomial_weights(
    prices: pd.DataFrame,
    M: int = 100,
    n: int = 20,
    ema_fast: int = 10,
    ema_slow: int = 20,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Compute daily portfolio weights for the Binomial Strategy.

    Logic:
        - Positions: long when EMA_fast > EMA_slow, flat otherwise,
          with forward-filled state (stay long until exit).
        - Signal strength: (EMA_fast - EMA_slow), only where position is long.
        - Top-K, signal-weighted:
            * each day keep top_k names by signal_strength
            * weights ∝ signal_strength
    """
    ema_fast_df, ema_slow_df = compute_binomial_emas(
        prices, M=M, n=n, ema_fast=ema_fast, ema_slow=ema_slow
    )
    raw_signal = (ema_fast_df > ema_slow_df).astype(int)
    positions = raw_signal.ffill().fillna(0)
    valid = prices.notna()
    positions = positions.where(valid, 0)
    signal_strength = (ema_fast_df - ema_slow_df).clip(lower=0)
    signal_strength = signal_strength * positions
    signal_strength = signal_strength.fillna(0.0)
    ranks = signal_strength.rank(axis=1, ascending=False, method="first")
    mask_top = (ranks <= top_k) & (signal_strength > 0)

    raw_w = signal_strength.where(mask_top, 0.0)
    row_sum = raw_w.sum(axis=1).replace(0, np.nan)
    weights = raw_w.div(row_sum, axis=0).fillna(0.0)
    weights = weights.reindex_like(prices).fillna(0.0)
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

def create_binomial_backtest(
    prices: pd.DataFrame,
    name: str = "BinomialStrategy",
    M: int = 100,
    n: int = 20,
    ema_fast: int = 10,
    ema_slow: int = 20,
    top_k: int = 50,
) -> bt.Backtest:
    weights = compute_binomial_weights(
        prices, M=M, n=n, ema_fast=ema_fast, ema_slow=ema_slow, top_k=top_k
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

    return bt.Backtest(strat, prices,commissions=lambda q, p: abs(q) * p * 0.0006)