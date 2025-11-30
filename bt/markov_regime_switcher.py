from typing import Dict, Tuple

import numpy as np
import pandas as pd
import bt
from hmmlearn.hmm import GaussianHMM

def build_eq_index_returns(
    prices: pd.DataFrame,
    w_eq: pd.DataFrame,
) -> pd.Series:
    """
    Build log-returns of equal-weight EQ portfolio from prices and equal
    weights.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (Date x Tickers).
    w_eq : pd.DataFrame
        Equal-weight matrix (Date x Tickers), e.g. 1/N per stock.

    Returns
    -------
    pd.Series
        Daily log returns of EQ portfolio.
    """
    # simple daily return of portfolio
    port_ret = (prices.pct_change().fillna(0.0) * w_eq).sum(axis=1)
    log_ret = np.log1p(port_ret)
    return log_ret


def fit_hmm_eq_returns_vol(
    log_ret: pd.Series,
    vol_window: int = 20,
    n_states: int = 3,
    random_state: int = 42,
) -> Tuple[pd.Series, GaussianHMM, pd.DataFrame]:
    vol = log_ret.rolling(vol_window).std()

    # align & drop NaN
    df_feat = pd.DataFrame({"ret": log_ret, "vol": vol}).dropna()

    X = df_feat[["ret", "vol"]].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(X)
    hidden_states = model.predict(X)

    regimes = pd.Series(hidden_states, index=df_feat.index, name="Regime")

    stats = (
        df_feat.assign(state=hidden_states)
        .groupby("state")
        .agg(mean_ret=("ret", "mean"), mean_vol=("vol", "mean"))
    )

    return regimes, model, stats


def classify_states_as_roles(
    state_stats: pd.DataFrame,
) -> Dict[str, int]:
    """
    Classify HMM states into economic roles: bull / neutral / bear.

    Heuristic:
        - bull  = state with highest mean_ret
        - bear  = state with lowest mean_ret
        - neut  = remaining state
    Returns
    -------
    dict
        {"bull": state_id, "neutral": state_id, "bear": state_id}
    """
    sorted_states = state_stats.sort_values("mean_ret")
    bear_state = sorted_states.index[0]
    bull_state = sorted_states.index[-1]

    neutral_state = [s for s in state_stats.index if s not in (bear_state, bull_state)][0]

    return {
        "bull": int(bull_state),
        "neutral": int(neutral_state),
        "bear": int(bear_state),
    }


def build_regime_mixed_weights(
    regimes: pd.Series,
    strategy_weights: Dict[str, pd.DataFrame],
    regime_to_mix: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    all_idx = None
    all_cols = None
    for w in strategy_weights.values():
        if all_idx is None:
            all_idx = w.index
            all_cols = w.columns
        else:
            all_idx = all_idx.intersection(w.index)
            all_cols = all_cols.intersection(w.columns)
    sw = {
        k: v.reindex(index=all_idx, columns=all_cols).fillna(0.0)
        for k, v in strategy_weights.items()
    }
    regimes = regimes.reindex(all_idx).ffill().bfill()
    final_weights = pd.DataFrame(0.0, index=all_idx, columns=all_cols)

    for dt in all_idx:
        state = regimes.loc[dt]
        mix = regime_to_mix.get(int(state), {})
        w_row = np.zeros(len(all_cols), dtype=float)
        for strat_key, alpha in mix.items():
            if alpha == 0 or strat_key not in sw:
                continue
            # add alpha * strategy weights
            w_row += alpha * sw[strat_key].loc[dt].values
        total_abs = np.abs(w_row).sum()
        if total_abs > 0:
            w_row /= total_abs

        final_weights.loc[dt] = w_row

    return final_weights


class WeighFromDF(bt.Algo):

    def __init__(self, weights: pd.DataFrame):
        self.weights = weights

    def __call__(self, target) -> bool:
        dt = target.now
        if dt not in self.weights.index:
            return False
        w = self.weights.loc[dt].dropna().to_dict()
        target.temp["weights"] = w
        return True


def create_markov_regime_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    name: str = "MarkovRegimeMix",
) -> bt.Backtest:
    """
    Build a bt.Backtest using precomputed regime-mixed weights.
    """
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