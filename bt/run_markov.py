import argparse
import numpy as np
import pandas as pd
import bt
import matplotlib.pyplot as plt
from binomial_strategy import compute_binomial_weights
from wvf_lr_strategy import compute_wvf_lr_weights
from scipy import stats

from markov_regime_switcher import (
    build_eq_index_returns,
    classify_states_as_roles,
    build_regime_mixed_weights,
    create_markov_regime_backtest,
)
from hmmlearn.hmm import GaussianHMM

close = pd.read_feather("data/nse500_prices.feather")
close.set_index("Date", inplace=True)
close.index = pd.to_datetime(close.index)
close = close.ffill().dropna(axis=1, how="any")

low = pd.read_feather("data/nse500_prices_low.feather")
low.set_index("Date", inplace=True)
low.index = pd.to_datetime(low.index)
low = low.ffill()[close.columns]

n_assets = close.shape[1]

# Equal-weight buy & hold
w_eq = pd.DataFrame(
    np.tile(1.0 / n_assets, (len(close), n_assets)),
    index=close.index,
    columns=close.columns,
)

w_bin = compute_binomial_weights(close, top_k=50)
w_wvf = compute_wvf_lr_weights(close, low=low, top_k=50)

strategy_weights = {
    "EQ": w_eq,
    "BIN": w_bin,
    "WVF": w_wvf,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Markov regime backtests with configurable dates"
    )
    p.add_argument(
        "--start-date",
        dest="start_date",
        default="2007-01-01",
        help="Start date for training window (format YYYY-MM-DD).",
    )
    p.add_argument(
        "--split-date",
        dest="split_date",
        default="2015-01-01",
        help="Split date: train before this, test from this date onward.",
    )
    p.add_argument(
        "--end-date",
        dest="end_date",
        default="2019-10-01",
        help="End date for test window (format YYYY-MM-DD).",
    )
    return p.parse_args()

args = parse_args()
SPLIT_DATE = args.split_date  
START_DATE = args.start_date
END_DATE = args.end_date
log_ret_eq = build_eq_index_returns(close, w_eq)

#features: log returns, vol20, mom20, mom60
ret = log_ret_eq
vol20 = ret.rolling(20).std()
mom20 = ret.rolling(20).sum()
mom60 = ret.rolling(60).sum()

features = (
    pd.DataFrame(
        {
            "ret": ret,
            "vol20": vol20,
            "mom20": mom20,
            "mom60": mom60,
        }
    )
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

train_feat = features.loc[START_DATE: SPLIT_DATE]
test_feat  = features.loc[SPLIT_DATE : END_DATE]

train_X = train_feat[["ret", "vol20", "mom20", "mom60"]].values
full_X  = features[["ret", "vol20", "mom20", "mom60"]].values
model = GaussianHMM(
    n_components=3,
    covariance_type="diag",  
    n_iter=300,
    random_state=42,
)
model.min_covar = 1e-6

model.fit(train_X)

hidden_states_full = model.predict(full_X)
regimes_full = pd.Series(hidden_states_full, index=features.index, name="Regime")


train_states = regimes_full.loc[START_DATE: SPLIT_DATE]
train_stats = (
    features.assign(state=train_states)
    .loc[train_states.index]
    .groupby("state")
    .agg(
        mean_ret=("ret", "mean"),
        mean_vol=("vol20", "mean"),
        mean_mom20=("mom20", "mean"),
        mean_mom60=("mom60", "mean"),
    )
)

print("\n==== TRAIN STATE STATS ====")
print(train_stats)

role_map = classify_states_as_roles(train_stats)
print("\nROLE MAP (bull/neutral/bear -> state):", role_map)

bull_state    = role_map["bull"]
neutral_state = role_map["neutral"]
bear_state    = role_map["bear"]

regime_to_mix = {
    bull_state:    {"EQ": 0.6, "BIN": 0.1, "WVF": 0.3},
    neutral_state: {"EQ": 0.25, "BIN": 0.7, "WVF": 0.0},
    bear_state:    {"EQ": 0.0, "BIN": 0.2, "WVF": 0.0},
 }

regimes_test = regimes_full.loc[SPLIT_DATE : END_DATE ]

w_markov_full = build_regime_mixed_weights(
    regimes_test,
    strategy_weights,
    regime_to_mix,
)

close_test = close.loc[SPLIT_DATE : END_DATE]

w_eq_test     = w_eq.loc[close_test.index]
w_bin_test    = w_bin.loc[close_test.index]
w_wvf_test    = w_wvf.loc[close_test.index]
w_markov_test = w_markov_full.loc[close_test.index]

bt_eq_test = create_markov_regime_backtest(
    prices=close_test,
    weights=w_eq_test,
    name="EQ_BuyHold_Test",
)

bt_bin_test = create_markov_regime_backtest(
    prices=close_test,
    weights=w_bin_test,
    name="Binomial_Test",
)
bt_wvf_test = create_markov_regime_backtest(
    prices=close_test,
    weights=w_wvf_test,
    name="WVF_LR_Test",
)

bt_markov = create_markov_regime_backtest(
    prices=close_test,
    weights=w_markov_test,
    name="Markov_Regime_Test",
)

results = bt.run(
    bt_eq_test,
    bt_bin_test,
    bt_wvf_test,
    bt_markov,
)

results.display()
results.plot()
plt.show()
