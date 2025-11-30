import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from hmmlearn.hmm import GaussianHMM
import os
import bt

from binomial_strategy import compute_binomial_weights
from wvf_lr_strategy import compute_wvf_lr_weights

from markov_regime_switcher import (
    build_eq_index_returns,
    classify_states_as_roles,
    build_regime_mixed_weights,
    create_markov_regime_backtest,
)

os.makedirs("results", exist_ok=True)

close = pd.read_feather("data/nse500_prices.feather")
close.set_index("Date", inplace=True)
close.index = pd.to_datetime(close.index)
close = close.ffill().dropna(axis=1)

low = pd.read_feather("data/nse500_prices_low.feather")
low.set_index("Date", inplace=True)
low.index = pd.to_datetime(low.index)
low = low.ffill()[close.columns]

BASELINE_NAME = "Binomial_Test"
MARKOV_NAME = "Markov_Regime_Test"

split_dates = pd.date_range("2014-01-01", "2024-01-01", freq="12M").strftime("%Y-%m-%d").tolist()

GLOBAL_START = "2006-01-01"
GLOBAL_END = "2025-10-01"

HMM_COMPONENTS = 3
HMM_RANDOM_SEED = 42

N_BOOT = 2000
BLOCK_SIZE = 5
RNG = np.random.default_rng(42)

def fit_hmm_and_regimes(features, train_start, split_date, n_components=3, random_state=42):
    train_feat = features.loc[train_start: split_date]
    train_X = train_feat[["ret", "vol20", "mom20", "mom60"]].values
    full_X = features[["ret", "vol20", "mom20", "mom60"]].values
    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=300, random_state=random_state)
    model.min_covar = 1e-6
    model.fit(train_X)
    hidden_states_full = model.predict(full_X)
    return model, pd.Series(hidden_states_full, index=features.index)

def build_features_and_align(close, w_eq, start_date, end_date):
    ret = build_eq_index_returns(close, w_eq)
    vol20 = ret.rolling(20).std()
    mom20 = ret.rolling(20).sum()
    mom60 = ret.rolling(60).sum()
    feat = pd.DataFrame({"ret": ret, "vol20": vol20, "mom20": mom20, "mom60": mom60})
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat.loc[start_date:end_date]

def create_results_for_split(split_date,
                             start_date=GLOBAL_START,
                             end_date=GLOBAL_END,
                             top_k=50,
                             n_components=HMM_COMPONENTS,
                             random_state=HMM_RANDOM_SEED,
                             regime_to_mix_override=None):

    n_assets = close.shape[1]
    w_eq = pd.DataFrame(np.tile(1.0 / n_assets, (len(close), n_assets)), index=close.index, columns=close.columns)
    w_bin = compute_binomial_weights(close, top_k=top_k)
    w_wvf = compute_wvf_lr_weights(close, low=low, top_k=top_k)
    strategy_weights = {"EQ": w_eq, "BIN": w_bin, "WVF": w_wvf}

    features = build_features_and_align(close, w_eq, start_date, end_date)
    if split_date not in features.index:
        idx = features.index.searchsorted(pd.to_datetime(split_date))
        split_date = str(features.index[idx].date())

    model, regimes_full = fit_hmm_and_regimes(features, start_date, split_date, n_components, random_state)

    train_states = regimes_full.loc[start_date: split_date]
    train_stats = (features.assign(state=train_states).loc[train_states.index]
                   .groupby("state")
                   .agg(mean_ret=("ret", "mean"),
                        mean_vol=("vol20", "mean"),
                        mean_mom20=("mom20", "mean"),
                        mean_mom60=("mom60", "mean")))
    rm = classify_states_as_roles(train_stats)
    bull, neutral, bear = rm["bull"], rm["neutral"], rm["bear"]

    regime_to_mix = regime_to_mix_override or {
        bull:    {"EQ": 0.6, "BIN": 0.1, "WVF": 0.3},
        neutral: {"EQ": 0.25, "BIN": 0.7,"WVF": 0.0},
        bear:    {"EQ": 0.0, "BIN": 0.2, "WVF": 0.0},
    }

    regimes_test = regimes_full.loc[split_date: end_date]
    w_markov = build_regime_mixed_weights(regimes_test, strategy_weights, regime_to_mix)

    close_test = close.loc[split_date: end_date]
    w_eq_test = w_eq.loc[close_test.index]
    w_bin_test = w_bin.loc[close_test.index]
    w_markov_test = w_markov.loc[close_test.index]

    bt_eq = create_markov_regime_backtest(close_test, w_eq_test, "EQ_BuyHold_Test")
    bt_bin = create_markov_regime_backtest(close_test, w_bin_test, "Binomial_Test")
    bt_mark = create_markov_regime_backtest(close_test, w_markov_test, "Markov_Regime_Test")

    results = bt.run(bt_eq, bt_bin, bt_mark)
    nav_markov = results[MARKOV_NAME].prices
    nav_base = results[BASELINE_NAME].prices

    d_m = nav_markov.pct_change().dropna()
    d_b = nav_base.pct_change().dropna()
    return results, nav_markov, nav_base, d_m, d_b

def paired_ttest(a, b):
    a, b = a.align(b, join="inner")
    a = a.dropna(); b = b.dropna()
    if len(a) < 5:
        return {"t": np.nan, "p": np.nan, "n": len(a), "mean_diff": np.nan}
    t, p = stats.ttest_rel(a, b, nan_policy='omit')
    return {"t": float(t), "p": float(p), "n": len(a), "mean_diff": float((a - b).mean())}

def circular_block_bootstrap_paired(a, b, block_size=BLOCK_SIZE, n_boot=N_BOOT, rng=RNG):
    a, b = a.align(b, join="inner")
    a = a.dropna().values; b = b.dropna().values
    n = len(a)
    if n == 0:
        return {"samples": [], "mean": np.nan, "ci95": (np.nan, np.nan), "p_gt": np.nan}
    nb = int(np.ceil(n / block_size))
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = []
        for _ in range(nb):
            s = rng.integers(0, n)
            idx.extend([(s + k) % n for k in range(block_size)])
        idx = np.array(idx[:n])
        nav1 = np.prod(1 + a[idx])
        nav2 = np.prod(1 + b[idx])
        samples[i] = nav1 - nav2
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"samples": samples, "mean": samples.mean(), "ci95": (lo, hi), "p_gt": np.mean(samples > 0)}

summary = []

for split in split_dates:
    try:
        res, nav_m, nav_b, d_m, d_b = create_results_for_split(split)
    except Exception as e:
        print("Skipping", split, e)
        continue

    def cagr(nav):
        daily = nav.pct_change().dropna()
        n = len(daily)
        return (nav.iloc[-1] / nav.iloc[0]) ** (252 / n) - 1 if n > 0 else np.nan

    c_mark = cagr(nav_m)
    c_base = cagr(nav_b)
    d_cagr = c_mark - c_base if (not pd.isna(c_mark) and not pd.isna(c_base)) else np.nan

    tt = paired_ttest(d_m, d_b)
    bb = circular_block_bootstrap_paired(d_m, d_b)

    summary.append({
        "split": split,
        "cagr_base": c_base,
        "cagr_markov": c_mark,
        "delta_cagr": d_cagr,
        "tstat": tt["t"],
        "pval": tt["p"],
        "n_days": tt["n"],
        "mean_daily_diff": tt["mean_diff"],
        "boot_mean_diff": bb["mean"],
        "boot_ci_lower": bb["ci95"][0],
        "boot_ci_upper": bb["ci95"][1],
        "boot_p_markov_gt": bb["p_gt"],
    })

wf_summary = pd.DataFrame(summary).set_index("split")
wf_summary.to_csv("results/walkforward_summary.csv")

plt.figure(figsize=(8,3))
plt.bar(wf_summary.index, wf_summary["delta_cagr"].fillna(0))
plt.axhline(0, color='black', linewidth=0.6)
plt.xticks(rotation=45)
plt.ylabel("ΔCAGR")
plt.title("ΔCAGR Across Splits")
plt.tight_layout()
plt.savefig("results/delta_cagr_plot.png", dpi=300)
plt.close()

plt.figure(figsize=(8,3))
plt.bar(wf_summary.index, wf_summary["boot_p_markov_gt"])
plt.xticks(rotation=45)
plt.ylabel("P(Markov > Baseline)")
plt.title("Bootstrap Probability Across Splits")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("results/bootstrap_prob_plot.png", dpi=300)
plt.close()
