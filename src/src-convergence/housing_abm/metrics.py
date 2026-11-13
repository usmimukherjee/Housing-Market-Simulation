"""Calibration / validation metrics and the explicit training-convergence criterion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RATIO_YEARS, TrainParams, ValScore


def fit_metrics(sim: pd.DataFrame, obs: pd.DataFrame, obs_ratio: pd.Series, years) -> dict:
    """Compare one simulated path (indexed by year) with observations over ``years``."""
    s = sim.set_index("year").loc[years]
    o = obs.loc[years]
    e = s["median"] - o["median"]
    out = {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "mape": float(np.mean(np.abs(e) / o["median"]) * 100),
        "smape": float(np.mean(2 * np.abs(e) / (np.abs(s["median"]) + o["median"])) * 100),
        "bias_pct": float(np.mean(e / o["median"]) * 100),
    }
    ss_tot = np.sum((o["median"] - o["median"].mean()) ** 2)
    out["r2"] = float(1 - np.sum(e ** 2) / ss_tot) if len(years) > 2 and ss_tot > 0 else np.nan
    lq = [np.log(s[c] / o[c]) for c in ["p25", "median", "p75"]]
    out["log_rmse_quantiles"] = float(np.sqrt(np.mean(np.concatenate(lq) ** 2)))
    ry = [y for y in years if y in RATIO_YEARS]
    if ry:
        re = s.loc[ry, "ratio"] - obs_ratio.loc[ry]
        out["ratio_mae"] = float(np.mean(np.abs(re)))
        out["ratio_rmse"] = float(np.sqrt(np.mean(re ** 2)))
    else:
        out["ratio_mae"] = out["ratio_rmse"] = np.nan
    if len(years) > 1:
        n = years[-1] - years[0]
        sim_g = (s["median"].iloc[-1] / s["median"].iloc[0]) ** (1 / n) - 1
        obs_g = (o["median"].iloc[-1] / o["median"].iloc[0]) ** (1 / n) - 1
        out["growth_sim_pct"] = float(sim_g * 100)
        out["growth_obs_pct"] = float(obs_g * 100)
    return out


def validation_score(m: dict, w: ValScore = ValScore()) -> float:
    """validation_score = log-RMSE over P25/median/P75 + lambda_1 x ratio RMSE.
    Volume is not scored (lambda_2 = 0): listing and buyer flows are fixed by assumption."""
    r = m["ratio_rmse"] if np.isfinite(m.get("ratio_rmse", np.nan)) else 0.0
    return m["log_rmse_quantiles"] + w.ratio_weight * r


def convergence(ck: pd.DataFrame, tp: TrainParams) -> dict:
    """Apply the convergence criterion to the checkpoint table of one run.

    A checkpoint j (j >= 1) is *stable* when all three hold:
      * |G_j - G_{j-1}| / max(|G_{j-1}|, 0.1) < conv_reward_rel, where G is the greedy reward per
        negotiation on training years averaged over the last five checkpoints (moving average);
      * policy change rate (share of fixed probe states whose greedy action changed) < conv_policy_change;
      * the validation score did not improve on the best earlier score by more than conv_val_improve.
    The run has converged at the first checkpoint that ends conv_k consecutive stable checkpoints.
    """
    g = ck["greedy_reward"].rolling(5, min_periods=1).mean().to_numpy()
    pc = ck["policy_change"].to_numpy()
    v = ck["val_score"].to_numpy()
    best_prev = np.minimum.accumulate(np.r_[np.inf, v[:-1]])
    stable = np.zeros(len(ck), bool)
    for j in range(1, len(ck)):
        dr = abs(g[j] - g[j - 1]) / max(abs(g[j - 1]), 0.1)
        dv = best_prev[j] - v[j]
        stable[j] = (dr < tp.conv_reward_rel) and (pc[j] < tp.conv_policy_change) \
            and (dv <= tp.conv_val_improve)
    run = 0
    for j in range(len(ck)):
        run = run + 1 if stable[j] else 0
        if run >= tp.conv_k:
            return {"converged": True, "episode": int(ck["episode"].iloc[j]), "stable": stable}
    return {"converged": False, "episode": None, "stable": stable}


def summarize(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean": np.nan, "sd": np.nan, "median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
    from scipy import stats
    sd = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": x.mean(), "sd": sd, "median": np.median(x), "ci_lo": x.mean() - h,
            "ci_hi": x.mean() + h, "n": n}
