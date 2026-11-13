"""Step 10. Market shock scenarios with the frozen policies, paired against no-shock runs.

Design (answers the review of the submitted shock section)
---------------------------------------------------------
* Shared market and frozen policies exactly as in steps 1-4 (no recalibration, no retraining,
  epsilon = 0, no learning). Shocks are implemented in ``housing_abm/shocks.py``.
* Main window 2022-2023: after every year used for calibration, training, model selection and
  checkpoint validation (2010-2021). The simulated market contains no COVID-19 shock, so a shock
  here is a counterfactual against a simulated no-shock baseline, never against observed data.
* Timing check 2016-2017: inside the simulated training years but before COVID-19.
* Every shocked run is paired with a no-shock run on the same market seed (and training seed);
  the two are identical until the shock begins (checked below).
* Prices are reported as a constant-quality index (median of price / dwelling quality, the
  simulated analogue of a repeat-sales index) and as the raw median, which also moves with the
  mix of dwellings that sell.

Scenarios (each shock lasts two years)
  supply        new listings -50%
  demand        new buyers   +50%
  tight         new listings -50% and new buyers +50%
  down_flow     new listings +50% and new buyers -50%
  down_credit   as down_flow, plus purchasing power (WTP and bid cap) -20% for all buyers
  supply_up     new listings +50% (supply increase on its own)
  demand_down   new buyers   -50% (demand drop on its own)
Dose checks in the main window: supply -30/-80%, demand +30/+100%, down_credit with WTP -10/-30%.

Writes results/results-scenario/shocks/{shock_paths.csv, shock_summary.csv,
shock_table.csv, shock_table.tex, shock_vs_theory.csv, qlearning_unvisited.csv} and graphs/Market-Shocks-*.png

Usage (from the package root): python src/src-scenario/market_shocks.py [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "src" / "src-convergence"))  # shared model package

from housing_abm.config import ALL_YEARS, FINAL_EVAL_SEEDS, RESULTS, TRAIN_SEEDS
from housing_abm.data import load
from housing_abm.experiment import load_bundle, load_state
from housing_abm.market import ACTIONS
from housing_abm.policies import DQNPolicy, QLearningPolicy, RandomPolicy, RulePolicy
from housing_abm.shocks import ShockMarket
from housing_abm.training import dqnparams_from, qparams_from, train_targets

OUT = PKG / "results" / "results-scenario" / "shocks"
# simulate five years past the observed record so the post-shock path is visible (no observed
# data are needed after 2024; the market only uses observed prices for the 2010 starting point)
SIM_YEARS = list(range(2010, 2029))
WINDOWS = {"2022-23": (2022, 2023), "2016-17": (2016, 2017)}
SCEN = {
    "supply": {"listings": 0.5},
    "demand": {"buyers": 1.5},
    "tight": {"listings": 0.5, "buyers": 1.5},
    "down_flow": {"listings": 1.5, "buyers": 0.5},
    "down_credit": {"listings": 1.5, "buyers": 0.5, "wtp": 0.8},
    "supply_up": {"listings": 1.5},
    "demand_down": {"buyers": 0.5},
}
DOSE = {
    "supply_30": {"listings": 0.7}, "supply_80": {"listings": 0.2},
    "demand_30": {"buyers": 1.3}, "demand_100": {"buyers": 2.0},
    "down_credit_10": {"listings": 1.5, "buyers": 0.5, "wtp": 0.9},
    "down_credit_30": {"listings": 1.5, "buyers": 0.5, "wtp": 0.7},
}
EXPECTED = {"supply": 1, "demand": 1, "tight": 1, "down_flow": -1, "down_credit": -1,
            "supply_up": -1, "demand_down": -1,
            "supply_30": 1, "supply_80": 1, "demand_30": 1, "demand_100": 1,
            "down_credit_10": -1, "down_credit_30": -1}
LABELS = {
    "supply": "Supply: new listings $-$50\\%",
    "demand": "Demand: new buyers $+$50\\%",
    "tight": "Combined tightening (supply $-$50\\%, demand $+$50\\%)",
    "down_flow": "Downturn: supply $+$50\\%, demand $-$50\\%",
    "down_credit": "Downturn with credit shock (also WTP $-$20\\%)",
    "supply_up": "Supply increase: new listings $+$50\\%",
    "demand_down": "Demand drop: new buyers $-$50\\%",
    "supply_30": "Supply $-$30\\%", "supply_80": "Supply $-$80\\%",
    "demand_30": "Demand $+$30\\%", "demand_100": "Demand $+$100\\%",
    "down_credit_10": "Downturn, WTP $-$10\\%", "down_credit_30": "Downturn, WTP $-$30\\%",
}
COLS = ["year", "listings", "buyers", "sales", "median", "cq_median", "sale_to_ask", "sell_through",
        "buyers_purchase", "ask_index"]


def runs():
    """(scenario, window) pairs to simulate, baseline first."""
    R = [("none", "none")]
    for w in WINDOWS:
        R += [(s, w) for s in SCEN]
    R += [(s, "2022-23") for s in DOSE]
    return R


def make_policy(unit):
    if unit["kind"] == "rule":
        return RulePolicy()
    if unit["kind"] == "random":
        return RandomPolicy(3, 77)
    b = load_bundle()
    n = len(ACTIONS[b.spec.actions])
    import json
    hp = json.loads((RESULTS / "tuning" / "best_hyperparameters.json").read_text())
    if unit["model"] == "qlearning":
        q = qparams_from(hp["qlearning"])
        p = QLearningPolicy(n, b.spec.state, b.market.max_rounds, q.alpha, q.gamma, unit["seed"], q.alpha_c)
    else:
        p = DQNPolicy(n, b.spec.state, dqnparams_from(hp["dqn"]), unit["seed"])
    p.load_state_dict(load_state(RESULTS / "training" / "main", f"{unit['model']}_s{unit['seed']}", "final"))
    return p


class Tracker:
    """Passes actions through; for Q-learning, counts decisions taken in never-visited states."""

    def __init__(self, p):
        self.p, self.kind, self.year = p, p.kind, None
        self.counts = {}

    def act(self, f, eps):
        if isinstance(self.p, QLearningPolicy):
            s = self.p.disc(f)
            c = self.counts.setdefault(self.year, [0, 0])
            c[0] += int((self.p.visits[s].sum(1) == 0).sum())
            c[1] += len(f)
        return self.p.act(f, eps)


def job(unit):
    d = load()
    b = load_bundle()
    targets = train_targets(d)
    base_pol = make_policy(unit)
    rows, unv = [], []
    for ms in FINAL_EVAL_SEEDS:
        for scen, win in runs():
            # reset policy randomness so every run of a pair sees the same tie-break draws
            if hasattr(base_pol, "rng"):
                base_pol.rng = np.random.default_rng([unit.get("seed") or 0, ms])
            pol = Tracker(base_pol) if unit["kind"] != "rule" else base_pol
            sh = {}
            if scen != "none":
                spec = {**SCEN, **DOSE}[scen]
                sh = {y: spec for y in WINDOWS[win]}
            m = ShockMarket(d, b.market, b.reward, b.spec, seed=ms, shocks=sh, targets=targets)
            for y in SIM_YEARS:
                if isinstance(pol, Tracker):
                    pol.year = y
                r = m.run_year(y, pol)
                rows.append({**{k: r[k] for k in COLS}, "scenario": scen, "window": win, "market_seed": ms,
                             "policy": unit["label"], "train_seed": unit.get("seed")})
            if isinstance(pol, Tracker) and pol.counts:
                for y, (u, t) in pol.counts.items():
                    unv.append({"policy": unit["label"], "train_seed": unit["seed"], "market_seed": ms,
                                "scenario": scen, "window": win, "year": y, "unvisited": u, "decisions": t})
    return pd.DataFrame(rows), pd.DataFrame(unv)


def units():
    U = [{"label": "Rule-based", "kind": "rule"}, {"label": "Random", "kind": "random"}]
    for model, lab in [("qlearning", "Q-learning"), ("dqn", "DQN")]:
        U += [{"label": lab, "kind": "trained", "model": model, "seed": s} for s in TRAIN_SEEDS]
    return U


# ---------------------------------------------------------------------- summaries
def paired(P):
    """Shocked minus paired no-shock run, per year, for each (policy, train seed, market seed)."""
    key = ["policy", "train_seed", "market_seed", "year"]
    P = P.copy()
    P["train_seed"] = P["train_seed"].astype(float).fillna(-1)
    P["bpl"] = P["buyers"] / P["listings"]
    base = P[P.scenario == "none"].set_index(key)
    sh = P[P.scenario != "none"].set_index(key)
    b = base.loc[sh.index]
    out = sh[["scenario", "window"]].copy()
    for c in ["cq_median", "median", "sales", "sell_through", "bpl", "ask_index"]:
        out[f"{c}_pct"] = (sh[c].to_numpy() / b[c].to_numpy() - 1) * 100
    out["cq_base"], out["cq_shock"] = b["cq_median"].to_numpy(), sh["cq_median"].to_numpy()
    out["median_base"], out["median_shock"] = b["median"].to_numpy(), sh["median"].to_numpy()
    return out.reset_index()


def summarize_runs(D):
    rows = []
    for (pol, scen, win), g in D.groupby(["policy", "scenario", "window"]):
        y0, y1 = WINDOWS[win]
        pre = g[g.year < y0]
        sy = g[g.year.between(y0, y1)]
        # per run (train seed x market seed)
        k = ["train_seed", "market_seed"]
        run_cq = sy.groupby(k).cq_median_pct.mean()
        run_med = sy.groupby(k).median_pct.mean()
        end = g[g.year == y1].set_index(k)
        before = g[g.year == y0 - 1].set_index(k)
        abs_cq = (end.cq_shock / before.cq_shock - 1) * 100       # shocked run, absolute change
        abs_cq_b = (end.cq_base / before.cq_base - 1) * 100       # no-shock run, same years
        abs_med = (end.median_shock / before.median_shock - 1) * 100
        # market-seed level (learners averaged over training seeds) for CIs
        ms_cq = run_cq.groupby("market_seed").mean()
        lo, hi = np.percentile([ms_cq.sample(len(ms_cq), replace=True, random_state=i).mean()
                                for i in range(2000)], [2.5, 97.5])
        e = EXPECTED[scen]
        rows.append({
            "policy": pol, "scenario": scen, "window": win,
            "pre_shock_max_abs_diff": float(np.nanmax(np.abs(pre[["cq_median_pct", "median_pct"]].to_numpy())))
            if len(pre) else 0.0,
            "cq_pct": run_cq.mean(), "cq_ci_lo": lo, "cq_ci_hi": hi,
            "cq_peak_pct": g.groupby("year").cq_median_pct.mean().loc[y0:].abs().max()
            * np.sign(g.groupby("year").cq_median_pct.mean().loc[y0:].sum()),
            "cq_2024_pct": g[g.year == 2024].cq_median_pct.mean(),
            "cq_2028_pct": g[g.year == 2028].cq_median_pct.mean(),
            "median_pct": run_med.mean(),
            "cq_abs_change": abs_cq.mean(), "cq_abs_change_noshock": abs_cq_b.mean(),
            "median_abs_change": abs_med.mean(),
            "share_expected_sign_cq": float((np.sign(run_cq) == e).mean()),
            "share_expected_sign_median": float((np.sign(run_med) == e).mean()),
            "sales_pct": sy.sales_pct.mean(), "sell_through_pct": sy.sell_through_pct.mean(),
            "bpl_pct": sy.bpl_pct.mean(), "n_runs": len(run_cq),
        })
    return pd.DataFrame(rows)


# Benchmark from a textbook supply-demand model. A flow shock scales buyers per listing by
# r = f_buyers / f_listings; with an inverse price elasticity k (percentage price change per
# percentage change in buyers per listing, in log terms) the price changes by r**k - 1.
# k is taken between 1/3 and 1 (elasticity between 1 and 3), giving the range reported below.
THEORY_K = (1 / 3, 1.0)
THEORY = {  # scenario -> (label, opposite scenario used for the symmetric prediction)
    "supply_up": ("Supply increase (listings +50%)", "supply"),
    "demand_down": ("Demand drop (buyers -50%)", "demand"),
    "supply": ("Supply drop (listings -50%)", "supply_up"),
    "demand": ("Demand increase (buyers +50%)", "demand_down"),
    "tight": ("Combined tightening (listings -50%, buyers +50%)", None),
    "down_flow": ("Combined downturn (listings +50%, buyers -50%)", None),
}


def theory_table(S, win="2022-23"):
    """Simulated constant-quality price response next to the supply-demand benchmark range."""
    rows = []
    for scen, (label, opp) in THEORY.items():
        spec = SCEN[scen]
        r = spec.get("buyers", 1.0) / spec.get("listings", 1.0)
        lo, hi = sorted((r ** k - 1) * 100 for k in THEORY_K)
        for pol in ["Rule-based", "Q-learning", "DQN", "Random"]:
            g = S[(S.scenario == scen) & (S.window == win) & (S.policy == pol)].iloc[0]
            sym = (-S[(S.scenario == opp) & (S.window == win) & (S.policy == pol)].cq_pct.iloc[0]
                   if opp else np.nan)
            rows.append({
                "shock": label, "model": pol,
                "theory_direction": "up" if r > 1 else "down",
                "simulated_pct": round(g.cq_pct, 1),
                "simulated_ci": f"{g.cq_ci_lo:.1f} to {g.cq_ci_hi:.1f}",
                "simulated_2024_pct": round(g.cq_2024_pct, 1),
                "predicted_if_model_symmetric_pct": round(sym, 1),
                "theory_range_pct": f"{lo:.1f} to {hi:.1f}",
                "direction_matches": bool(np.sign(g.cq_pct) == np.sign(r - 1)),
                "within_theory_range": bool(lo <= g.cq_pct <= hi),
                # simulated response as a share of the smallest effect the benchmark allows
                "share_of_min_theory_effect": round(g.cq_pct / (hi if r < 1 else lo), 2),
            })
    return pd.DataFrame(rows)


def latex_table(S):
    pols = ["Rule-based", "Q-learning", "DQN", "Random"]
    order = ["supply", "demand", "tight", "down_flow", "down_credit"]
    f = lambda v: f"{v:+.1f}".replace("-", "$-$")
    lines = [r"\begin{tabular}{p{5.6cm}rrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{Constant-quality price vs no-shock (\%)} & \multicolumn{3}{c}{Raw median vs no-shock (\%)} & \multicolumn{2}{c}{Rule-based CQ change (\%)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
             r"Scenario & Rule & Q & DQN & Random & Rule & Q & DQN & Shock & No shock \\", r"\midrule"]
    for win in ["2022-23", "2016-17"]:
        if win == "2016-17":
            lines += [r"\midrule", r"\multicolumn{10}{l}{\textit{Timing check: shocks in 2016--2017 (before COVID-19)}} \\"]
        for s in order:
            g = S[(S.scenario == s) & (S.window == win)].set_index("policy")
            cq = [f(g.loc[p, "cq_pct"]) for p in pols]
            md = [f(g.loc[p, "median_pct"]) for p in pols[:3]]
            ab = [f(g.loc["Rule-based", "cq_abs_change"]), f(g.loc["Rule-based", "cq_abs_change_noshock"])]
            lines.append(f"{LABELS[s]} & " + " & ".join(cq + md + ab) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(job, units(), chunksize=1)
    P = pd.concat([r[0] for r in res], ignore_index=True)
    UV = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True)
    P.to_csv(OUT / "shock_paths.csv", index=False)
    D = paired(P)
    S = summarize_runs(D)
    S.to_csv(OUT / "shock_summary.csv", index=False)
    # Q-learning: share of decisions in never-visited states, shock years vs same years without shock
    UV["window_years"] = UV.window.map(lambda w: WINDOWS.get(w, (2022, 2023)))
    q = UV[UV.policy == "Q-learning"]
    rows = []
    for (scen, win), g in q[q.scenario != "none"].groupby(["scenario", "window"]):
        y0, y1 = WINDOWS[win]
        gs = g[g.year.between(y0, y1)]
        gb = q[(q.scenario == "none") & q.year.between(y0, y1)]
        rows.append({"scenario": scen, "window": win,
                     "unvisited_share_shock": gs.unvisited.sum() / gs.decisions.sum(),
                     "unvisited_share_noshock": gb.unvisited.sum() / gb.decisions.sum()})
    pd.DataFrame(rows).to_csv(OUT / "qlearning_unvisited.csv", index=False)
    T = theory_table(S)
    T.to_csv(OUT / "shock_vs_theory.csv", index=False)
    print(T.to_string(index=False))
    tab = latex_table(S)
    (OUT / "shock_table.tex").write_text(tab)
    main_tab = S[S.scenario.isin(SCEN)].pivot_table(index=["window", "scenario"], columns="policy",
                                                     values=["cq_pct", "median_pct"]).round(2)
    main_tab.to_csv(OUT / "shock_table.csv")
    print(f"done in {time.time() - t:.0f} s; max pre-shock difference "
          f"{S.pre_shock_max_abs_diff.max():.2e}%")


if __name__ == "__main__":
    main()
