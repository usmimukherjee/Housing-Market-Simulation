"""Step 4. Frozen-policy evaluation, baselines, per-model recalibration and sanity checks.

Every policy is evaluated with epsilon = 0 and no learning, over 2010-2024, on the same 20 market
seeds (FINAL_EVAL_SEEDS). New listings and buyers come from separate random streams, so all models
face identical listing and buyer draws in every year.

Models: rule-based, random, untrained Q / DQN, trained Q / DQN (converged end-of-training policy,
10 seeds each; the best-validation checkpoint is reported as a sensitivity), fixed-action
references, and the ablation variants (end-of-training policies).

Also:
* per-model recalibration: the market grid search of step 1 repeated for each frozen policy with
  the same grid, data, objective and selection rule (equal calibration budget);
* deviation test: does a 10% minority playing a policy earn more than other minorities, in a
  population playing the learned, random or always-match policy? (individual learning evidence);
* sanity checks 1-10 from the protocol.

Writes results/results-convergence/evaluation/*.csv
"""

from __future__ import annotations

import argparse
import os
import dataclasses
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from housing_abm.calibration import GRID, calibrate
from housing_abm.config import (ALL_YEARS, CALIB_SEEDS, FINAL_EVAL_SEEDS, RESULTS, TEST_YEARS,
                                TRAIN_FIT_YEARS, TRAIN_SEEDS, TRAIN_YEARS, VAL_YEARS, TrainParams)
from housing_abm.data import load
from housing_abm.experiment import load_bundle, load_state
from housing_abm.market import ACTION_LABELS, ACTIONS, Market
from housing_abm.metrics import fit_metrics, validation_score
from housing_abm.policies import (DQNPolicy, FixedPolicy, MixedPolicy, QLearningPolicy, RandomPolicy,
                                  RulePolicy)
from housing_abm.training import (dqnparams_from, evaluate, qparams_from, random_rollout_features, train,
                                  train_targets)

OUT = RESULTS / "evaluation"
QUICK = bool(os.environ.get("EVAL_QUICK"))  # smoke test of the full evaluation path
if QUICK:
    FINAL_EVAL_SEEDS, TRAIN_SEEDS = FINAL_EVAL_SEEDS[:2], TRAIN_SEEDS[:2]
    OUT = RESULTS / "evaluation-quick"
PERIODS = {"train": TRAIN_FIT_YEARS, "val": VAL_YEARS, "test": TEST_YEARS,
           "all": list(range(2011, 2025))}
VARIANTS = {"main": {}, "reward-deal": {"reward": "deal"}, "reward-surplus": {"reward": "surplus"},
            "state-minimal": {"state": "minimal"}, "actions-A5": {"actions": "A5"}, "double-dqn": {}}


def best_hp():
    return json.loads((RESULTS / "tuning" / "best_hyperparameters.json").read_text())


def build_policy(model, variant, seed, trained=True, which="final"):
    kw = VARIANTS[variant]
    b = load_bundle(kw.get("reward", "realism"), kw.get("state", "rich"), kw.get("actions", "A3"))
    n = len(ACTIONS[b.spec.actions])
    hp = best_hp()
    if model == "qlearning":
        q = qparams_from(hp["qlearning"])
        p = QLearningPolicy(n, b.spec.state, b.market.max_rounds, q.alpha, q.gamma, seed, q.alpha_c)
    else:
        p = DQNPolicy(n, b.spec.state, dqnparams_from(hp["dqn"]), seed)
    if trained:
        p.load_state_dict(load_state(RESULTS / "training" / variant, f"{model}_s{seed}", which))
    elif model == "dqn":
        F, _ = random_rollout_features(load(), b, seed)
        p.set_normalization(F)
    return p, b


def run_metrics(paths, stats, d):
    rows = []
    for s, g in paths.groupby("market_seed"):
        r = {"market_seed": s}
        for per, yrs in PERIODS.items():
            m = fit_metrics(g, d.observed, d.observed_ratio, yrs)
            r.update({f"{per}_{k}": v for k, v in m.items()})
        r["val_score"] = validation_score(fit_metrics(g, d.observed, d.observed_ratio, VAL_YEARS))
        st = stats[stats["market_seed"] == s].iloc[0]
        for k in ["reward_per_negotiation", "deal_rate", "lost_rate", "fail_rate", "mean_rounds",
                  "mean_price_error"]:
            r[k] = st[k]
        for k in [c for c in stats.columns if c.startswith("action_share_")]:
            r[k] = st[k]
        r["sale_to_ask"] = g["sale_to_ask"].mean()
        r["buyers_purchase"] = g["buyers_purchase"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def eval_job(job):
    """job: model, variant, seed (train seed or None), kind (trained/untrained/rule/random/fixed),
    env ('shared' or 'recalibrated')."""
    d = load()
    t = time.time()
    kind = job["kind"]
    if kind == "rule":
        pol, b = RulePolicy(), load_bundle()
    elif kind == "random":
        pol, b = RandomPolicy(3, 77), load_bundle()
    elif kind == "fixed":
        pol, b = FixedPolicy(job["action"]), load_bundle()
    else:
        pol, b = build_policy(job["model"], job["variant"], job["seed"], kind == "trained",
                              job.get("which", "final"))
    recal = None
    if job.get("env") == "recalibrated":
        grid_kw = {"grid": {k: v[:2] for k, v in GRID.items()}} if QUICK else {}
        mp, grid = calibrate(pol, d, b, CALIB_SEEDS, **grid_kw)
        b = dataclasses.replace(b, market=mp)
        sel = grid[grid["selected"]].iloc[0].to_dict()
        recal = {"label": job["label"], "seed": job.get("seed"), **sel}
    paths, stats = evaluate(pol, d, b, FINAL_EVAL_SEEDS, ALL_YEARS, train_targets(d))
    met = run_metrics(paths, stats, d)
    for df in (paths, met):
        df["label"] = job["label"]
        df["model"] = job.get("model", kind)
        df["variant"] = job.get("variant", "main")
        df["kind"] = kind
        df["train_seed"] = job.get("seed")
        df["env"] = job.get("env", "shared")
    met["eval_time_s"] = time.time() - t
    return paths, met, recal


def eval_jobs(env="shared"):
    J = [{"label": "Rule-based", "kind": "rule", "env": env}]
    if env == "shared":
        J.append({"label": "Random", "kind": "random"})
        for a, name in enumerate(ACTION_LABELS["A3"]):
            J.append({"label": f"Always {name}", "kind": "fixed", "action": a})
    for model, lab in [("qlearning", "Q-learning"), ("dqn", "DQN")]:
        for s in TRAIN_SEEDS:
            J.append({"label": lab, "kind": "trained", "model": model, "variant": "main", "seed": s,
                      "env": env})
            if env == "shared":
                J.append({"label": f"Untrained {lab}", "kind": "untrained", "model": model,
                          "variant": "main", "seed": s})
                J.append({"label": f"{lab} (best val. checkpoint)", "kind": "trained", "model": model,
                          "variant": "main", "seed": s, "which": "best"})
    if env == "shared":
        tr = pd.read_csv(RESULTS / "training" / "training_runs.csv")
        for _, r in tr[tr["variant"] != "main"].iterrows():
            lab = "Q-learning" if r["model"] == "qlearning" else "DQN"
            J.append({"label": f"{lab} [{r['variant']}]", "kind": "trained", "model": r["model"],
                      "variant": r["variant"], "seed": int(r["seed"])})
    return J


# ------------------------------------------------------------------ deviation test
def deviation_job(job):
    d = load()
    model, seed = job["model"], job["seed"]
    learned, b = build_policy(model, "main", seed, True)
    t = train_targets(d)
    pols = {"Learned": learned, "Random": RandomPolicy(3, 1000 + seed), "Always +1%": FixedPolicy(0),
            "Always +6%": FixedPolicy(2)}
    rows = []
    for pn in ["Learned", "Random", "Always +6%"]:
        for dn in pols:
            for ms in FINAL_EVAL_SEEDS[:5]:
                m = Market(d, b.market, b.reward, b.spec, seed=ms, targets=t)
                m.run(TRAIN_YEARS, MixedPolicy(pols[pn], pols[dn], 0.1))
                g = m.group_stats
                rows.append({"model": model, "train_seed": seed, "population": pn, "deviator": dn,
                             "market_seed": ms, "deviator_reward": g[1][0] / g[1][1],
                             "deviator_win_rate": g[1][2] / g[1][1],
                             "population_reward": g[0][0] / g[0][1]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sanity checks
def sanity_checks(met: pd.DataFrame, paths: pd.DataFrame, dev: pd.DataFrame):
    d = load()
    rows = []

    def add(i, name, value, passed, note=""):
        rows.append({"check": i, "name": name, "value": value, "pass": bool(passed), "note": note})

    main = met[met["env"] == "shared"]
    byl = main.groupby("label")
    for lab in ["Q-learning", "DQN"]:
        tr = dev[(dev["model"] == ("qlearning" if lab == "Q-learning" else "dqn"))]
        a = tr[(tr.population == "Random") & (tr.deviator == "Learned")]["deviator_reward"].mean()
        r = tr[(tr.population == "Random") & (tr.deviator == "Random")]["deviator_reward"].mean()
        add(1, f"{lab}: trained deviator beats random deviator (random population)",
            f"{a:.3f} vs {r:.3f}", a > r)
        v_t = byl.get_group(lab)["val_score"].mean()
        v_r = byl.get_group("Random")["val_score"].mean()
        add(1, f"{lab}: validation score trained vs random policy", f"{v_t:.3f} vs {v_r:.3f}", v_t < v_r,
            "lower is better")
    # 2. greedy evaluation is deterministic and consistent
    pol, b = build_policy("qlearning", "main", 0)
    p1, _ = evaluate(pol, d, b, FINAL_EVAL_SEEDS[:2], ALL_YEARS)
    p2, _ = evaluate(pol, d, b, FINAL_EVAL_SEEDS[:2], ALL_YEARS)
    same = np.allclose(p1["median"], p2["median"], equal_nan=True)
    add(2, "Q-learning greedy evaluation repeats exactly", same, same)
    pol, b = build_policy("dqn", "main", 0)
    p1, _ = evaluate(pol, d, b, FINAL_EVAL_SEEDS[:2], ALL_YEARS)
    p2, _ = evaluate(pol, d, b, FINAL_EVAL_SEEDS[:2], ALL_YEARS)
    same = np.allclose(p1["median"], p2["median"], equal_nan=True)
    add(2, "DQN greedy evaluation repeats exactly", same, same)
    for lab in ["Q-learning", "DQN"]:
        cv = byl.get_group(lab).groupby("market_seed")["val_mape"].mean().std()
        add(2, f"{lab}: SD across market seeds of validation MAPE (pp)", f"{cv:.2f}", cv < 5)
    # 3-6 from training logs
    import glob
    for model, lab in [("qlearning", "Q-learning"), ("dqn", "DQN")]:
        eps = [pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS / "training" / "main" / f"{model}_s*_episodes.csv")))]
        cks = [pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS / "training" / "main" / f"{model}_s*_checkpoints.csv")))]
        first = np.mean([e["reward_per_negotiation"].head(20).mean() for e in eps])
        last = np.mean([e["reward_per_negotiation"].tail(20).mean() for e in eps])
        add(3, f"{lab}: training reward, first vs last 20 episodes (mean over seeds)",
            f"{first:.3f} -> {last:.3f}", True, "reported; population reward can fall (see deviation test)")
        v0 = np.mean([c["val_score"].iloc[0] for c in cks])
        vb = np.mean([c["val_score"].min() for c in cks])
        vl = np.mean([c["val_score"].tail(5).mean() for c in cks])
        add(4, f"{lab}: validation score untrained / best / last 5 checkpoints", f"{v0:.3f} / {vb:.3f} / {vl:.3f}",
            vb <= v0, "end-of-training (converged) policy is evaluated")
        qmax = max(e["q_max"].abs().max() for e in eps)
        add(5, f"{lab}: max |Q| during training", f"{qmax:.3f}", qmax < 10,
            "rewards are bounded in about [-1.5, 1.5] per negotiation")
        finite = all(np.isfinite(e["td_error"].dropna()).all() for e in eps)
        add(6, f"{lab}: TD error / loss finite in every episode", finite, finite)
    # 7. reproducibility of training with a fixed seed
    b = load_bundle()
    hp = best_hp()
    tp = TrainParams(episodes=20, eval_every=10)
    r1 = train("qlearning", d, b, qparams_from(hp["qlearning"]), tp, 0)[1]
    r2 = train("qlearning", d, b, qparams_from(hp["qlearning"]), tp, 0)[1]
    same = np.allclose(r1["val_score"], r2["val_score"])
    add(7, "Q-learning retrained with the same seed gives identical checkpoints", same, same)
    tp = TrainParams(episodes=4, eval_every=2)
    dq = dataclasses.replace(dqnparams_from(hp["dqn"]), warmup=500)
    r1 = train("dqn", d, b, dq, tp, 0)[1]
    r2 = train("dqn", d, b, dq, tp, 0)[1]
    same = np.allclose(r1["val_score"], r2["val_score"])
    add(7, "DQN retrained with the same seed gives identical checkpoints", same, same)
    # 8. no leakage between training, validation and test
    t = train_targets(d)
    add(8, "Reward targets restricted to training years", sorted(t), set(t) <= set(TRAIN_YEARS))
    add(8, "Test years absent from training, calibration and selection years",
        TEST_YEARS, not (set(TEST_YEARS) & (set(TRAIN_YEARS) | set(VAL_YEARS))))
    # 9. historical targets never enter the state: changing them leaves greedy behaviour unchanged
    for model in ["qlearning", "dqn"]:
        pol, b = build_policy(model, "main", 0)
        t2 = {y: v * 3 for y, v in t.items()}
        pa, _ = evaluate(pol, d, b, FINAL_EVAL_SEEDS[:2], ALL_YEARS, t)
        pb, _ = evaluate(pol, d, b, FINAL_EVAL_SEEDS[:2], ALL_YEARS, t2)
        same = np.allclose(pa["median"], pb["median"], equal_nan=True)
        add(9, f"{model}: tripling the reward targets leaves simulated prices unchanged", same, same)
    # 10. seed sensitivity
    for lab in ["Q-learning", "DQN"]:
        g = byl.get_group(lab).groupby("train_seed")["test_mape"].mean()
        add(10, f"{lab}: test MAPE across training seeds, mean (SD, min-max)",
            f"{g.mean():.1f} ({g.std():.1f}, {g.min():.1f}-{g.max():.1f})", g.std() < 5)
    return pd.DataFrame(rows)


def benchmarks():
    d = load()
    o = d.observed["median"]
    rows = []
    for fit_years, name in [(list(range(2010, 2019)), "Log-linear trend fitted 2010-2018"),
                            (list(range(2010, 2022)), "Log-linear trend fitted 2010-2021")]:
        c = np.polyfit(fit_years, np.log(o.loc[fit_years]), 1)
        pred = np.exp(np.polyval(c, ALL_YEARS))
        rows += [{"benchmark": name, "year": y, "median": p} for y, p in zip(ALL_YEARS, pred)]
    for last, name in [(2018, "No change from 2018"), (2021, "No change from 2021")]:
        rows += [{"benchmark": name, "year": y, "median": (o.loc[y] if y <= last else o.loc[last])}
                 for y in ALL_YEARS]
    df = pd.DataFrame(rows)
    met = []
    for name, g in df.groupby("benchmark"):
        g = g.assign(p25=g["median"], p75=g["median"], ratio=np.nan)
        r = {"benchmark": name}
        for per in ["val", "test"]:
            m = fit_metrics(g, d.observed.assign(p25=d.observed["median"], p75=d.observed["median"]),
                            d.observed_ratio, PERIODS[per])
            r.update({f"{per}_rmse": m["rmse"], f"{per}_mae": m["mae"], f"{per}_mape": m["mape"]})
        met.append(r)
    return df, pd.DataFrame(met)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-recalibration", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    J = eval_jobs("shared")
    if not a.skip_recalibration:
        J = eval_jobs("recalibrated") + J  # long jobs first
    P, M, R = [], [], []
    with Pool(a.workers) as p:
        for i, (paths, met, recal) in enumerate(p.imap_unordered(eval_job, J), 1):
            P.append(paths), M.append(met)
            if recal:
                R.append(recal)
            if i % 10 == 0 or i == len(J):
                print(f"eval {i}/{len(J)} {time.time() - t:.0f}s", flush=True)
        dev = pd.concat(p.map(deviation_job, [{"model": m, "seed": s} for m in ["qlearning", "dqn"]
                                               for s in TRAIN_SEEDS]), ignore_index=True)
    d = load()
    paths = pd.concat(P, ignore_index=True)
    paths["obs_median"] = paths["year"].map(d.observed["median"])
    paths["obs_p25"] = paths["year"].map(d.observed["p25"])
    paths["obs_p75"] = paths["year"].map(d.observed["p75"])
    paths["obs_ratio"] = paths["year"].map(d.observed_ratio)
    paths["obs_n_sales"] = paths["year"].map(d.observed["n_sales"])
    met = pd.concat(M, ignore_index=True)
    paths.to_csv(OUT / "yearly_sim_vs_obs_runs.csv", index=False)
    met.to_csv(OUT / "eval_runs.csv", index=False)
    dev.to_csv(OUT / "deviation_test.csv", index=False)
    if R:
        pd.DataFrame(R).to_csv(OUT / "recalibration_params.csv", index=False)
    bdf, bmet = benchmarks()
    bdf.to_csv(OUT / "benchmarks_yearly.csv", index=False)
    bmet.to_csv(OUT / "benchmarks.csv", index=False)
    sc = sanity_checks(met, paths, dev)
    sc.to_csv(OUT / "sanity_checks.csv", index=False)
    print(sc.to_string())
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
