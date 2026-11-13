"""Step 2. Hyperparameter tuning for Q-learning and DQN on the validation years.

Selection objective: validation score (log-RMSE of P25/median/P75 + ratio RMSE, 2019-2021) of the
policy at the END of training (mean of the last five checkpoints), averaged over the tuning seeds, among configurations whose runs meet
the convergence criterion in at least half of the seeds (all configurations if none do).
Why not the best checkpoint: taking the minimum over 30 noisy checkpoints favours configurations whose
greedy policy keeps changing (a winner's-curse effect). In the first pass every top-ranked Q-learning
configuration had a constant step size, a 55-76% policy change rate and 0% converged runs.
Both rankings are saved. Test years 2022-2024 are never computed here.

Q-learning: full grid alpha x gamma x eps_start x eps_min with a decaying step size (72 configs).
DQN: 20 configurations sampled (fixed seed) from the grid network x lr x gamma x batch x buffer x
target-update (486 combinations).

Writes results/results-convergence/tuning/tuning_{qlearning,dqn}.csv and best_hyperparameters.json.

Usage: python 2_tune_hyperparameters.py [--model qlearning|dqn|both] [--workers 8]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from housing_abm.config import RESULTS, TUNE_SEEDS
from housing_abm.jobs import train_job

Q_GRID = {
    "alpha": [0.05, 0.1, 0.2, 0.4],
    "gamma": [0.90, 0.95, 0.99],
    "eps_start": [0.5, 0.8, 1.0],
    "eps_min": [0.01, 0.05],
    # stage 1 (five actions) showed that constant step sizes (alpha_c = 0) never converge (0 of 144
    # runs); stage 2 (three actions, the final specification) searches decaying step sizes only
    "alpha_c": [200.0],
}
Q_TP = {"episodes": 300, "eval_every": 10, "patience": 10_000}

DQN_GRID = {
    "hidden": [(32, 32), (64, 64), (128, 64)],
    "lr": [1e-3, 5e-4, 1e-4],
    "gamma": [0.90, 0.95, 0.99],
    "batch_size": [32, 64, 128],
    "buffer_size": [10_000, 50_000],
    "target_update": [100, 500, 1000],
}
DQN_N_SAMPLES = 20
DQN_TP = {"episodes": 200, "eval_every": 10, "patience": 10_000}


def q_jobs():
    keys = list(Q_GRID)
    for i, vals in enumerate(itertools.product(*Q_GRID.values())):
        hp = dict(zip(keys, vals))
        for s in TUNE_SEEDS:
            yield {"model": "qlearning", "config_id": i, "seed": s, "hp": hp, "tp": Q_TP}


def dqn_jobs():
    keys = list(DQN_GRID)
    allc = list(itertools.product(*DQN_GRID.values()))
    rng = np.random.default_rng(2024)
    pick = rng.choice(len(allc), DQN_N_SAMPLES, replace=False)
    for i, j in enumerate(sorted(pick)):
        hp = dict(zip(keys, allc[j]))
        for s in TUNE_SEEDS:
            yield {"model": "dqn", "config_id": i, "seed": s, "hp": hp, "tp": DQN_TP}


def run(model, jobs, workers):
    out = RESULTS / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    jobs = list(jobs)
    rows = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(train_job, jobs), 1):
            rows.append(r)
            if i % 8 == 0 or i == len(jobs):
                print(f"[{model}] {i}/{len(jobs)} runs, {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["config_id", "seed"])
    df.to_csv(out / f"tuning_{model}_runs.csv", index=False)
    return aggregate(model)


def aggregate(model):
    out = RESULTS / "tuning"
    df = pd.read_csv(out / f"tuning_{model}_runs.csv")
    hp_cols = [c for c in df.columns if c.startswith("hp_")]
    vcol = "final_val_score_5" if "final_val_score_5" in df else "final_val_score"
    agg = df.groupby(["config_id"] + hp_cols, dropna=False).agg(
        final_val_score_mean=(vcol, "mean"), final_val_score_sd=(vcol, "std"),
        best_ckpt_val_score_mean=("best_val_score", "mean"),
        final_greedy_reward=("final_greedy_reward", "mean"),
        untrained_greedy_reward=("untrained_greedy_reward", "mean"),
        final_policy_change=("final_policy_change", "mean"),
        converged_share=("converged", "mean"), converged_episode=("converged_episode", "mean"),
        runtime_s=("runtime_s", "mean"),
    ).reset_index()
    agg["eligible"] = agg["converged_share"] >= 0.5
    if not agg["eligible"].any():
        agg["eligible"] = True
    agg = agg.sort_values(["eligible", "final_val_score_mean"], ascending=[False, True])
    agg["rank_best_checkpoint_rule"] = agg["best_ckpt_val_score_mean"].rank(method="first").astype(int)
    agg.to_csv(out / f"tuning_{model}.csv", index=False)
    best = agg.iloc[0]
    hp = {c[3:]: best[c] for c in hp_cols}
    if "hidden" in hp and isinstance(hp["hidden"], str):
        hp["hidden"] = list(eval(hp["hidden"]))
    for k, v in list(hp.items()):
        if isinstance(v, (np.integer,)):
            hp[k] = int(v)
        elif isinstance(v, (np.floating,)):
            hp[k] = float(v)
    for k in ("batch_size", "buffer_size", "target_update"):
        if k in hp:
            hp[k] = int(hp[k])
    print(agg.head(10).to_string())
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aggregate-only", action="store_true", help="re-rank saved runs, no training")
    a = ap.parse_args()
    if a.aggregate_only:
        path = RESULTS / "tuning" / "best_hyperparameters.json"
        best = {m: aggregate(m) for m in ("qlearning", "dqn")
                if (RESULTS / "tuning" / f"tuning_{m}_runs.csv").exists()}
        path.write_text(json.dumps(best, indent=2))
        print(json.dumps(best, indent=2))
        return
    path = RESULTS / "tuning" / "best_hyperparameters.json"
    best = json.loads(path.read_text()) if path.exists() else {}
    if a.model in ("qlearning", "both"):
        best["qlearning"] = run("qlearning", q_jobs(), a.workers)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(best, indent=2))
    if a.model in ("dqn", "both"):
        best["dqn"] = run("dqn", dqn_jobs(), a.workers)  # stage 1; then run 2b_dqn_stability_fixes.py
        path.write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
