"""Step 2b. DQN stability fixes, applied only because stage-1 tuning demonstrated a problem.

Stage 1 (2_tune_hyperparameters.py, five actions): none of 20 DQN configurations met the
convergence criterion; the greedy action changed in 16-48% of probe states between checkpoints and
max Q reached 8-11 although per-negotiation returns are bounded near +/-1.5.

Round 1 (A3 actions, linear learning-rate decay, gamma / batch / target / Double DQN variants):
policy change fell to 9-25%, max Q to 1.4-3.5, still no convergence. Diagnosis: Adam takes
steps of about the learning rate regardless of gradient size, so ~50,000 gradient steps between
checkpoints keep moving the weights and flip the argmax where actions are nearly tied.

Round 2 (learning-rate annealing: constant during exploration, then exponential decay to 0.1% of
the initial rate, the stochastic-approximation analogue of the decaying Q-learning step size;
optionally an exponential moving average (Polyak) of the weights for the evaluated policy):
policy change 1.7-4%, and 5 of 8 variants converge.

Selection rule (same as step 2): lowest validation score averaged over the last five checkpoints
among converged runs. Test years are not used.

Usage: python 2b_dqn_stability_fixes.py [--select-only]
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import pandas as pd

from housing_abm.config import RESULTS
from housing_abm.jobs import train_job

STAGE1_BEST = {"hidden": [128, 64], "lr": 5e-4, "gamma": 0.99, "batch_size": 32, "buffer_size": 10_000,
               "target_update": 500}
ROUND1 = {
    "A3 g.99 b32": {"lr_decay": True},
    "A3 g.9 b32": {"lr_decay": True, "gamma": 0.9},
    "A3 g.9 b128": {"lr_decay": True, "gamma": 0.9, "batch_size": 128},
    "A3 g.9 b128 tu1000": {"lr_decay": True, "gamma": 0.9, "batch_size": 128, "target_update": 1000},
    "A3 g.9 b128 double": {"lr_decay": True, "gamma": 0.9, "batch_size": 128, "double": True},
    "A3 g.9 b128 lr1e-4": {"lr_decay": True, "gamma": 0.9, "batch_size": 128, "lr": 1e-4},
    "A3 g.9 b64 (64,64)": {"lr_decay": True, "gamma": 0.9, "batch_size": 64, "hidden": [64, 64]},
    "A3 g.9 b128 double tu1000": {"lr_decay": True, "gamma": 0.9, "batch_size": 128, "double": True,
                                  "target_update": 1000},
}
ROUND2 = {
    "g.99 b32 anneal": {"lr_anneal": True},
    "g.99 b32 anneal ema": {"lr_anneal": True, "polyak_eval": 0.999},
    "g.9 b128 tu1000 anneal": {"lr_anneal": True, "gamma": 0.9, "batch_size": 128, "target_update": 1000},
    "g.9 b128 tu1000 anneal ema": {"lr_anneal": True, "gamma": 0.9, "batch_size": 128, "target_update": 1000,
                                   "polyak_eval": 0.999},
    "g.9 b128 lr1e-4 anneal": {"lr_anneal": True, "gamma": 0.9, "batch_size": 128, "lr": 1e-4},
    "g.9 b128 lr1e-4 anneal ema": {"lr_anneal": True, "gamma": 0.9, "batch_size": 128, "lr": 1e-4,
                                   "polyak_eval": 0.999},
    "g.99 b32 ema only": {"polyak_eval": 0.9995},
    "g.99 b32 double tu1000 anneal ema": {"lr_anneal": True, "double": True, "target_update": 1000,
                                          "polyak_eval": 0.999},
}
TP = {"episodes": 200, "eval_every": 10, "patience": 10_000}


def run_round(name, variants, workers):
    J = [{"model": "dqn", "seed": 100, "hp": {**STAGE1_BEST, **v}, "actions": "A3", "name": k, "tp": TP}
         for k, v in variants.items()]
    with Pool(workers) as p:
        df = pd.DataFrame(p.map(train_job, J))
    df.to_csv(RESULTS / "tuning" / f"dqn_stability_{name}.csv", index=False)
    return df


def select():
    frames = []
    for r, variants in [("round1", ROUND1), ("round2", ROUND2)]:
        df = pd.read_csv(RESULTS / "tuning" / f"dqn_stability_{r}.csv")
        df["round"] = r
        df["hp_full"] = df["name"].map(lambda k: json.dumps({**STAGE1_BEST, **variants[k]}))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    cols = ["round", "name", "final_val_score_5", "final_greedy_reward", "untrained_greedy_reward",
            "final_policy_change", "final_policy_change_onpolicy", "converged", "converged_episode",
            "final_q_max", "runtime_s", "hp_full"]
    df = df[cols].sort_values(["converged", "final_val_score_5"], ascending=[False, True])
    df.to_csv(RESULTS / "tuning" / "tuning_dqn_stability.csv", index=False)
    best = json.loads(df.iloc[0]["hp_full"])
    path = RESULTS / "tuning" / "best_hyperparameters.json"
    allbest = json.loads(path.read_text()) if path.exists() else {}
    allbest["dqn"] = best
    path.write_text(json.dumps(allbest, indent=2))
    print(df.drop(columns="hp_full").round(4).to_string(index=False))
    print("selected:", best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not a.select_only:
        run_round("round1", ROUND1, a.workers)
        run_round("round2", ROUND2, a.workers)
    select()


if __name__ == "__main__":
    main()
