"""Step 3. Final training runs with the tuned hyperparameters, plus ablations.

Main runs: 10 seeds each for Q-learning and DQN, realism reward, rich state, three actions (+1/3/6%).
Ablations (all selected/compared on validation years only):
  reward:   "deal" (close quickly, the submitted design) and "surplus" (buyer surplus)
  state:    "minimal" (bid/ask and counter/ask only, closest to the submitted state)   [Q-learning]
  actions:  "A5" (adds hold and match-counter to the three increments)                  [Q-learning]
  double:   Double DQN, to check for overestimation                                     [DQN]

Writes results/results-convergence/training/<variant>/<model>_s<seed>_{episodes,checkpoints}.csv,
model parameters (.pkl), and training_runs.csv (one row per run).

Usage: python 3_train_final.py [--workers 8] [--only main]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import pandas as pd

from housing_abm.config import RESULTS, TRAIN_SEEDS
from housing_abm.jobs import train_job

Q_EPISODES = 400
DQN_EPISODES = 250
ABLATION_SEEDS_Q = TRAIN_SEEDS[:5]
ABLATION_SEEDS_DQN = TRAIN_SEEDS[:2]


def jobs(best, only=None, models=None):
    out = RESULTS / "training"
    tq = {"episodes": Q_EPISODES, "eval_every": 10, "patience": 10_000}
    td = {"episodes": DQN_EPISODES, "eval_every": 10, "patience": 10_000}
    J = []

    def add(model, variant, seeds, tp, hp, **kw):
        if (only and variant not in only) or (models and model not in models):
            return
        for s in seeds:
            J.append({"model": model, "variant": variant, "seed": s, "hp": hp, "tp": tp,
                      "outdir": str(out / variant), "tag": f"{model}_s{s}", **kw})

    # DQN first: longest jobs
    add("dqn", "main", TRAIN_SEEDS, td, best["dqn"])
    add("dqn", "reward-deal", ABLATION_SEEDS_DQN, td, best["dqn"], reward="deal")
    add("dqn", "reward-surplus", ABLATION_SEEDS_DQN, td, best["dqn"], reward="surplus")
    add("dqn", "double-dqn", ABLATION_SEEDS_DQN, td, {**best["dqn"], "double": True})
    add("qlearning", "main", TRAIN_SEEDS, tq, best["qlearning"])
    add("qlearning", "reward-deal", ABLATION_SEEDS_Q, tq, best["qlearning"], reward="deal")
    add("qlearning", "reward-surplus", ABLATION_SEEDS_Q, tq, best["qlearning"], reward="surplus")
    add("qlearning", "state-minimal", ABLATION_SEEDS_Q, tq, best["qlearning"], state="minimal")
    add("qlearning", "actions-A5", ABLATION_SEEDS_Q, tq, best["qlearning"], actions="A5")
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    best = json.loads((RESULTS / "tuning" / "best_hyperparameters.json").read_text())
    J = jobs(best, a.only, a.models)
    t = time.time()
    rows = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(train_job, J), 1):
            rows.append(r)
            print(f"{i}/{len(J)} {r['model']} {r['variant']} s{r['seed']} best_val {r['best_val_score']:.4f} "
                  f"converged {r['converged']} ({r['converged_episode']}) {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["model", "variant", "seed"])
    path = RESULTS / "training" / "training_runs.csv"
    if (a.only or a.models) and path.exists():
        old = pd.read_csv(path)
        keys = set(zip(df["model"], df["variant"]))
        old = old[[k not in keys for k in zip(old["model"], old["variant"])]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    print(df.groupby(["model", "variant"])[["best_val_score", "final_greedy_reward", "untrained_greedy_reward",
                                            "converged"]].mean().to_string())


if __name__ == "__main__":
    main()
