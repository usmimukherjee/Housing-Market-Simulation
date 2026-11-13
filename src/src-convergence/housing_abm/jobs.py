"""Picklable job functions for multiprocessing (one training run per job)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import TrainParams
from .data import load
from .experiment import load_bundle, save_run
from .metrics import convergence
from .training import dqnparams_from, qparams_from, train


def train_job(job: dict) -> dict:
    """job keys: model, seed, hp (dict), tp (dict), reward, state, actions, outdir, tag."""
    t = time.time()
    d = load()
    b = load_bundle(job.get("reward", "realism"), job.get("state", "rich"), job.get("actions", "A3"))
    tp = TrainParams(**job.get("tp", {}))
    hp = qparams_from(job["hp"]) if job["model"] == "qlearning" else dqnparams_from(job["hp"])
    ep, ck, best_state, final_state, meta = train(job["model"], d, b, hp, tp, job["seed"])
    conv = convergence(ck, tp)
    meta.update(converged=conv["converged"], converged_episode=conv["episode"])
    if job.get("outdir"):
        save_run(Path(job["outdir"]), job["tag"], ep, ck, best_state, final_state, meta)
    tail = ep.tail(tp.ma_window)
    row = {k: v for k, v in job.items() if k not in ("hp", "tp", "outdir")}
    row.update({f"hp_{k}": (str(v) if isinstance(v, (list, tuple)) else v) for k, v in job["hp"].items()})
    row.update(
        best_episode=meta["best_episode"],
        best_val_score=meta["best_val_score"],
        final_val_score=float(ck["val_score"].iloc[-1]),
        untrained_val_score=float(ck["val_score"].iloc[0]),
        untrained_greedy_reward=float(ck["greedy_reward"].iloc[0]),
        final_greedy_reward=float(ck["greedy_reward"].tail(5).mean()),
        final_reward_ma=float(ep["reward_ma"].iloc[-1]),
        final_td_error=float(tail["td_error"].mean()),
        final_q_max=float(ep["q_max"].max()),
        final_policy_change=float(ck["policy_change"].tail(5).mean()),
        final_policy_change_onpolicy=float(ck["policy_change_onpolicy"].tail(5).mean()),
        final_val_score_5=float(ck["val_score"].tail(5).mean()),
        converged=conv["converged"],
        converged_episode=conv["episode"],
        stopped_early_at=meta["stopped_early_at"],
        episodes_run=int(ep["episode"].iloc[-1]),
        runtime_s=time.time() - t,
        nan_loss=bool(np.isnan(ep["td_error"].dropna()).any()) if "td_error" in ep else False,
    )
    return row
