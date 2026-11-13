"""Training and evaluation loops.

Definitions used throughout
---------------------------
* **MDP episode**: one buyer-seller negotiation (terminal on deal, impasse or timeout).
* **Training episode**: one complete simulated pass through the training years 2010-2018, with
  a fresh market realisation (new seed) each episode. A training episode contains thousands of
  MDP episodes. Training curves are indexed by training episode, never by simulation year.
* **Checkpoint**: every ``eval_every`` training episodes (and at episode 0, before any learning)
  the policy is frozen (epsilon = 0, no updates) and run over 2010-2021 on fixed market seeds.
  This yields the greedy reward on training years, the validation score on 2019-2021, and the
  greedy action on a fixed set of probe states (for the policy change rate).
* The parameters at the checkpoint with the best validation score are kept (early stopping).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import (CHECKPOINT_EVAL_SEEDS, TRAIN_YEARS, VAL_YEARS, Bundle, DQNParams, QParams,
                     TrainParams)
from .data import Data
from .market import ACTIONS, Market
from .metrics import fit_metrics, validation_score
from .policies import DQNPolicy, QLearningPolicy, RandomPolicy, RulePolicy


def train_targets(d: Data) -> dict:
    """Observed medians for training years only. Used in the reward, never in the state."""
    return {y: float(d.observed.loc[y, "median"]) for y in TRAIN_YEARS}


def make_policy(model: str, b: Bundle, hp, seed: int):
    n = len(ACTIONS[b.spec.actions])
    if model == "rule":
        return RulePolicy()
    if model == "random":
        return RandomPolicy(n, seed)
    if model == "qlearning":
        return QLearningPolicy(n, b.spec.state, b.market.max_rounds, hp.alpha, hp.gamma, seed,
                               getattr(hp, 'alpha_c', 0.0))
    if model == "dqn":
        return DQNPolicy(n, b.spec.state, hp, seed)
    raise ValueError(model)


class _Recorder:
    """Wraps a frozen policy and records the observations it acts on (on-policy states)."""

    def __init__(self, policy):
        self.p, self.kind, self.F = policy, policy.kind, []

    def act(self, f, eps):
        self.F.append(f.copy())
        return self.p.act(f, eps)


class _Collector:
    """Random policy that records the observations it sees."""
    kind = "discrete"

    def __init__(self, n, seed):
        self.inner = RandomPolicy(n, seed)
        self.F = []

    def act(self, f, eps):
        self.F.append(f.copy())
        return self.inner.act(f, eps)


def random_rollout_features(d: Data, b: Bundle, seed: int, n_probe: int = 2000):
    """Observations from a random-policy run on training years: used to fix the DQN input scaling
    and to draw the probe states for the policy change rate."""
    col = _Collector(len(ACTIONS[b.spec.actions]), seed)
    Market(d, b.market, b.reward, b.spec, seed=[seed, 999_999]).run(TRAIN_YEARS, col)
    F = np.concatenate(col.F)
    rng = np.random.default_rng(seed)
    probe = F[rng.choice(len(F), min(n_probe, len(F)), replace=False)]
    return F, probe


def evaluate(policy, d: Data, b: Bundle, seeds, years, targets=None):
    """Frozen policy (epsilon = 0, no learning). Returns yearly paths and negotiation stats."""
    paths, stats = [], []
    for s in seeds:
        m = Market(d, b.market, b.reward, b.spec, seed=s, targets=targets)
        df = m.run(years, policy, eps=0.0, learn=False)
        df["market_seed"] = s
        paths.append(df)
        st = m.stats.summary()
        st["market_seed"] = s
        stats.append(st)
    return pd.concat(paths, ignore_index=True), pd.DataFrame(stats)


def checkpoint_eval(policy, d, b, targets):
    years = TRAIN_YEARS + VAL_YEARS
    rec = _Recorder(policy)
    paths, stats = evaluate(rec, d, b, CHECKPOINT_EVAL_SEEDS, years, targets)
    F = np.concatenate(rec.F)
    policy._onpolicy_states = F[np.random.default_rng(0).choice(len(F), min(3000, len(F)), replace=False)]
    scores, mapes, rmses = [], [], []
    for s, g in paths.groupby("market_seed"):
        m = fit_metrics(g, d.observed, d.observed_ratio, VAL_YEARS)
        scores.append(validation_score(m))
        mapes.append(m["mape"])
        rmses.append(m["rmse"])
    return {
        "greedy_reward": stats["reward_per_negotiation"].mean(),
        "greedy_deal_rate": stats["deal_rate"].mean(),
        "greedy_rounds": stats["mean_rounds"].mean(),
        "greedy_price_error": stats["mean_price_error"].mean(),
        "val_score": float(np.mean(scores)),
        "val_mape": float(np.mean(mapes)),
        "val_rmse": float(np.mean(rmses)),
    }


def epsilon(ep, hp, episodes):
    frac = min(ep / max(hp.explore_fraction * episodes, 1), 1.0)
    return hp.eps_start + frac * (hp.eps_min - hp.eps_start)


def train(model: str, d: Data, b: Bundle, hp, tp: TrainParams, seed: int, verbose=False):
    targets = train_targets(d)
    policy = make_policy(model, b, hp, seed)
    F, probe = random_rollout_features(d, b, seed)
    if model == "dqn":
        policy.set_normalization(F)
    ep_rows, ck_rows = [], []
    prev_greedy = None
    prev_on = None
    best, best_state, best_ep, no_improve = np.inf, None, 0, 0
    t_start = time.time()

    def do_checkpoint(ep, eps):
        nonlocal prev_greedy, prev_on, best, best_state, best_ep, no_improve
        r = checkpoint_eval(policy, d, b, targets)
        g = policy.greedy(probe, tie_break=False)
        r["policy_change"] = float(np.mean(g != prev_greedy)) if prev_greedy is not None else np.nan
        prev_greedy = g
        # on-policy change: states visited at the previous checkpoint, actions then vs now
        if prev_on is not None:
            r["policy_change_onpolicy"] = float(np.mean(policy.greedy(prev_on[0], tie_break=False) != prev_on[1]))
        else:
            r["policy_change_onpolicy"] = np.nan
        S = policy._onpolicy_states
        prev_on = (S, policy.greedy(S, tie_break=False))
        for i in range(policy.n):
            r[f"probe_action_share_{i}"] = float(np.mean(g == i))
        r.update(episode=ep, epsilon=eps, elapsed_s=time.time() - t_start)
        if r["val_score"] < best - 1e-12:
            best, best_state, best_ep, no_improve = r["val_score"], policy.state_dict(), ep, 0
        else:
            no_improve += 1
        ck_rows.append(r)
        if verbose:
            print(f"  [{model} s{seed}] ep {ep:4d} eps {eps:.2f} greedy_r {r['greedy_reward']:.3f} "
                  f"val {r['val_score']:.4f} mape {r['val_mape']:.1f} pc {r['policy_change']}", flush=True)

    do_checkpoint(0, hp.eps_start)
    stopped = None
    for ep in range(1, tp.episodes + 1):
        eps = epsilon(ep - 1, hp, tp.episodes)
        policy.reset_metrics()
        if model == "dqn" and getattr(hp, "lr_decay", False):
            frac = (ep - 1) / max(tp.episodes - 1, 1)
            policy.set_lr_factor(1 - frac * (1 - hp.lr_final_frac))
        if model == "dqn" and getattr(hp, "lr_anneal", False):
            e0 = hp.explore_fraction * tp.episodes
            frac = max(0.0, (ep - 1 - e0) / max(tp.episodes - e0, 1))
            policy.set_lr_factor(hp.lr_anneal_frac ** frac)
        before = policy.snapshot()
        t0 = time.time()
        m = Market(d, b.market, b.reward, b.spec, seed=[seed, ep], targets=targets)
        m.run(TRAIN_YEARS, policy, eps=eps, learn=True)
        row = {"episode": ep, "epsilon": eps, "runtime_s": time.time() - t0}
        row.update(m.stats.summary())
        row.update(policy.metrics())
        if before is not None:
            denom = np.abs(before).sum()
            row["q_change_pct"] = float(np.abs(policy.Q - before).sum() / denom * 100) if denom > 0 else np.nan
        ep_rows.append(row)
        if ep % tp.eval_every == 0:
            do_checkpoint(ep, eps)
            explored = eps <= hp.eps_min + 1e-9
            if explored and no_improve >= tp.patience:
                stopped = ep
                break
    ep_df = pd.DataFrame(ep_rows)
    ep_df["reward_ma"] = ep_df["reward_per_negotiation"].rolling(tp.ma_window, min_periods=1).mean()
    ck_df = pd.DataFrame(ck_rows)
    final_state = policy.state_dict()
    meta = {"best_episode": best_ep, "best_val_score": best, "stopped_early_at": stopped,
            "train_time_s": time.time() - t_start, "n_params": _n_params(policy)}
    return ep_df, ck_df, best_state, final_state, meta


def _n_params(policy):
    if isinstance(policy, QLearningPolicy):
        return int(policy.Q.size)
    if isinstance(policy, DQNPolicy):
        return int(sum(p.numel() for p in policy.online.parameters()))
    return 0


def qparams_from(dct) -> QParams:
    return QParams(**{k: v for k, v in dct.items() if k in QParams.__dataclass_fields__})


def dqnparams_from(dct) -> DQNParams:
    kw = {k: v for k, v in dct.items() if k in DQNParams.__dataclass_fields__}
    if "hidden" in kw:
        kw["hidden"] = tuple(kw["hidden"])
    return DQNParams(**kw)
