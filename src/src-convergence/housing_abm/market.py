"""Shared housing-market environment. Identical for the rule-based, Q-learning and DQN models.

One simulated year
------------------
1. ``n_listings`` new listings are drawn from the dwelling pool. A listing's ask is
   ``quality x ask_index x lognormal noise``. Its seller has a private reservation price.
2. ``buyers_per_listing x n_listings`` new buyers are drawn from the survey movers. A buyer's
   WTP is ``wtp_scale x survey WTP x (1 + income_growth)^(year - 2010) x lognormal noise``.
   Bids can never exceed ``cap x WTP``.
3. ``cycles`` sub-market cycles. In each cycle every active buyer picks one unsold listing at
   random among those with ask in ``[search_floor x WTP, cap x WTP]``. Every buyer-listing pair
   negotiates for up to ``max_rounds`` rounds (the MDP episode, see below). Among pairs that
   reach a deal on the same listing, the highest price wins. Winners and sold listings leave.
4. The ask index for next year moves with the realised median sale-to-ask ratio and the
   sell-through rate. Unsold listings and buyers carry over with fixed probabilities.

Negotiation (the MDP episode for the learning agents)
-----------------------------------------------------
* Buyer opens at ``min(open_wtp x WTP, open_ask x ask)``. Seller counter starts at
  ``ask x (1 + escalation x rivals)``.
* Each round the buyer acts (see ``ACTIONS``). If bid >= counter the deal closes at the bid.
  Otherwise the seller concedes ``concession / (1 + rivals)`` of the gap, never below the
  reservation price.
* Ends with a deal, an impasse (buyer at cap, seller at reservation, bid still below), or a
  timeout after ``max_rounds``.
* Terminal reward is assigned after competition is resolved (win, lost to a rival, or fail).

Observation (all available to the buyer at decision time; no historical target is included)
-------------------------------------------------------------------------------------------
0 bid / ask, 1 counter / ask, 2 (counter - bid) / ask, 3 (cap - bid) / ask, 4 round / max_rounds,
5 WTP / ask, 6 previous increment / ask, 7 number of rival bidders,
8 last year's log median sale-to-ask ratio, 9 last year's sell-through rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FIRST_YEAR, RATIO_YEARS, MarketParams, RewardParams, Spec
from .data import Data

N_FEATURES = 10
ACTIONS = {
    # name: list of (kind, value). kind "inc" adds value x ask to the bid, "hold" keeps it,
    # "match" sets the bid to the seller's counter (capped).
    "A5": [("hold", 0.0), ("inc", 0.01), ("inc", 0.03), ("inc", 0.06), ("match", 0.0)],
    "A3": [("inc", 0.01), ("inc", 0.03), ("inc", 0.06)],
}
ACTION_LABELS = {
    "A5": ["hold", "+1%", "+3%", "+6%", "match"],
    "A3": ["+1%", "+3%", "+6%"],
}


@dataclass
class EpisodeStats:
    n_neg: int = 0
    n_win: int = 0
    n_lost: int = 0
    n_fail: int = 0
    rounds: int = 0
    reward: float = 0.0          # summed over negotiations in years that have reward targets
    n_rewarded: int = 0
    price_err: float = 0.0       # |P - P*| / P* summed over winning deals in target years
    n_price_err: int = 0
    actions: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.int64))

    def summary(self) -> dict:
        n = max(self.n_neg, 1)
        out = {
            "negotiations": self.n_neg,
            "reward_total": self.reward,
            "reward_per_negotiation": self.reward / max(self.n_rewarded, 1),
            "deal_rate": self.n_win / n,
            "lost_rate": self.n_lost / n,
            "fail_rate": self.n_fail / n,
            "mean_rounds": self.rounds / n,
            "mean_price_error": self.price_err / max(self.n_price_err, 1),
        }
        tot = max(self.actions.sum(), 1)
        for i, c in enumerate(self.actions):
            out[f"action_share_{i}"] = c / tot
        return out


class Market:
    def __init__(self, data: Data, mp: MarketParams, rp: RewardParams, spec: Spec, seed: int,
                 targets: dict | None = None):
        self.d, self.mp, self.rp, self.spec = data, mp, rp, spec
        self.actions = ACTIONS[spec.actions]
        self.targets = targets or {}
        ss = np.random.SeedSequence(seed)
        s_list, s_buy, s_mkt, s_rule = ss.spawn(4)
        # separate streams so new listings and buyers are identical across policies
        self.rng_list = np.random.default_rng(s_list)
        self.rng_buy = np.random.default_rng(s_buy)
        self.rng = np.random.default_rng(s_mkt)
        self.rng_rule = np.random.default_rng(s_rule)
        self.index = float(data.observed.loc[FIRST_YEAR, "median"])
        self.lag_sta = 0.0   # neutral before the first simulated year
        self.lag_st = mp.target_sellthrough
        # listing state
        self.L_q = np.empty(0)
        self.L_av = np.empty((0, len(RATIO_YEARS)))
        self.L_ask = np.empty(0)
        self.L_res = np.empty(0)
        # buyer state
        self.B_wtp = np.empty(0)
        self.group_stats = {}
        self.stats = EpisodeStats(actions=np.zeros(len(self.actions), dtype=np.int64))

    # ------------------------------------------------------------------ features
    def features(self, b, c, ask, C, W, k, prev_inc, n_riv):
        f = np.empty((len(b), N_FEATURES))
        f[:, 0] = b / ask
        f[:, 1] = c / ask
        f[:, 2] = (c - b) / ask
        f[:, 3] = (C - b) / ask
        f[:, 4] = k / self.mp.max_rounds
        f[:, 5] = W / ask
        f[:, 6] = prev_inc
        f[:, 7] = n_riv
        f[:, 8] = self.lag_sta
        f[:, 9] = self.lag_st
        return f

    # ------------------------------------------------------------------ one year
    def _new_listings(self, year):
        mp = self.mp
        idx = self.rng_list.integers(0, len(self.d.quality), mp.n_listings)
        z = self.rng_list.standard_normal(mp.n_listings)
        r = self.rng_list.uniform(mp.reservation_lo, mp.reservation_hi, mp.n_listings)
        q = self.d.quality[idx]
        ask = q * self.index * np.exp(mp.ask_noise * z)
        self.L_q = np.concatenate([self.L_q, q])
        self.L_av = np.concatenate([self.L_av, self.d.av[idx]])
        self.L_ask = np.concatenate([self.L_ask, ask])
        self.L_res = np.concatenate([self.L_res, r])  # stored as a share of ask

    def _new_buyers(self, year):
        mp = self.mp
        n = int(round(mp.buyers_per_listing * mp.n_listings))
        idx = self.rng_buy.integers(0, len(self.d.wtp), n)
        z = self.rng_buy.standard_normal(n)
        w = (mp.wtp_scale * self.d.wtp[idx] * (1 + mp.income_growth) ** (year - FIRST_YEAR)
             * np.exp(mp.wtp_noise * z))
        self.B_wtp = np.concatenate([self.B_wtp, w])

    def run_year(self, year, policy, eps=0.0, learn=False):
        mp = self.mp
        self._new_listings(year)
        self._new_buyers(year)
        nL, nB = len(self.L_ask), len(self.B_wtp)
        sold = np.zeros(nL, bool)
        bought = np.zeros(nB, bool)
        price = np.full(nL, np.nan)
        rounds_win = np.full(nL, np.nan)
        for _ in range(mp.cycles):
            pb, pl = self._match(sold, bought)
            if len(pb) == 0:
                break
            win_b, win_l, win_p, win_k = self._negotiate(year, pb, pl, policy, eps, learn)
            sold[win_l] = True
            bought[win_b] = True
            price[win_l] = win_p
            rounds_win[win_l] = win_k
        # ---- year statistics
        s = sold
        p = price[s]
        sta = p / self.L_ask[s]
        row = {
            "year": year, "listings": nL, "buyers": nB, "sales": int(s.sum()),
            "median": np.median(p) if len(p) else np.nan,
            "p25": np.quantile(p, 0.25) if len(p) else np.nan,
            "p75": np.quantile(p, 0.75) if len(p) else np.nan,
            "mean": p.mean() if len(p) else np.nan,
            "sale_to_ask": np.median(sta) if len(p) else np.nan,
            "sell_through": s.mean(),
            "buyers_purchase": bought.mean(),
            "mean_rounds_winners": np.nanmean(rounds_win) if len(p) else np.nan,
            "ask_index": self.index,
            "cq_median": np.median(p / self.L_q[s]) if len(p) else np.nan,
        }
        if year in RATIO_YEARS:
            j = RATIO_YEARS.index(year)
            row["ratio"] = np.median(self.L_av[s, j] / p) if len(p) else np.nan
        else:
            row["ratio"] = np.nan
        # ---- seller expectations for next year
        if len(p):
            self.lag_sta = float(np.log(np.median(sta)))
        self.lag_st = float(s.mean())
        self.index *= np.exp(mp.feedback_sale_to_ask * self.lag_sta
                             + mp.feedback_sellthrough * (self.lag_st - mp.target_sellthrough))
        # ---- carry-over
        keepL = (~sold) & (self.rng.random(nL) < mp.listing_carry)
        self.L_q, self.L_av = self.L_q[keepL], self.L_av[keepL]
        self.L_ask = self.L_ask[keepL] * (1 - mp.listing_cut)
        self.L_res = self.L_res[keepL]
        keepB = (~bought) & (self.rng.random(nB) < mp.buyer_carry)
        self.B_wtp = self.B_wtp[keepB] * (1 + mp.income_growth)
        return row

    def _match(self, sold, bought):
        mp = self.mp
        bi = np.flatnonzero(~bought)
        li = np.flatnonzero(~sold)
        if len(bi) == 0 or len(li) == 0:
            return np.empty(0, int), np.empty(0, int)
        W = self.B_wtp[bi][:, None]
        A = self.L_ask[li][None, :]
        elig = (A >= mp.search_floor * W) & (A <= mp.cap * W)
        score = np.where(elig, self.rng.random(elig.shape), -1.0)
        choice = score.argmax(1)
        ok = elig[np.arange(len(bi)), choice]
        return bi[ok], li[choice[ok]]

    # ------------------------------------------------------------------ negotiation
    def _negotiate(self, year, pb, pl, policy, eps, learn):
        mp, rp = self.mp, self.rp
        P = len(pb)
        ask = self.L_ask[pl]
        R = self.L_res[pl] * ask
        n_riv = (np.bincount(pl, minlength=len(self.L_ask)) - 1)[pl].astype(float)
        W = self.B_wtp[pb]
        C = mp.cap * W
        c = ask * (1 + mp.escalation * n_riv)
        b = np.minimum(mp.open_wtp * W, mp.open_ask * ask)
        prev_inc = np.zeros(P)
        done = np.zeros(P, bool)
        deal = np.zeros(P, bool)
        last_k = np.full(P, -1)
        is_rule = policy.kind == "rule"
        rule_u = self.rng_rule.uniform(0.01, 0.05, P) if is_rule else None
        record = learn and not is_rule
        S, A, S2, IDX, K = [], [], [], [], []
        group = None
        for k in range(mp.max_rounds):
            act = np.flatnonzero(~done)
            if len(act) == 0:
                break
            ba, ca, aa, Ca = b[act], c[act], ask[act], C[act]
            f = self.features(ba, ca, aa, Ca, W[act], k, prev_inc[act], n_riv[act])
            if k == 0 and hasattr(policy, "group"):
                group = policy.group(f)
            if is_rule:
                nb = ba * (1 + rule_u[act])
            else:
                a = policy.act(f, eps)
                nb = ba.copy()
                for j, (kind, v) in enumerate(self.actions):
                    m = a == j
                    if not m.any():
                        continue
                    if kind == "inc":
                        nb[m] = ba[m] + v * aa[m]
                    elif kind == "match":
                        nb[m] = ca[m]
                self.stats.actions += np.bincount(a, minlength=len(self.actions))
            nb = np.minimum(nb, Ca)
            prev_inc[act] = (nb - ba) / aa
            b[act] = nb
            closed = nb >= ca
            deal[act[closed]] = True
            open_ = act[~closed]
            c[open_] = np.maximum(R[open_], c[open_] - mp.concession / (1 + n_riv[open_])
                                  * (c[open_] - b[open_]))
            impasse = (b[open_] >= C[open_] - 1e-6) & (c[open_] <= R[open_] + 1e-6)
            ended = np.zeros(len(act), bool)
            ended[closed] = True
            ended[~closed] = impasse | (k == mp.max_rounds - 1)
            done[act[ended]] = True
            last_k[act[ended]] = k
            if record:
                f2 = self.features(b[act], c[act], aa, Ca, W[act], k + 1, prev_inc[act], n_riv[act])
                S.append(f), A.append(a), S2.append(f2), IDX.append(act), K.append(np.full(len(act), k))
        # ---- competition: highest deal price on each listing wins
        dp = np.where(deal, b, -np.inf)
        order = np.lexsort((-dp + 1e-9 * self.rng.random(P), pl))
        first = np.ones(P, bool)
        first[1:] = pl[order][1:] != pl[order][:-1]
        win = np.zeros(P, bool)
        cand = order[first]
        win[cand[deal[cand]]] = True
        lost = deal & ~win
        fail = ~deal
        # ---- terminal reward
        term = np.where(win, rp.deal, np.where(lost, -rp.lost, -rp.fail))
        target = self.targets.get(year)
        q = self.L_q[pl]
        if rp.kind == "realism":
            term = term - np.where(win, rp.overpay * np.maximum(0.0, (b - W) / W), 0.0)
            if target is not None:
                err = np.minimum(np.abs(b - q * target) / (q * target), 1.0)
                term = term - np.where(win, rp.price_error * err, 0.0)
        elif rp.kind == "surplus":
            term = term + np.where(win, rp.surplus_weight * (C - b) / C, 0.0)
        n_steps = last_k + 1
        st = self.stats
        st.n_neg += P
        st.n_win += int(win.sum())
        st.n_lost += int(lost.sum())
        st.n_fail += int(fail.sum())
        st.rounds += int(n_steps.sum())
        if target is not None:
            st.reward += float((term - rp.step * n_steps).sum())
            st.n_rewarded += P
            err = np.abs(b[win] - q[win] * target) / (q[win] * target)
            st.price_err += float(err.sum())
            st.n_price_err += int(win.sum())
        if group is not None:
            tot = term - rp.step * n_steps
            for gi in np.unique(group):
                m = group == gi
                acc = self.group_stats.setdefault(int(gi), [0.0, 0, 0])
                acc[0] += float(tot[m].sum()); acc[1] += int(m.sum()); acc[2] += int(win[m].sum())
        if record and S:
            idx = np.concatenate(IDX)
            ks = np.concatenate(K)
            is_last = ks == last_k[idx]
            r = -rp.step + np.where(is_last, term[idx], 0.0)
            policy.learn(np.concatenate(S), np.concatenate(A), r, np.concatenate(S2),
                         is_last.astype(float), ks)
        return pb[win], pl[win], b[win], n_steps[win]

    # ------------------------------------------------------------------ full path
    def run(self, years, policy, eps=0.0, learn=False) -> pd.DataFrame:
        rows = [self.run_year(y, policy, eps, learn) for y in years]
        return pd.DataFrame(rows)
