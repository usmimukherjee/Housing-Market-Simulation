"""Exogenous market shocks on top of the shared market (``market.Market``), which is not modified.

A shock is a dict ``{year: {"listings": f_L, "buyers": f_B, "wtp": f_W}}``:

* ``listings``: new listings entering in that year are scaled by ``f_L`` (supply).
* ``buyers``:   new buyers entering in that year are scaled by ``f_B`` (demand).
* ``wtp``:      the purchasing power (WTP, and therefore the bid cap) of every active buyer is
                scaled by ``f_W`` in that year only, e.g. a credit or interest-rate shock.

Pairing. The baseline draws of new listings and buyers are made first, from the same random
streams as the no-shock run. Contractions drop a random subset of those draws; expansions add
extra draws from a separate stream seeded by (market seed, year). A shocked run is therefore
identical to its paired no-shock run in every year before the shock begins, and the difference
afterwards is caused by the shock alone.
"""

from __future__ import annotations

import numpy as np

from .market import Market


class ShockMarket(Market):
    def __init__(self, *args, seed: int, shocks: dict | None = None, **kw):
        super().__init__(*args, seed=seed, **kw)
        self.market_seed = seed
        self.shocks = shocks or {}

    def _f(self, year, key):
        return float(self.shocks.get(year, {}).get(key, 1.0))

    def _rng(self, year, stream):
        return np.random.default_rng([self.market_seed, year, stream, 7919])

    # ------------------------------------------------------------------ supply
    def _new_listings(self, year):
        f = self._f(year, "listings")
        n0, L0 = self.mp.n_listings, len(self.L_ask)
        super()._new_listings(year)
        if f < 1.0:
            drop = self._rng(year, 1).choice(n0, int(round((1 - f) * n0)), replace=False)
            keep = np.ones(len(self.L_ask), bool)
            keep[L0 + drop] = False
            self.L_q, self.L_av = self.L_q[keep], self.L_av[keep]
            self.L_ask, self.L_res = self.L_ask[keep], self.L_res[keep]
        elif f > 1.0:
            saved = self.rng_list
            self.rng_list = self._rng(year, 2)
            self.mp.n_listings = int(round((f - 1) * n0))
            try:
                super()._new_listings(year)
            finally:
                self.mp.n_listings, self.rng_list = n0, saved

    # ------------------------------------------------------------------ demand
    def _new_buyers(self, year):
        f = self._f(year, "buyers")
        B0 = len(self.B_wtp)
        super()._new_buyers(year)
        n0 = len(self.B_wtp) - B0
        if f < 1.0:
            drop = self._rng(year, 3).choice(n0, int(round((1 - f) * n0)), replace=False)
            keep = np.ones(len(self.B_wtp), bool)
            keep[B0 + drop] = False
            self.B_wtp = self.B_wtp[keep]
        elif f > 1.0:
            saved, bpl = self.rng_buy, self.mp.buyers_per_listing
            self.rng_buy = self._rng(year, 4)
            self.mp.buyers_per_listing = bpl * (f - 1)
            try:
                super()._new_buyers(year)
            finally:
                self.mp.buyers_per_listing, self.rng_buy = bpl, saved
        # purchasing-power shock applies to every active buyer this year
        self.B_wtp = self.B_wtp * self._f(year, "wtp")

    def run_year(self, year, policy, eps=0.0, learn=False):
        row = super().run_year(year, policy, eps, learn)
        self.B_wtp = self.B_wtp / self._f(year, "wtp")  # carried-over buyers regain their WTP
        return row
