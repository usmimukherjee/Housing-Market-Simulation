"""Market (environment) calibration: grid search on training years, selection on validation years.

Objective on training years 2011-2018: log-RMSE of simulated vs observed P25, median and P75
(average path over market seeds). Selection: among candidates within ``tolerance`` of the best
training objective, the one with the lowest validation score (2019-2021). Test years are never used.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd

from .config import TRAIN_FIT_YEARS, VAL_YEARS, Bundle, MarketParams
from .data import Data
from .metrics import fit_metrics, validation_score
from .training import evaluate

GRID = {
    "wtp_scale": [1.4, 1.6, 1.8, 2.0, 2.2],
    "income_growth": [0.01, 0.02, 0.03, 0.04],
    "buyers_per_listing": [0.8, 1.0, 1.2, 1.5],
    "feedback_sellthrough": [0.0, 0.1, 0.25],
    "feedback_sale_to_ask": [0.5, 1.0],
}
TOLERANCE = 1.25


def mean_path(paths: pd.DataFrame) -> pd.DataFrame:
    cols = ["median", "p25", "p75", "mean", "ratio", "sale_to_ask", "sell_through",
            "buyers_purchase", "cq_median", "sales"]
    return paths.groupby("year")[cols].mean().reset_index()


def score_candidate(policy, d: Data, b: Bundle, seeds) -> dict:
    years = sorted(set(range(2010, 2019)) | set(VAL_YEARS))
    paths, _ = evaluate(policy, d, b, seeds, years)
    mp = mean_path(paths)
    tr = fit_metrics(mp, d.observed, d.observed_ratio, TRAIN_FIT_YEARS)
    va = fit_metrics(mp, d.observed, d.observed_ratio, VAL_YEARS)
    return {"train_obj": tr["log_rmse_quantiles"], "train_mape": tr["mape"],
            "val_score": validation_score(va), "val_mape": va["mape"]}


def calibrate(policy, d: Data, base: Bundle, seeds, grid=GRID) -> tuple[MarketParams, pd.DataFrame]:
    rows = []
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(keys, vals))
        b = dataclasses.replace(base, market=dataclasses.replace(base.market, **kw))
        r = score_candidate(policy, d, b, seeds)
        r.update(kw)
        rows.append(r)
    res = pd.DataFrame(rows)
    best_train = res["train_obj"].min()
    res["candidate"] = res["train_obj"] <= TOLERANCE * best_train
    chosen = res[res["candidate"]].sort_values("val_score").iloc[0]
    res["selected"] = False
    res.loc[chosen.name, "selected"] = True
    mp = dataclasses.replace(base.market, **{k: (float(chosen[k])) for k in keys})
    return mp, res
