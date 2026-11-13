"""Load the derived inputs written by ``src/data-preparation/prepare_observed_data.py``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import DERIVED, RATIO_YEARS


@dataclass
class Data:
    quality: np.ndarray        # relative dwelling quality (2019 assessment / pool median)
    av: np.ndarray             # assessed values, columns = RATIO_YEARS
    wtp: np.ndarray            # survey WTP of movers, dollars
    observed: pd.DataFrame     # annual observed statistics, indexed by year
    observed_ratio: pd.Series  # median assessed value / sale price, indexed by year


@lru_cache(maxsize=1)
def load() -> Data:
    pool = pd.read_csv(DERIVED / "listing_pool.csv")
    buyers = pd.read_csv(DERIVED / "buyers.csv")
    obs = pd.read_csv(DERIVED / "observed_annual.csv").set_index("year")
    rat = pd.read_csv(DERIVED / "observed_ratio.csv").set_index("year")["median_ratio"]
    return Data(
        quality=pool["quality"].to_numpy(float),
        av=pool[[f"av_{y}" for y in RATIO_YEARS]].to_numpy(float),
        wtp=buyers["wtp"].to_numpy(float),
        observed=obs,
        observed_ratio=rat,
    )
