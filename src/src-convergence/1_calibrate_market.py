"""Step 1. Calibrate the shared market environment with the rule-based policy.

Writes
  results/results-convergence/calibration/calibration_grid_rule.csv   every grid point
  results/results-convergence/calibration/market_params.json         selected parameters
"""

from __future__ import annotations

import dataclasses
import json
import time

from housing_abm.calibration import calibrate
from housing_abm.config import CALIB_SEEDS, RESULTS, Bundle
from housing_abm.data import load
from housing_abm.policies import RulePolicy


def main():
    out = RESULTS / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    mp, grid = calibrate(RulePolicy(), load(), Bundle(), CALIB_SEEDS)
    grid.to_csv(out / "calibration_grid_rule.csv", index=False)
    (out / "market_params.json").write_text(json.dumps(dataclasses.asdict(mp), indent=2))
    sel = grid[grid["selected"]].iloc[0]
    print(f"{len(grid)} grid points in {time.time() - t:.0f}s; "
          f"{int(grid['candidate'].sum())} within tolerance of best training error")
    print(sel.to_string())


if __name__ == "__main__":
    main()
