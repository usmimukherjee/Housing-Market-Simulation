"""How simulation and training costs change with the agent population (RQ3 scalability check).

Population scaling. The calibrated market draws ``n_listings`` = 300 new listings a year and
``buyers_per_listing`` x 300 new buyers. Scale factor k multiplies ``n_listings`` (buyers follow,
since they are drawn per listing): k in {0.5, 1, 2, 4, 8}, i.e. 150 to 2,400 new listings a year.
Nothing else in the market, the policies or the training code is changed.

Protocol (same as compute_benchmark.py)
* CPU only, one fresh Python process per run, runs executed one at a time, every BLAS/OMP thread
  count and torch.set_num_threads fixed to 1.
* Simulation: frozen end-of-training policies (Q-learning and DQN training seed 0; rule-based),
  epsilon = 0, no learning, 2010-2024, market seeds 1000-1002. Warm-up year on an unused seed;
  timed run repeated 3 times, median reported. Peak RAM = max RSS of the process.
* Training cost: an untrained policy (as at the start of 3_train_final.py) runs 2 training
  episodes (2010-2018, learning on, epsilon = 0.5, i.e. mid-exploration); mean wall time per
  episode reported. The number of episodes needed to converge at larger populations is NOT
  measured (that would need full retraining), so total training time is not extrapolated here.

Usage (from the package root):
    python src/src-model-complexity/scaling_benchmark.py            # all runs, then summary + figure
    python src/src-model-complexity/scaling_benchmark.py --summarise-only
Writes results/results-complexity/scaling/ and graphs/complexity/Scaling-*.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
OUT = PKG / "results" / "results-complexity" / "scaling"
GRAPHS = PKG / "graphs" / "complexity"
THREAD_ENV = {k: "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]}
MODELS = ["Rule-based", "Q-learning", "DQN"]
COLORS = {"Rule-based": "#2a78d6", "Q-learning": "#eb6834", "DQN": "#1baf7a"}
MARKERS = {"Rule-based": "s", "Q-learning": "^", "DQN": "D"}
LINES = {"Rule-based": (0, (5, 2)), "Q-learning": (0, (5, 1.5, 1, 1.5)), "DQN": "-"}
SCALES = [0.5, 1, 2, 4, 8]
SIM_SEEDS = [1000, 1001, 1002]
REPEATS = 3
TRAIN_EPISODES = 2
WARMUP_SEED = 999
MB = 2 ** 20


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / MB if sys.platform == "darwin" else r / 1024


# ============================================================================ worker
def worker(spec: dict) -> dict:
    out = {**spec}
    import copy
    import importlib.util

    import numpy as np
    import psutil
    sys.path.insert(0, str(PKG / "src" / "src-convergence"))
    from housing_abm.config import ALL_YEARS, TRAIN_YEARS
    from housing_abm.data import load
    from housing_abm.experiment import load_bundle
    from housing_abm.market import Market
    from housing_abm.policies import RulePolicy
    from housing_abm.training import train_targets
    s = importlib.util.spec_from_file_location("ev", PKG / "src" / "src-convergence" / "4_evaluate.py")
    ev = importlib.util.module_from_spec(s)
    s.loader.exec_module(ev)
    d = load()
    targets = train_targets(d)
    proc = psutil.Process()

    name = {"Q-learning": "qlearning", "DQN": "dqn"}.get(spec["model"])
    if spec["model"] == "Rule-based":
        pol, b = RulePolicy(), load_bundle()
    else:
        pol, b = ev.build_policy(name, "main", 0, trained=spec["task"] == "sim", which="final")
    b.market.n_listings = int(round(300 * spec["scale"]))
    out["n_listings_per_year"] = b.market.n_listings
    out["rss_after_load_mb"] = proc.memory_info().rss / MB

    if spec["task"] == "sim":
        Market(d, b.market, b.reward, b.spec, seed=WARMUP_SEED, targets=targets).run(
            ALL_YEARS[:1], copy.deepcopy(pol), eps=0.0, learn=False)
        walls, cpus = [], []
        for k in range(REPEATS):
            w, c = time.perf_counter(), _cpu()
            m = Market(d, b.market, b.reward, b.spec, seed=spec["market_seed"], targets=targets)
            path = m.run(ALL_YEARS, pol if k == 0 else copy.deepcopy(pol), eps=0.0, learn=False)
            walls.append(time.perf_counter() - w)
            cpus.append(_cpu() - c)
            if k == 0:
                m0, p0 = m, path
        out["sim_s"] = float(np.median(walls))
        out["sim_cpu_s"] = float(np.median(cpus))
        out["negotiations"] = int(m0.stats.n_neg)
        out["sales"] = int(p0.sales.sum())
        out["buyers_mean"] = float(p0.buyers.mean())
        out["listings_mean"] = float(p0.listings.mean())
    else:
        times, negs = [], []
        for ep in range(TRAIN_EPISODES):
            m = Market(d, b.market, b.reward, b.spec, seed=[0, 10_000 + ep], targets=targets)
            w = time.perf_counter()
            m.run(TRAIN_YEARS, pol, eps=0.5, learn=True)
            times.append(time.perf_counter() - w)
            negs.append(m.stats.n_neg)
        out["episode_s"] = float(np.mean(times))
        out["episode_negotiations"] = float(np.mean(negs))
    out["peak_rss_mb"] = _maxrss_mb()
    return out


# ============================================================================ driver
def specs():
    S = []
    for k in SCALES:
        for mdl in MODELS:
            S += [{"task": "sim", "model": mdl, "scale": k, "market_seed": ms} for ms in SIM_SEEDS]
        for mdl in ["Q-learning", "DQN"]:
            S.append({"task": "train", "model": mdl, "scale": k})
    return S


def run_all():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **THREAD_ENV}
    rows = []
    S = specs()
    t0 = time.time()
    for i, sp in enumerate(S, 1):
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(sp)], env=env,
                           capture_output=True, text=True, check=True)
        rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
        print(f"[{i}/{len(S)}] {sp} {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "scaling_runs.csv", index=False)


def summarise():
    import numpy as np
    import pandas as pd
    R = pd.read_csv(OUT / "scaling_runs.csv")
    sim = R[R.task == "sim"].copy()
    sim["neg_per_s"] = sim.negotiations / sim.sim_s
    S = sim.groupby(["model", "scale"]).agg(
        n_listings=("n_listings_per_year", "first"), active_buyers=("buyers_mean", "mean"),
        active_listings=("listings_mean", "mean"), negotiations=("negotiations", "mean"),
        sim_s=("sim_s", "mean"), sim_s_sd=("sim_s", "std"), neg_per_s=("neg_per_s", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean")).reset_index()
    tr = R[R.task == "train"][["model", "scale", "episode_s", "episode_negotiations", "peak_rss_mb"]]
    tr = tr.rename(columns={"peak_rss_mb": "train_peak_rss_mb"})
    S = S.merge(tr, on=["model", "scale"], how="left")
    # growth relative to the calibrated market (scale 1) and log-log slope (cost ~ k^slope)
    for c in ["sim_s", "peak_rss_mb", "episode_s"]:
        base = S[S.scale == 1].set_index("model")[c]
        S[f"{c}_x"] = S[c] / S.model.map(base)
    slopes = []
    for mdl, g in S.groupby("model"):
        row = {"model": mdl}
        for c in ["sim_s", "episode_s"]:
            h = g.dropna(subset=[c])
            if len(h) > 1:
                row[f"{c}_slope"] = np.polyfit(np.log(h.scale), np.log(h[c]), 1)[0]
        slopes.append(row)
    S.to_csv(OUT / "scaling_summary.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUT / "scaling_slopes.csv", index=False)
    print(S.round(3).to_string(index=False))
    print(pd.DataFrame(slopes).round(2).to_string(index=False))
    figure(S)


def figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="paper", style="ticks", font="serif",
                  rc={"pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8})
    panels = [("sim_s", "Simulation time, 2010–2024 (s)", MODELS),
              ("episode_s", "Training time per episode (s)", ["Q-learning", "DQN"]),
              ("peak_rss_mb", "Peak RAM, simulation (MB)", MODELS)]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for ax, (c, lab, mods) in zip(axes, panels):
        for m in mods:
            g = S[S.model == m].sort_values("scale")
            ax.plot(g.n_listings, g[c], color=COLORS[m], ls=LINES[m], marker=MARKERS[m], ms=4.5, lw=1.5,
                    mec="white", mew=0.6, label=m)
        ax.set_xscale("log", base=2)
        if c != "peak_rss_mb":
            ax.set_yscale("log")
        ax.set_xticks(sorted(S.n_listings.unique()))
        ax.set_xticklabels([str(int(v)) for v in sorted(S.n_listings.unique())])
        ax.tick_params(labelsize=7.5)
        ax.yaxis.grid(True, color="0.9", lw=0.6)
        ax.set_xlabel("New listings per year", fontsize=8.5)
        ax.set_title(lab, fontsize=8.5, loc="left")
        sns.despine(ax=ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False,
               handlelength=3)
    fig.tight_layout()
    GRAPHS.mkdir(parents=True, exist_ok=True)
    p = GRAPHS / "Scaling-Population-Costs"
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args()
    if a.worker:
        print(json.dumps(worker(json.loads(a.worker))))
    else:
        if not a.summarise_only:
            run_all()
        summarise()
