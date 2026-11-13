"""CPU-only performance and resource-utilisation comparison: rule-based, Q-learning and DQN.

Nothing in the market, the policies or the training code is changed; this script only times and
measures existing code paths.

Simulation benchmark (measured here)
------------------------------------
* Frozen-policy evaluation exactly as in 4_evaluate.py: shared calibrated market, 2010-2024,
  epsilon = 0, no learning. 20 market seeds (1000-1019) for every model; the 10 saved end-of-training
  policies for each of Q-learning and DQN (420 runs).
* One fresh Python process per run, executed one at a time (no concurrent runs), so each run has its
  own peak-RAM figure and no run competes with another for CPU.
* Identical thread configuration in every run: OMP/MKL/OPENBLAS/VECLIB/NUMEXPR threads = 1, and
  torch.set_num_threads(1) (already set in housing_abm/policies.py).
* Phases timed separately inside each process: imports; policy loading (building the policy and
  reading its saved state); a warm-up simulation of one year on a copy of the policy and an unused
  market seed (excluded from the timed run, so first-call overheads do not count as simulation time);
  and the timed simulation, repeated 5 times in the process (median reported; one run lasts only
  ~0.05 s). Repeats 2-5 use a copy of the policy so the first, output-checked repeat is unchanged.
* CPU utilisation = process CPU time (user + system, all threads) / wall time x 100, i.e. per cent
  of ONE core; the machine-normalised value divides by the number of logical cores.
* Peak RAM = maximum resident set size of the process (getrusage), in MB (2^20 bytes).
* Completed negotiations = buyer-listing negotiations that ended (deal, lost to a rival or failed);
  deals are also reported. Throughput = count / simulation wall time.
* Instrumentation check: every run's yearly output (median price, quantiles, sales, ratio, ...) is
  compared with the stored 4_evaluate.py output for the same model, training seed and market seed.

Training costs (reused, not re-measured)
----------------------------------------
Read from the saved training files of 3_train_final.py: wall time to the convergence episode
(checkpoint ``elapsed_s``) and total training wall time. Those runs used 8 concurrent worker
processes with one torch thread each; their peak RAM and CPU use were not recorded and are reported
as not measured.

Usage (from the package root):
    python src/src-model-complexity/compute_benchmark.py            # all 420 runs, then summaries
    python src/src-model-complexity/compute_benchmark.py --summarise-only
Writes results/results-complexity/ and graphs/complexity/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
OUT = PKG / "results" / "results-complexity"
GRAPHS = PKG / "graphs" / "complexity"
THREAD_ENV = {k: "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]}
MODELS = ["Rule-based", "Q-learning", "DQN"]
COLORS = {"Rule-based": "#2a78d6", "Q-learning": "#eb6834", "DQN": "#1baf7a"}
WARMUP_SEED = 999          # not one of the 20 evaluation seeds
REPEATS = 5                # timed repeats of each run; median reported (runs last only ~0.05 s)
MB = 2 ** 20


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / MB if sys.platform == "darwin" else r / 1024   # bytes on macOS, KiB on Linux


# ============================================================================ worker (one run)
def worker(spec: dict) -> dict:
    out = {**spec}
    w0, c0 = time.perf_counter(), _cpu()
    import copy
    import importlib.util

    import numpy as np
    import pandas as pd
    import psutil
    sys.path.insert(0, str(PKG / "src" / "src-convergence"))
    from housing_abm.config import ALL_YEARS, RESULTS
    from housing_abm.data import load
    from housing_abm.experiment import load_bundle
    from housing_abm.market import Market
    from housing_abm.policies import RulePolicy
    from housing_abm.training import train_targets
    import torch
    s = importlib.util.spec_from_file_location("ev", PKG / "src" / "src-convergence" / "4_evaluate.py")
    ev = importlib.util.module_from_spec(s)
    s.loader.exec_module(ev)
    d = load()
    targets = train_targets(d)
    proc = psutil.Process()
    out["import_s"] = time.perf_counter() - w0
    out["rss_after_import_mb"] = proc.memory_info().rss / MB
    out["torch_threads"] = torch.get_num_threads()

    # ---- policy loading
    w1, c1 = time.perf_counter(), _cpu()
    if spec["model"] == "Rule-based":
        pol, b = RulePolicy(), load_bundle()
    else:
        name = {"Q-learning": "qlearning", "DQN": "dqn"}[spec["model"]]
        pol, b = ev.build_policy(name, "main", spec["train_seed"], trained=True, which="final")
    out["load_s"] = time.perf_counter() - w1
    out["load_cpu_s"] = _cpu() - c1
    out["rss_after_load_mb"] = proc.memory_info().rss / MB

    # ---- warm-up on a copy of the policy (timed run unaffected)
    w2 = time.perf_counter()
    Market(d, b.market, b.reward, b.spec, seed=WARMUP_SEED, targets=targets).run(
        ALL_YEARS[:1], copy.deepcopy(pol), eps=0.0, learn=False)
    out["warmup_s"] = time.perf_counter() - w2
    rss_before = proc.memory_info().rss / MB

    # ---- timed simulation (same call as training.evaluate), repeated; the first repeat is the
    #      one checked against the stored output, the median wall/CPU time is reported
    walls, cpus = [], []
    for k in range(REPEATS):
        w3, c3 = time.perf_counter(), _cpu()
        mk = Market(d, b.market, b.reward, b.spec, seed=spec["market_seed"], targets=targets)
        pk = mk.run(ALL_YEARS, pol if k == 0 else copy.deepcopy(pol), eps=0.0, learn=False)
        walls.append(time.perf_counter() - w3)
        cpus.append(_cpu() - c3)
        if k == 0:
            m, path = mk, pk
    out["sim_s"] = float(np.median(walls))
    out["sim_s_min"] = float(np.min(walls))
    out["sim_s_first"] = walls[0]
    out["sim_cpu_s"] = float(np.median(cpus))
    out["repeats"] = REPEATS
    out["rss_before_sim_mb"] = rss_before
    out["peak_rss_mb"] = _maxrss_mb()
    out["negotiations"] = int(m.stats.n_neg)
    out["deals"] = int(m.stats.n_win)
    out["sales"] = int(path["sales"].sum())
    out["total_s"] = time.perf_counter() - w0
    out["total_cpu_s"] = _cpu() - c0

    # ---- instrumentation check against stored 4_evaluate output
    st = pd.read_csv(RESULTS / "evaluation" / "yearly_sim_vs_obs_runs.csv")
    st = st[(st.env == "shared") & (st.label == spec["model"]) & (st.market_seed == spec["market_seed"])]
    if spec["model"] != "Rule-based":
        st = st[st.train_seed == spec["train_seed"]]
    st = st.set_index("year").sort_index()
    p = path.set_index("year").sort_index()
    diffs = []
    for c in ["median", "p25", "p75", "mean", "sales", "sale_to_ask", "sell_through", "ask_index", "ratio"]:
        a, z = p[c].to_numpy(float), st[c].to_numpy(float)
        if (np.isnan(a) != np.isnan(z)).any():      # missing in one output only = mismatch
            diffs.append(float("inf"))
            continue
        ok = ~np.isnan(a)
        rel = np.abs(a[ok] - z[ok]) / np.maximum(np.abs(z[ok]), 1e-12)
        diffs.append(float(rel.max()) if len(rel) else 0.0)
    out["n_years_compared"] = len(st)
    out["max_rel_diff_vs_stored"] = max(diffs)
    return out


# ============================================================================ driver
def run_all():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1000, 1020))
    specs = []
    for ms in seeds:              # interleave models so drift in machine state affects all equally
        specs.append({"model": "Rule-based", "train_seed": -1, "market_seed": ms})
        for t in range(10):
            specs.append({"model": "Q-learning", "train_seed": t, "market_seed": ms})
            specs.append({"model": "DQN", "train_seed": t, "market_seed": ms})
    env = {**os.environ, **THREAD_ENV, "PYTHONHASHSEED": "0"}
    rows, load_avg = [], []
    path = OUT / "benchmark_runs.csv"
    import pandas as pd
    done = set()
    if path.exists():  # resume an interrupted benchmark: keep completed runs, do the rest
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(zip(prev.model, prev.train_seed, prev.market_seed))
        print(f"resuming: {len(done)} runs already measured", flush=True)
    for i, sp in enumerate(specs):
        if (sp["model"], sp["train_seed"], sp["market_seed"]) in done:
            continue
        load_avg.append(os.getloadavg()[0])
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(sp)], env=env,
                           capture_output=True, text=True, cwd=PKG)
        if r.returncode != 0:
            raise SystemExit(f"run failed: {sp}\n{r.stderr[-2000:]}")
        row = json.loads(r.stdout.strip().splitlines()[-1])
        row["load_avg_1min_before"] = load_avg[-1]
        rows.append(row)
        if (i + 1) % 21 == 0:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{i + 1}/{len(specs)} runs", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def environment() -> dict:
    import numpy
    import pandas
    import psutil
    import torch

    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    return {
        "cpu_model": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "performance_cores": sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": sysctl("hw.perflevel1.physicalcpu"),
        "ram_total_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "numpy": numpy.__version__, "pandas": pandas.__version__, "torch": torch.__version__,
        "blas": "Apple Accelerate (numpy build config)" if sys.platform == "darwin" else "see numpy.show_config()",
        "gpu_used": False,
        "torch_cuda_available": torch.cuda.is_available(),
        "thread_env": THREAD_ENV, "torch_num_threads": 1,
        "execution": "one process per run, runs executed sequentially",
        "cpu_percent_definition": "process CPU time / wall time x 100 = % of one core; "
                                  "machine-wide % = that value / logical cores",
    }


# ============================================================================ training (reused)
def training_costs():
    import json as js

    import pandas as pd
    tr = PKG / "results" / "results-convergence" / "training" / "main"
    rows = []
    for model, tag in [("Q-learning", "qlearning"), ("DQN", "dqn")]:
        for s in range(10):
            meta = js.loads((tr / f"{tag}_s{s}_meta.json").read_text())
            ck = pd.read_csv(tr / f"{tag}_s{s}_checkpoints.csv")
            ep = pd.read_csv(tr / f"{tag}_s{s}_episodes.csv")
            ce = meta.get("converged_episode")
            conv = ck.loc[ck.episode == ce, "elapsed_s"]
            rows.append({"model": model, "train_seed": s, "converged": meta.get("converged"),
                         "converged_episode": ce, "episodes_run": int(ep.episode.max()),
                         "time_to_convergence_s": float(conv.iloc[0]) if len(conv) else float("nan"),
                         "total_training_s": float(meta["train_time_s"]),
                         "n_params": meta.get("n_params"),
                         "policy_file_kb": (tr / f"{tag}_s{s}_final.pkl").stat().st_size / 1024,
                         "source": "3_train_final.py checkpoints (8 concurrent workers, 1 torch thread each)"})
    return pd.DataFrame(rows)


# ============================================================================ summaries
def summarise():
    import numpy as np
    import pandas as pd
    R = pd.read_csv(OUT / "benchmark_runs.csv")
    env = environment()
    ncpu = env["logical_cores"]
    R["sim_cpu_pct_one_core"] = 100 * R.sim_cpu_s / R.sim_s
    R["sim_cpu_pct_machine"] = R.sim_cpu_pct_one_core / ncpu
    R["negotiations_per_s"] = R.negotiations / R.sim_s
    R["deals_per_s"] = R.deals / R.sim_s
    R["sim_ram_increase_mb"] = R.peak_rss_mb - R.rss_after_import_mb
    R.to_csv(OUT / "benchmark_runs.csv", index=False)

    T = training_costs()
    T.to_csv(OUT / "training_costs.csv", index=False)

    checks = {
        "runs": len(R),
        "runs_per_model": R.model.value_counts().to_dict(),
        "max_rel_diff_vs_stored_output": float(R.max_rel_diff_vs_stored.max()),
        "all_15_years_compared": bool((R.n_years_compared == 15).all()),
        "torch_threads_all_1": bool((R.torch_threads == 1).all()),
        "load_avg_1min_before_runs_mean": float(R.load_avg_1min_before.mean()),
        "load_avg_1min_before_runs_max": float(R.load_avg_1min_before.max()),
    }
    (OUT / "environment.json").write_text(json.dumps({**env, "checks": checks}, indent=2))

    cols = {"sim_s": "Simulation time per run (s)", "load_s": "Policy loading (s)",
            "import_s": "Imports and data loading (s)", "warmup_s": "Warm-up, 1 year (s)",
            "peak_rss_mb": "Peak RAM (MB)", "rss_after_load_mb": "RAM after policy load (MB)",
            "sim_ram_increase_mb": "RAM increase during load and simulation (MB)",
            "sim_cpu_pct_one_core": "CPU, % of one core", "sim_cpu_pct_machine": "CPU, % of machine",
            "negotiations_per_s": "Completed negotiations per second", "deals_per_s": "Deals per second",
            "negotiations": "Negotiations per run", "total_s": "Process total (s)"}
    S = []
    for m in MODELS:
        g = R[R.model == m]
        row = {"model": m, "runs": len(g)}
        for c in cols:
            row[f"{c}_mean"], row[f"{c}_sd"] = g[c].mean(), g[c].std(ddof=1)
        t = T[T.model == m]
        for c in ["time_to_convergence_s", "total_training_s", "converged_episode", "policy_file_kb"]:
            row[f"{c}_mean"] = t[c].mean() if len(t) else np.nan
            row[f"{c}_sd"] = t[c].std(ddof=1) if len(t) else np.nan
        S.append(row)
    S = pd.DataFrame(S)
    S.to_csv(OUT / "benchmark_summary.csv", index=False)
    latex_table(S, env)
    figure(R)
    pd.set_option("display.width", 250)
    print(json.dumps(checks, indent=1))
    show = ["model"] + [f"{c}_{k}" for c in ["sim_s", "load_s", "peak_rss_mb", "sim_cpu_pct_one_core",
                                              "negotiations_per_s", "time_to_convergence_s",
                                              "total_training_s"] for k in ("mean", "sd")]
    print(S[show].round(3).T.to_string())


def latex_table(S, env):
    import numpy as np

    def f(r, c, d):
        if not np.isfinite(r[f"{c}_mean"]):
            return "--"
        return f"{r[c + '_mean']:.{d}f} ({r[c + '_sd']:.{d}f})"
    rows = [("Training to convergence (min)", lambda r: "--" if r.model == "Rule-based" else
             f"{r.time_to_convergence_s_mean / 60:.1f} ({r.time_to_convergence_s_sd / 60:.1f})"),
            ("Total training (min)", lambda r: "--" if r.model == "Rule-based" else
             f"{r.total_training_s_mean / 60:.1f} ({r.total_training_s_sd / 60:.1f})"),
            ("Policy loading (ms)", lambda r: f"{1e3 * r.load_s_mean:.1f} ({1e3 * r.load_s_sd:.1f})"),
            ("Simulation, 2010--2024 (s)", lambda r: f(r, "sim_s", 3)),
            ("Peak RAM (MB)", lambda r: f(r, "peak_rss_mb", 0)),
            ("\\quad above shared baseline after imports (MB)", lambda r: f(r, "sim_ram_increase_mb", 1)),
            ("CPU utilisation (\\% of one core)", lambda r: f(r, "sim_cpu_pct_one_core", 0)),
            ("CPU utilisation (\\% of machine, 8 cores)", lambda r: f(r, "sim_cpu_pct_machine", 1)),
            ("Completed negotiations per second", lambda r: f"{r.negotiations_per_s_mean:,.0f} ({r.negotiations_per_s_sd:,.0f})"),
            ("Policy size (kB)", lambda r: "--" if r.model == "Rule-based" else f"{r.policy_file_kb_mean:.0f}")]
    head = " & ".join(["Measure"] + [m.replace("DQN", "DQN") for m in S.model]) + r" \\"
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", head, r"\midrule"]
    for lab, fn in rows:
        lines.append(" & ".join([lab] + [fn(r) for _, r in S.iterrows()]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_compute_comparison.tex").write_text("\n".join(lines) + "\n")


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="paper", style="ticks", font="serif",
                  rc={"pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8})
    import pandas as pd
    GRAPHS.mkdir(parents=True, exist_ok=True)
    T = training_costs()
    T["train_min"] = T.time_to_convergence_s / 60
    R = R.assign(sim_ms=R.sim_s * 1e3)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
    panels = [(T, "train_min", "Training to convergence (min)", True),
              (R, "sim_ms", "Simulation time per run (ms)", False),
              (R, "peak_rss_mb", "Peak RAM (MB)", False)]
    for ax, (df, col, lab, log) in zip(axes, panels):
        sns.barplot(data=df, x="model", y=col, hue="model", order=MODELS, hue_order=MODELS, palette=COLORS,
                    errorbar="sd", err_kws={"linewidth": 0.9, "color": "0.2"}, capsize=0.15, width=0.6,
                    legend=False, ax=ax, alpha=0.9)
        sns.stripplot(data=df, x="model", y=col, order=MODELS, color="0.15", size=1.6, alpha=0.35,
                      jitter=0.18, ax=ax)
        if log:
            ax.set_yscale("log")
            ax.text(0, ax.get_ylim()[0] * 1.6, "no\ntraining", ha="center", va="bottom", fontsize=7, color="0.35")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("")
        ax.set_ylabel(lab, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.tick_params(axis="x", rotation=25)
        for t in ax.get_xticklabels():
            t.set_ha("right")
            t.set_rotation_mode("anchor")
        sns.despine(ax=ax)
    fig.tight_layout()
    p = GRAPHS / "Compute-Time-Memory-Comparison.pdf"
    fig.savefig(p, bbox_inches="tight")
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
