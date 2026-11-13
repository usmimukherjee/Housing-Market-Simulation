"""Step 5. Figures, tables and summary CSV files from the outputs of steps 1-4.

Figures (graphs/):
  RL-DRL-Training-Convergence.png              new Figure 5: learning over training episodes
  RL-Training-Convergence.png                  Q-learning diagnostics in detail
  DRL-Training-Convergence.png                 DQN diagnostics in detail
  RL-DRL-Validation-Error.png                  validation RMSE at each training checkpoint
  Rule-Based-RL-DRL-Calibration-2010-2024.png  new Figure 6: observed vs simulated, shared environment
  Rule-Based-RL-DRL-Recalibrated-2010-2024.png same, with the environment recalibrated per model
  Rule-Based-RL-DRL-Residuals-2010-2024.png    yearly percentage error
  RL-DRL-Reward-Ablation.png                   reward / state / action ablations
  RL-DRL-Learning-Sanity-Checks.png            deviation test and reference policies

Tables (results/results-convergence/tables/, CSV and LaTeX):
  table1_hyperparameters, table2_convergence, table3_performance, table4_vs_rule_based,
  table5_ablation, table6_sanity_checks
"""

from __future__ import annotations

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from housing_abm.calibration import GRID, TOLERANCE
from housing_abm.config import (CALIB_SEEDS, CHECKPOINT_EVAL_SEEDS, FINAL_EVAL_SEEDS, GRAPHS as _GRAPHS, RESULTS as _RESULTS,
                                TEST_YEARS, TRAIN_SEEDS, TRAIN_YEARS, VAL_YEARS, DQNParams, QParams,
                                RewardParams, Spec, TrainParams, ValScore)
from housing_abm.data import load
from housing_abm.experiment import asdict_all, load_bundle
from housing_abm.metrics import convergence, summarize

import os as _os
_Q = bool(_os.environ.get("EVAL_QUICK"))
RESULTS = _RESULTS
GRAPHS = (_RESULTS / "evaluation-quick" / "graphs") if _Q else _GRAPHS / "archive"  # diagnostic figures
EV = RESULTS / ("evaluation-quick" if __import__("os").environ.get("EVAL_QUICK") else "evaluation")
TR = RESULTS / "training"
TAB = (RESULTS / "evaluation-quick" / "tables") if _Q else RESULTS / "tables"

INK, INK2, GRID_C, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"Rule-based": "#2a78d6", "Q-learning": "#eb6834", "DQN": "#1baf7a", "Random": "#8a8984",
       "Observed": INK}
MODEL_LABEL = {"qlearning": "Q-learning", "dqn": "DQN"}
PERIOD_SHADE = {"Training": (2010, 2018.5), "Validation": (2018.5, 2021.5), "Test": (2021.5, 2024.5)}
SHADE_C = {"Training": "#f1f0ec", "Validation": "#e6eef9", "Test": "#fbe9e1"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6, "axes.facecolor": SURF,
    "figure.facecolor": "white", "lines.linewidth": 1.8, "axes.titleweight": "bold",
    "axes.titlesize": 9.5, "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------- loading
def load_training(variant="main"):
    E, C = [], []
    for model in ["qlearning", "dqn"]:
        for f in sorted(glob.glob(str(TR / variant / f"{model}_s*_episodes.csv"))):
            s = int(f.split("_s")[-1].split("_")[0])
            E.append(pd.read_csv(f).assign(model=model, seed=s, variant=variant))
            C.append(pd.read_csv(f.replace("_episodes", "_checkpoints")).assign(model=model, seed=s,
                                                                                 variant=variant))
    return pd.concat(E, ignore_index=True), pd.concat(C, ignore_index=True)


def band(df, x, y, q=None):
    """Mean across seeds and 95% t-interval of the mean."""
    g = df.groupby(x)[y]
    m, s, n = g.mean(), g.std(), g.count()
    h = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * s / np.sqrt(n)
    return m.index.to_numpy(), m.to_numpy(), (m - h).to_numpy(), (m + h).to_numpy()


def shade_periods(ax, labels=True, top=None):
    for name, (a, b) in PERIOD_SHADE.items():
        ax.axvspan(a, b, color=SHADE_C[name], zorder=0, lw=0)
        if labels:
            ax.text((a + b) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, color=INK2)


def save(fig, name):
    GRAPHS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHS / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", GRAPHS / name)


def latex(df, path, caption_cols=None, fmt=None):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    cols = "l" + "r" * (len(df.columns) - 1)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             " & ".join(str(c).replace("%", "\\%").replace("_", " ") for c in df.columns) + " \\\\",
             "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append("--" if not np.isfinite(v) else f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}")
            else:
                cells.append(str(v).replace("%", "\\%").replace("_", " ").replace("&", "\\&"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.with_suffix(".tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- figures
def ref_rewards(met):
    sh = met[(met["env"] == "shared")]
    return {lab: sh[sh["label"] == lab]["reward_per_negotiation"].mean()
            for lab in ["Rule-based", "Random", "Always +6%"]}


def fig_training(E, C, met, conv):
    ref = ref_rewards(met)
    tp = TrainParams()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for row, model in enumerate(["qlearning", "dqn"]):
        lab = MODEL_LABEL[model]
        e, c = E[E.model == model], C[C.model == model]
        col = COL[lab]
        ax = axes[row, 0]
        x, m, lo, hi = band(e, "episode", "reward_ma")
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, m, color=col, label=f"{lab}, training (ε-greedy), {tp.ma_window}-episode MA")
        x, m, lo, hi = band(c, "episode", "greedy_reward")
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", ms=3, color=INK, elinewidth=0.8,
                    label="Greedy checkpoint (ε = 0)")
        for k, ls in [("Rule-based", "--"), ("Random", ":"), ("Always +6%", "-.")]:
            ax.axhline(ref[k], color=COL.get(k, INK2), ls=ls, lw=1.2, label=f"{k} (frozen)")
        ax.set_title(f"{'AB'[row]}. {lab}: reward per negotiation")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Reward per negotiation")
        ax.legend(loc="center right", fontsize=7)
        ax = axes[row, 1]
        x, m, lo, hi = band(e, "episode", "td_error")
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, m, color=col)
        ax.set_title(f"{lab}: {'mean |TD error|' if model == 'qlearning' else 'Huber TD loss'}")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Per update, mean over episode")
        ax = axes[row, 2]
        x, m, lo, hi = band(c.dropna(subset=["policy_change"]), "episode", "policy_change")
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, m, color=col, marker="o", ms=2.5)
        ax.axhline(tp.conv_policy_change, color=INK2, ls="--", lw=1, label="Convergence threshold")
        ce = conv[conv.model == model]["converged_episode"].dropna()
        if len(ce):
            ax.axvline(ce.median(), color=INK, lw=1, ls=":",
                       label=f"Median convergence episode ({len(ce)}/{len(conv[conv.model == model])} seeds)")
        ax.set_ylim(0, None)
        ax.set_title(f"{lab}: policy change between checkpoints")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Share of probe states with new action")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Training convergence over training episodes (mean of 10 seeds, 95% interval of the mean)",
                 fontsize=10, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "RL-DRL-Training-Convergence.png")


def fig_model_detail(E, C, model):
    lab = MODEL_LABEL[model]
    col = COL[lab]
    e, c = E[E.model == model], C[C.model == model]
    panels = [("reward_ma", "Training reward per negotiation (MA)", e),
              ("deal_rate", "Deal (win) rate per negotiation", e),
              ("mean_rounds", "Rounds per negotiation", e),
              ("mean_price_error", "|price - target| / target, winners", e),
              ("epsilon", "Exploration rate ε", e),
              ("q_max" if model == "qlearning" else "q_mean", "Max Q (visited)" if model == "qlearning"
               else "Mean Q (batch)", e),
              ("q_change_pct" if model == "qlearning" else "td_error",
               "Q-table change per episode (%)" if model == "qlearning" else "Huber TD loss", e),
              ("val_rmse", "Validation RMSE of median price ($), checkpoints", c)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, (y, title, df) in zip(axes.flat, panels):
        x, m, lo, hi = band(df.dropna(subset=[y]), "episode", y)
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, m, color=col, marker="o" if df is c else None, ms=2.5)
        ax.set_title(title)
        ax.set_xlabel("Training episode")
    acts = [k for k in e.columns if k.startswith("action_share_")]
    fig.suptitle(f"{lab}: training diagnostics (mean of {e.seed.nunique()} seeds, 95% interval)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, ("RL" if model == "qlearning" else "DRL") + "-Training-Convergence.png")


def fig_validation(C, met):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for model in ["qlearning", "dqn"]:
        lab = MODEL_LABEL[model]
        c = C[C.model == model]
        x, m, lo, hi = band(c, "episode", "val_rmse")
        ax.fill_between(x, lo / 1e3, hi / 1e3, color=COL[lab], alpha=0.25, lw=0)
        ax.plot(x, m / 1e3, color=COL[lab], marker="o", ms=2.5, label=lab)
    sh = met[met.env == "shared"]
    for k, ls in [("Rule-based", "--"), ("Random", ":")]:
        v = sh[(sh.label == k) & sh.market_seed.isin(FINAL_EVAL_SEEDS)]["val_rmse"].mean()
        ax.axhline(v / 1e3, color=COL[k], ls=ls, lw=1.2, label=f"{k} (frozen)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Validation RMSE, median price ($000)")
    ax.set_title("Validation error (2019-2021) during training, greedy checkpoints")
    ax.text(0.01, 0.01, "Curves: 5 checkpoint market seeds, mean of 10 training seeds with 95% interval. "
            "Reference lines: 20 final-evaluation seeds.", transform=ax.transAxes, fontsize=6.5, color=INK2)
    ax.legend(ncol=2)
    fig.tight_layout()
    save(fig, "RL-DRL-Validation-Error.png")


def fig_calibration(Y, bench, env, name):
    d = load()
    o = d.observed
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    shade_periods(ax)
    y = Y[(Y.env == env)]
    for lab in ["Rule-based", "Q-learning", "DQN"]:
        g = y[y.label == lab].groupby("year")["median"]
        ax.fill_between(g.mean().index, g.quantile(0.1) / 1e3, g.quantile(0.9) / 1e3, color=COL[lab],
                        alpha=0.18, lw=0)
        ax.plot(g.mean().index, g.mean() / 1e3, color=COL[lab], marker="o", ms=3, label=lab)
    ax.plot(o.index, o["median"] / 1e3, color=INK, marker="s", ms=4, lw=2.2, label="Observed (HRM sales)")
    tb = bench[bench.benchmark == "Log-linear trend fitted 2010-2018"]
    ax.plot(tb.year, tb["median"] / 1e3, color=INK2, ls=":", lw=1.2, label="Trend fitted 2010-2018")
    ax.set_xlim(2009.6, 2024.4)
    ax.set_xticks(range(2010, 2025, 2))
    ax.set_ylabel("Median sale price ($000)")
    ax.set_xlabel("Year")
    ax.set_title("Median sale price: observed vs simulated", pad=14)
    ax.legend(loc="upper left", ncol=1)
    ax = axes[1]
    for name_, (a, b) in PERIOD_SHADE.items():
        ax.axvspan(max(a, 2018.6), b, color=SHADE_C[name_], zorder=0, lw=0)
    for lab in ["Rule-based", "Q-learning", "DQN"]:
        g = y[(y.label == lab) & y.year.between(2019, 2023)].groupby("year")["ratio"]
        ax.fill_between(g.mean().index, g.quantile(0.1), g.quantile(0.9), color=COL[lab], alpha=0.18, lw=0)
        ax.plot(g.mean().index, g.mean(), color=COL[lab], marker="o", ms=3, label=lab)
    r = d.observed_ratio
    ax.plot(r.index, r.values, color=INK, marker="s", ms=4, lw=2.2, label="Observed")
    ax.set_xlim(2018.6, 2023.4)
    ax.set_xticks(range(2019, 2024))
    ax.set_title("Assessed value / sale price (median)", pad=14)
    ax.set_xlabel("Tax year")
    ax.set_ylabel("Ratio")
    env_txt = "shared environment" if env == "shared" else "environment recalibrated per model"
    fig.suptitle(f"Historical calibration and out-of-sample validation, {env_txt}. Lines: mean over runs; "
                 "bands: 10th-90th percentile of runs", fontsize=9, y=1.03)
    fig.tight_layout()
    save(fig, name)


def fig_residuals(Y):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    shade_periods(ax)
    y = Y[Y.env == "shared"]
    w = 0.26
    for i, lab in enumerate(["Rule-based", "Q-learning", "DQN"]):
        g = y[y.label == lab].groupby("year")
        e = (g["median"].mean() - g["obs_median"].first()) / g["obs_median"].first() * 100
        ax.bar(e.index + (i - 1) * w, e.values, width=w - 0.03, color=COL[lab], label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(2010, 2025, 2))
    ax.set_ylabel("(simulated - observed) / observed, %")
    ax.set_title("Yearly error of the simulated median price, shared environment", pad=14)
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "Rule-Based-RL-DRL-Residuals-2010-2024.png")


def fig_ablation(abl):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    order = abl.groupby("Variant")["val_score"].mean().sort_values().index
    for ax, ycol, title in [(axes[0], "val_score", "Validation score (lower is better)"),
                            (axes[1], "greedy_reward", "Greedy reward per negotiation, training years")]:
        for i, v in enumerate(order):
            g = abl[abl.Variant == v]
            lab = g["Model"].iloc[0]
            ax.scatter(g[ycol], np.full(len(g), i) + np.random.default_rng(0).uniform(-.12, .12, len(g)),
                       s=14, color=COL[lab], alpha=0.7)
            ax.plot([g[ycol].mean()] * 2, [i - .3, i + .3], color=INK, lw=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[1].text(0.99, 0.01, "Greedy reward is not comparable across reward definitions",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    fig.suptitle("Ablations (dots: seeds, bar: mean). Selection uses validation years only",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "RL-DRL-Reward-Ablation.png")


def fig_sanity(dev):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    devs = ["Learned", "Random", "Always +1%", "Always +6%"]
    pops = ["Random", "Always +6%", "Learned"]
    shades = ["#3a3a38", "#8a8984", "#c9c8c2", "#e3e2dd"]
    for ax, model in zip(axes, ["qlearning", "dqn"]):
        lab = MODEL_LABEL[model]
        g = dev[dev.model == model].groupby(["population", "deviator", "train_seed"])["deviator_reward"].mean()
        g = g.reset_index()
        w = 0.2
        for j, dv in enumerate(devs):
            vals = [g[(g.population == p) & (g.deviator == dv)]["deviator_reward"] for p in pops]
            m = [v.mean() for v in vals]
            se = [stats.t.ppf(.975, len(v) - 1) * v.std() / np.sqrt(len(v)) for v in vals]
            ax.bar(np.arange(3) + (j - 1.5) * w, m, width=w - 0.02, yerr=se, capsize=2,
                   color=COL[lab] if dv == "Learned" else shades[j], label=f"10% minority plays: {dv}",
                   error_kw={"elinewidth": 0.8, "ecolor": INK})
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"Population: {p}" for p in pops])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"{lab}: reward of a 10% minority, by population (training years)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Minority reward per negotiation")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [x.replace("10% minority plays: ", "Minority plays: ") for x in l], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout()
    save(fig, "RL-DRL-Learning-Sanity-Checks.png")


# ---------------------------------------------------------------------------- tables
def per_seed(met):
    """One row per model and seed. Learners: training seeds (metrics averaged over market seeds).
    Rule-based / random / fixed: market seeds."""
    num = met.select_dtypes("number").columns.difference(["market_seed", "train_seed"])
    lrn = met[met.train_seed.notna()].groupby(["label", "env", "variant", "train_seed"])[num].mean().reset_index()
    lrn["unit"] = "training seed"
    oth = met[met.train_seed.isna()].copy()
    oth = oth[["label", "env", "variant", "market_seed"] + list(num)].rename(columns={"market_seed": "seed"})
    oth["unit"] = "market seed"
    lrn = lrn.rename(columns={"train_seed": "seed"})
    return pd.concat([lrn, oth], ignore_index=True)


def fmt_ms(x, scale=1.0, dec=1):
    s = summarize(np.asarray(x) / scale)
    return f"{s['mean']:,.{dec}f} ({s['sd']:,.{dec}f})"


def table_hyper(best, tuneq, tuned):
    rows = []
    q = best["qlearning"]
    for k, space in [("alpha", "0.05, 0.1, 0.2, 0.4"), ("alpha_c", "0 (constant), 200 (decaying)"),
                     ("gamma", "0.90, 0.95, 0.99"), ("eps_start", "0.5, 0.8, 1.0"), ("eps_min", "0.01, 0.05")]:
        rows.append({"Model": "Q-learning", "Parameter": k, "Selected": q[k], "Searched": space})
    rows += [{"Model": "Q-learning", "Parameter": "epsilon decay", "Selected": "linear over 50% of episodes",
              "Searched": "fixed"},
             {"Model": "Q-learning", "Parameter": "states x actions", "Selected": "864 x 3", "Searched": "fixed"},
             {"Model": "Q-learning", "Parameter": "training episodes", "Selected": 400, "Searched": "300 in tuning"}]
    dq = best["dqn"]
    for k, space in [("hidden", "[32,32], [64,64], [128,64]"), ("lr", "1e-3, 5e-4, 1e-4"),
                     ("gamma", "0.90, 0.95, 0.99"), ("batch_size", "32, 64, 128"),
                     ("buffer_size", "10,000, 50,000"), ("target_update", "100, 500, 1000 gradient steps")]:
        rows.append({"Model": "DQN", "Parameter": k, "Selected": str(dq[k]), "Searched": space})
    dp = DQNParams()
    rows += [{"Model": "DQN", "Parameter": "replay warm-up", "Selected": f"{dp.warmup} transitions", "Searched": "fixed"},
             {"Model": "DQN", "Parameter": "update frequency", "Selected": f"1 step / {dp.train_every} transitions", "Searched": "fixed"},
             {"Model": "DQN", "Parameter": "loss, gradient clip", "Selected": f"Huber, norm {dp.grad_clip}", "Searched": "fixed"},
             {"Model": "DQN", "Parameter": "epsilon", "Selected": f"{dp.eps_start} to {dp.eps_min}, linear over 50%", "Searched": "fixed"},
             {"Model": "DQN", "Parameter": "inputs, actions", "Selected": "10 standardised features, 3 actions", "Searched": "fixed"},
             {"Model": "DQN", "Parameter": "training episodes", "Selected": 250, "Searched": "200 in tuning (20 sampled configs + 16 stability variants)"}]
    mp = load_bundle().market
    for k in GRID:
        rows.append({"Model": "Market (all)", "Parameter": k, "Selected": getattr(mp, k),
                     "Searched": ", ".join(map(str, GRID[k]))})
    rp = RewardParams()
    rows.append({"Model": "Reward (learners)", "Parameter": "realism reward",
                 "Selected": f"+{rp.deal} deal; -{rp.price_error}|P-P*|/P*; -{rp.overpay} overpay; "
                             f"-{rp.step}/round; -{rp.fail} fail; -{rp.lost} lost", "Searched": "ablation: deal, surplus"})
    df = pd.DataFrame(rows)
    latex(df, TAB / "table1_hyperparameters")
    return df


def table_convergence(runs, E, C, ps, conv):
    rows = []
    sh = ps[(ps.env == "shared") & (ps.variant == "main")]
    for model in ["qlearning", "dqn"]:
        lab = MODEL_LABEL[model]
        r = runs[(runs.model == model) & (runs.variant == "main")]
        cv = conv[conv.model == model]
        e = E[E.model == model]
        last = e[e.episode > e.episode.max() - 20]
        p = sh[sh.label == lab]
        n_conv = int(cv["converged"].sum())
        ce = cv["converged_episode"].dropna()
        row = {
            "Model": lab,
            "Seeds": len(r),
            "Converged seeds": f"{n_conv}/{len(r)}",
            "Episodes to convergence": (f"{ce.median():.0f} [{ce.min():.0f}-{ce.max():.0f}]" if n_conv else "Not reached"),
            "Best checkpoint episode": f"{r['best_episode'].median():.0f}",
            "Final MA reward": fmt_ms(last.groupby("seed")["reward_ma"].last(), dec=3),
            "Greedy reward, untrained": fmt_ms(r["untrained_greedy_reward"], dec=3),
            "Greedy reward, final": fmt_ms(r["final_greedy_reward"], dec=3),
            "Val RMSE ($000)": fmt_ms(p["val_rmse"], 1e3),
            "Val MAE ($000)": fmt_ms(p["val_mae"], 1e3),
            "Test RMSE ($000)": fmt_ms(p["test_rmse"], 1e3),
            "Test MAE ($000)": fmt_ms(p["test_mae"], 1e3),
            "Policy change, last 5 checkpoints": fmt_ms(r["final_policy_change"], dec=3),
            "Final TD error / loss": fmt_ms(last.groupby("seed")["td_error"].mean(), dec=4),
            "Q-table change, last 20 episodes (%)": (fmt_ms(last.groupby("seed")["q_change_pct"].mean(), dec=2)
                                                     if model == "qlearning" else "--"),
            "Mean training time (min)": f"{r['runtime_s'].mean() / 60:.1f}",
        }
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model").T.reset_index().rename(columns={"index": "Quantity"})
    latex(df, TAB / "table2_convergence")
    return df


def table_performance(ps, bench_m):
    rows = []
    sh = ps[(ps.env == "shared") & (ps.variant == "main")]
    order = ["Rule-based", "Q-learning", "DQN", "Q-learning (best val. checkpoint)", "DQN (best val. checkpoint)", "Random", "Untrained Q-learning", "Untrained DQN",
             "Always +1%", "Always +3%", "Always +6%"]
    for lab in order:
        p = sh[sh.label == lab]
        if p.empty:
            continue
        row = {"Model": lab, "Unit (n)": f"{p.unit.iloc[0]} ({len(p)})"}
        for per, name in [("train", "Train"), ("val", "Val"), ("test", "Test")]:
            row[f"{name} RMSE ($000)"] = fmt_ms(p[f"{per}_rmse"], 1e3)
            row[f"{name} MAE ($000)"] = fmt_ms(p[f"{per}_mae"], 1e3)
            row[f"{name} MAPE (%)"] = fmt_ms(p[f"{per}_mape"])
        row["Train sMAPE (%)"] = fmt_ms(p["train_smape"])
        row["Test sMAPE (%)"] = fmt_ms(p["test_smape"])
        row["Train R2"] = fmt_ms(p["train_r2"], dec=2)
        row["Val ratio MAE"] = fmt_ms(p["val_ratio_mae"], dec=3)
        row["Test ratio MAE"] = fmt_ms(p["test_ratio_mae"], dec=3)
        row["Test growth (%/yr)"] = fmt_ms(p["test_growth_sim_pct"])
        rows.append(row)
    for _, b in bench_m.iterrows():
        rows.append({"Model": b["benchmark"], "Unit (n)": "deterministic",
                     "Val RMSE ($000)": f"{b['val_rmse'] / 1e3:,.1f}" if "2021" not in b["benchmark"] else "in sample",
                     "Val MAE ($000)": f"{b['val_mae'] / 1e3:,.1f}" if "2021" not in b["benchmark"] else "in sample",
                     "Val MAPE (%)": f"{b['val_mape']:.1f}" if "2021" not in b["benchmark"] else "in sample",
                     "Test RMSE ($000)": f"{b['test_rmse'] / 1e3:,.1f}", "Test MAE ($000)": f"{b['test_mae'] / 1e3:,.1f}",
                     "Test MAPE (%)": f"{b['test_mape']:.1f}"})
    df = pd.DataFrame(rows).fillna("--")
    latex(df, TAB / "table3_performance")
    return df


def table_vs_rule(met):
    """Improvement = (RMSE_rule - RMSE_model) / RMSE_rule x 100, paired by market seed.
    For learners the model RMSE at a market seed is the mean over training seeds."""
    rows = []
    rng = np.random.default_rng(0)
    for env in ["shared", "recalibrated"]:
        m = met[(met.env == env) & (met.variant == "main")]
        rule = m[m.label == "Rule-based"].set_index("market_seed")
        for lab in ["Q-learning", "DQN"]:
            x = m[m.label == lab].groupby("market_seed").mean(numeric_only=True)
            for per in ["val", "test"]:
                for metric in ["rmse", "mae"]:
                    a = rule.loc[x.index, f"{per}_{metric}"].to_numpy()
                    b = x[f"{per}_{metric}"].to_numpy()
                    imp = (a.mean() - b.mean()) / a.mean() * 100
                    boots = []
                    for _ in range(4000):
                        j = rng.integers(0, len(a), len(a))
                        boots.append((a[j].mean() - b[j].mean()) / a[j].mean() * 100)
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                    rows.append({"Environment": env, "Model": lab, "Period": per,
                                 "Metric": metric.upper(), "Rule-based ($000)": a.mean() / 1e3,
                                 "Model ($000)": b.mean() / 1e3, "Improvement (%)": imp,
                                 "95% CI low": lo, "95% CI high": hi, "Wilcoxon p": p,
                                 "Better than rule": "yes" if lo > 0 else ("no (worse)" if hi < 0 else "no difference")})
    df = pd.DataFrame(rows)
    latex(df, TAB / "table4_vs_rule_based")
    return df


def table_ablation(met, runs):
    m = met[(met.env == "shared") & (met.kind == "trained") & ~met.label.str.contains("checkpoint")]
    g = m.groupby(["model", "variant", "train_seed"]).agg(
        val_score=("val_score", "mean"), val_mape=("val_mape", "mean"), test_mape=("test_mape", "mean"),
        greedy_reward=("reward_per_negotiation", "mean"), deal_rate=("deal_rate", "mean"),
        rounds=("mean_rounds", "mean")).reset_index()
    g = g.merge(runs[["model", "variant", "seed", "converged", "converged_episode", "final_policy_change"]],
                left_on=["model", "variant", "train_seed"], right_on=["model", "variant", "seed"], how="left")
    g["Model"] = g.model.map(MODEL_LABEL)
    g["Variant"] = g.Model + ": " + g.variant
    rows = []
    for v, x in g.groupby("Variant"):
        rows.append({"Variant": v, "Seeds": len(x), "Val score": fmt_ms(x.val_score, dec=3),
                     "Val MAPE (%)": fmt_ms(x.val_mape), "Test MAPE (%)": fmt_ms(x.test_mape),
                     "Greedy reward": fmt_ms(x.greedy_reward, dec=3), "Deal rate": fmt_ms(x.deal_rate, dec=3),
                     "Rounds": fmt_ms(x.rounds, dec=2), "Converged": f"{int(x.converged.sum())}/{len(x)}",
                     "Policy change (last 5)": fmt_ms(x.final_policy_change, dec=3)})
    df = pd.DataFrame(rows).sort_values("Variant")
    latex(df, TAB / "table5_ablation")
    return df, g


def main():
    d = load()
    TAB.mkdir(parents=True, exist_ok=True)
    met = pd.read_csv(EV / "eval_runs.csv")
    Y = pd.read_csv(EV / "yearly_sim_vs_obs_runs.csv")
    dev = pd.read_csv(EV / "deviation_test.csv")
    bench = pd.read_csv(EV / "benchmarks_yearly.csv")
    bench_m = pd.read_csv(EV / "benchmarks.csv")
    runs = pd.read_csv(TR / "training_runs.csv")
    best = json.loads((RESULTS / "tuning" / "best_hyperparameters.json").read_text())
    E, C = load_training("main")
    tp = TrainParams()
    conv = []
    for (model, seed), c in C.groupby(["model", "seed"]):
        r = convergence(c.sort_values("episode").reset_index(drop=True), tp)
        conv.append({"model": model, "seed": seed, "converged": r["converged"], "converged_episode": r["episode"]})
    conv = pd.DataFrame(conv)
    conv.to_csv(RESULTS / "convergence_by_seed.csv", index=False)

    # ---- CSV outputs requested by the protocol
    E.to_csv(RESULTS / "training_curves.csv", index=False)
    C.to_csv(RESULTS / "training_checkpoints.csv", index=False)
    ps = per_seed(met)
    ps.to_csv(RESULTS / "per_seed_results.csv", index=False)
    ys = Y.groupby(["env", "label", "year"]).agg(
        sim_median_mean=("median", "mean"), sim_median_p10=("median", lambda v: v.quantile(.1)),
        sim_median_p90=("median", lambda v: v.quantile(.9)), sim_p25=("p25", "mean"), sim_p75=("p75", "mean"),
        sim_ratio=("ratio", "mean"), sim_sale_to_ask=("sale_to_ask", "mean"), sim_sales=("sales", "mean"),
        obs_median=("obs_median", "first"), obs_p25=("obs_p25", "first"), obs_p75=("obs_p75", "first"),
        obs_ratio=("obs_ratio", "first"), obs_n_sales=("obs_n_sales", "first"), runs=("median", "size"),
    ).reset_index()
    ys["period"] = np.select([ys.year.isin(TEST_YEARS), ys.year.isin(VAL_YEARS), ys.year == 2010],
                             ["test", "validation", "initial condition"], "train")
    ys.to_csv(RESULTS / "yearly_sim_vs_obs.csv", index=False)
    fm = []
    for (env, lab, var), p in ps.groupby(["env", "label", "variant"]):
        for col in [c for c in p.columns if c.split("_")[0] in ("train", "val", "test", "all")]:
            s = summarize(p[col])
            fm.append({"env": env, "label": lab, "variant": var, "metric": col, **s})
    pd.DataFrame(fm).to_csv(RESULTS / "final_metrics.csv", index=False)

    # ---- configuration with every final value
    cfg = {"market": json.loads((RESULTS / "calibration" / "market_params.json").read_text()),
           "reward": asdict_all(RewardParams())["RewardParams"], "spec": asdict_all(Spec())["Spec"],
           "qlearning": best["qlearning"], "dqn": {**asdict_all(DQNParams())["DQNParams"], **best["dqn"]},
           "training": {**asdict_all(TrainParams())["TrainParams"], "episodes_qlearning": 400, "episodes_dqn": 250},
           "validation_score": asdict_all(ValScore())["ValScore"],
           "split": {"initial_condition": 2010, "train_reward_years": TRAIN_YEARS, "train_fit_years": list(range(2011, 2019)),
                     "validation": VAL_YEARS, "test": TEST_YEARS},
           "seeds": {"train": TRAIN_SEEDS, "checkpoint_eval": CHECKPOINT_EVAL_SEEDS, "final_eval": FINAL_EVAL_SEEDS,
                     "calibration": CALIB_SEEDS},
           "calibration": {"grid": GRID, "tolerance": TOLERANCE}}
    (RESULTS / "config_final.json").write_text(json.dumps(cfg, indent=2, default=str))

    # ---- figures
    fig_training(E, C, met, conv)
    fig_model_detail(E, C, "qlearning")
    fig_model_detail(E, C, "dqn")
    fig_validation(C, met)
    fig_calibration(Y, bench, "shared", "Rule-Based-RL-DRL-Calibration-2010-2024.png")
    if (Y.env == "recalibrated").any():
        fig_calibration(Y, bench, "recalibrated", "Rule-Based-RL-DRL-Recalibrated-2010-2024.png")
    fig_residuals(Y)
    t5, abl = table_ablation(met, runs)
    fig_ablation(abl)
    fig_sanity(dev)

    # ---- tables
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table_hyper(best, None, None).to_string())
    print(table_convergence(runs, E, C, ps, conv).to_string())
    print(table_performance(ps, bench_m).to_string())
    print(table_vs_rule(met).round(3).to_string())
    print(t5.to_string())
    sc = pd.read_csv(EV / "sanity_checks.csv")
    latex(sc, TAB / "table6_sanity_checks")
    print(sc.to_string())
    dsum = dev.groupby(["model", "population", "deviator"])[["deviator_reward", "population_reward"]].mean()
    dsum.round(4).to_csv(RESULTS / "deviation_test_summary.csv")
    print(dsum.round(3).to_string())


if __name__ == "__main__":
    main()
