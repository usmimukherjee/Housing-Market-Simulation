"""Shock responsiveness, recovery and stability of the rule-based, Q-learning and DQN markets.

Analysis only. No simulation is run, no policy is trained or changed, and no file written by the
earlier steps is modified. The input is the per-run yearly output of step 10
(``results/results-scenario/shocks/shock_paths.csv``), produced as follows (see
``src/src-scenario/market_shocks.py`` and ``housing_abm/shocks.py``):

* the shared calibrated market of step 1, unchanged;
* frozen policies: the rule-based policy and the converged end-of-training Q-learning and DQN
  policies (10 training seeds each), evaluated with epsilon = 0 and no learning;
* 20 market seeds (1000-1019), shared by every model and scenario;
* scenarios: no shock; supply shock (new listings -50%); demand shock (new buyers +50%), both in
  2022 and 2023 only, after every year used for calibration, training and model selection
  (2010-2021). Each shocked run is identical to its paired no-shock run (same model, training seed
  and market seed) until 2022; the script checks this.

Price measure: constant-quality (quality-adjusted) median = median of sale price / dwelling quality.

Experiment 1 (price response): deviation_y = (P_shock,y / P_noshock,y - 1) x 100 for each paired run.
Experiment 2 (recovery): per run, peak deviation over 2022-2024 (largest absolute value, sign kept),
  deviation in 2024, and the reduction in absolute deviation after the shock ends,
  |dev_2023| - |dev_2024| (pp). 2024 is the only post-shock year in the main horizon. The step-10
  runs continue to 2028 with no further shock; those years are reported separately as an extended
  horizon (simulation only, no observed data involved).
Experiment 3 (stability): SD (ddof=1) of annual constant-quality price growth within each run,
  before the shock (growth in 2019-2021) and during/after it (growth in 2022-2024); 3 growth rates each.

Uncertainty: 95% percentile bootstrap (4,000 resamples). Learners: market seeds and training seeds
resampled independently (two-way); rule-based: market seeds. Model differences are paired by
market seed.

Usage (from the package root): python src/src-scenario/shock_response_analysis.py
Writes results/results-scenario/ and graphs/shocks/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
SRC = PKG / "results" / "results-scenario" / "shocks" / "shock_paths.csv"
OUT = PKG / "results" / "results-scenario"
GRAPHS = PKG / "graphs" / "shocks"

MODELS = ["Rule-based", "Q-learning", "DQN"]
SCEN = {"supply": "Supply shock (new listings $-$50%)", "demand": "Demand shock (new buyers +50%)"}
COLORS = {"Rule-based": "#2a78d6", "Q-learning": "#eb6834", "DQN": "#1baf7a"}
MARKERS = {"Rule-based": "s", "Q-learning": "^", "DQN": "D"}
LINES = {"Rule-based": (0, (5, 2)), "Q-learning": (0, (5, 1.5, 1, 1.5)), "DQN": "-"}
SHOCK = (2022, 2023)
POST = 2024
B = 4000
RNG = np.random.default_rng(20251106)


# ----------------------------------------------------------------------------- data
def load_paths() -> pd.DataFrame:
    P = pd.read_csv(SRC)
    P = P[P["window"].isin(["none", "2022-23"]) & P["scenario"].isin(["none", *SCEN])
          & P["policy"].isin(MODELS)].copy()
    P["train_seed"] = P["train_seed"].fillna(-1).astype(int)
    P = P.sort_values(["policy", "train_seed", "market_seed", "scenario", "year"])
    P["growth"] = P.groupby(["policy", "train_seed", "market_seed", "scenario"])["cq_median"].pct_change() * 100
    return P


def paired(P: pd.DataFrame) -> pd.DataFrame:
    key = ["policy", "train_seed", "market_seed", "year"]
    base = P[P.scenario == "none"].set_index(key)[["cq_median", "growth"]]
    out = []
    for s in SCEN:
        sh = P[P.scenario == s].set_index(key)[["cq_median", "growth"]]
        d = sh.join(base, lsuffix="_shock", rsuffix="_base", how="inner").reset_index()
        d["scenario"] = s
        d["dev_pct"] = (d.cq_median_shock / d.cq_median_base - 1) * 100
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- run metrics
def run_metrics(D: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pol, ts, ms, s), g in D.groupby(["policy", "train_seed", "market_seed", "scenario"]):
        dev = g.set_index("year")["dev_pct"]
        win = dev.loc[SHOCK[0]:POST]
        py = win.abs().idxmax()
        gs = g.set_index("year")["growth_shock"]
        gb = g.set_index("year")["growth_base"]
        rows.append({
            "policy": pol, "train_seed": ts, "market_seed": ms, "scenario": s,
            "pre_shock_max_abs_dev": dev.loc[:SHOCK[0] - 1].abs().max(),
            "dev_2022": dev[2022], "dev_2023": dev[2023], "dev_2024": dev[2024],
            "response_2022_23": dev.loc[SHOCK[0]:SHOCK[1]].mean(),
            "expected_sign_both_years": bool((dev.loc[SHOCK[0]:SHOCK[1]] > 0).all()),
            "peak_dev": dev[py], "peak_year": py,
            "post_reduction_pp": abs(dev[2023]) - abs(dev[2024]),
            "deviation_shrinks_2024": bool(abs(dev[2024]) < abs(dev[2023])),
            "vol_pre_2019_21": gs.loc[2019:2021].std(ddof=1),
            "vol_shock_2022_24": gs.loc[2022:2024].std(ddof=1),
            "vol_noshock_2022_24": gb.loc[2022:2024].std(ddof=1),
            "later_movement_pp": abs(dev[2024] - dev[2023]),
            # extended horizon (no further shock; simulation only)
            "dev_2028": dev.get(2028, np.nan),
            "post_reduction_2028_pp": abs(dev[2023]) - abs(dev.get(2028, np.nan)),
            "peak_dev_2022_28": dev.loc[SHOCK[0]:].loc[lambda x: x.abs().idxmax()],
        })
    R = pd.DataFrame(rows)
    # no-shock scenario volatility for Figure 2
    nb = P[P.scenario == "none"].groupby(["policy", "train_seed", "market_seed"])
    N = nb.apply(lambda g: pd.Series({
        "vol_pre_2019_21": g.set_index("year").growth.loc[2019:2021].std(ddof=1),
        "vol_shock_2022_24": g.set_index("year").growth.loc[2022:2024].std(ddof=1)}),
        include_groups=False).reset_index()
    N["scenario"] = "none"
    return pd.concat([R, N], ignore_index=True)


# ----------------------------------------------------------------------------- bootstrap
def _grid(df, col):
    """Matrix [training seed x market seed] of a run-level metric."""
    return df.pivot_table(index="train_seed", columns="market_seed", values=col, aggfunc="mean")


def boot_mean(df, col, stat=np.mean):
    M = _grid(df, col).to_numpy()
    nt, nm = M.shape
    est = stat(M)
    bs = np.empty(B)
    for b in range(B):
        ti = RNG.integers(0, nt, nt) if nt > 1 else np.zeros(1, int)
        mi = RNG.integers(0, nm, nm)
        bs[b] = stat(M[np.ix_(ti, mi)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def boot_diff(a, b, col):
    """Model a minus model b, paired by market seed (each model's training seeds resampled)."""
    A, Bm = _grid(a, col), _grid(b, col)
    ms = A.columns.intersection(Bm.columns)
    A, Bm = A[ms].to_numpy(), Bm[ms].to_numpy()
    est = A.mean() - Bm.mean()
    bs = np.empty(B)
    for i in range(B):
        mi = RNG.integers(0, len(ms), len(ms))
        ta = RNG.integers(0, A.shape[0], A.shape[0])
        tb = RNG.integers(0, Bm.shape[0], Bm.shape[0])
        bs[i] = A[np.ix_(ta, mi)].mean() - Bm[np.ix_(tb, mi)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


METRICS = [("response_2022_23", "Mean deviation 2022-2023 (%)"),
           ("peak_dev", "Peak deviation 2022-2024 (%)"),
           ("dev_2024", "Deviation in 2024 (%)"),
           ("post_reduction_pp", "Reduction in |deviation| 2023 to 2024 (pp)"),
           ("vol_pre_2019_21", "Volatility before shock, 2019-2021 (pp)"),
           ("vol_shock_2022_24", "Volatility 2022-2024 (pp)"),
           ("dev_2028", "Deviation in 2028, extended horizon (%)"),
           ("post_reduction_2028_pp", "Reduction in |deviation| 2023 to 2028, extended (pp)")]


def summary(R):
    rows = []
    for s in ["none", *SCEN]:
        for pol in MODELS:
            g = R[(R.scenario == s) & (R.policy == pol)]
            row = {"scenario": s, "policy": pol, "n_runs": len(g),
                   "n_training_seeds": g.train_seed.nunique() if pol != "Rule-based" else 0,
                   "n_market_seeds": g.market_seed.nunique()}
            for col, _ in METRICS:
                if g[col].notna().any():
                    est, lo, hi = boot_mean(g, col)
                    row.update({col: est, f"{col}_lo": lo, f"{col}_hi": hi,
                                f"{col}_sd_runs": g[col].std(ddof=1)})
            if s != "none":
                row["share_expected_sign_pct"] = 100 * g.expected_sign_both_years.mean()
                row["share_deviation_shrinks_2024_pct"] = 100 * g.deviation_shrinks_2024.mean()
                row["share_peak_in_2024_pct"] = 100 * (g.peak_year == 2024).mean()
                row["max_pre_shock_abs_dev"] = g.pre_shock_max_abs_dev.max()
                row["response_cv_across_runs_pct"] = 100 * g.response_2022_23.std(ddof=1) / abs(g.response_2022_23.mean())
                # initial adjustment vs later movement (Spearman, across runs)
                row["spearman_initial_vs_later"] = g.dev_2022.abs().corr(g.later_movement_pp, method="spearman")
                # variance split for learners: across training seeds vs across market seeds
                if pol != "Rule-based":
                    row["sd_across_training_seeds"] = g.groupby("train_seed").response_2022_23.mean().std(ddof=1)
                row["sd_across_market_seeds"] = g.groupby("market_seed").response_2022_23.mean().std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def comparisons(R):
    rows = []
    pairs = [("DQN", "Q-learning"), ("DQN", "Rule-based"), ("Q-learning", "Rule-based")]
    for s in ["none", *SCEN]:
        for a, b in pairs:
            for col, lab in METRICS:
                if s == "none" and col not in ("vol_pre_2019_21", "vol_shock_2022_24"):
                    continue
                ga = R[(R.scenario == s) & (R.policy == a)]
                gb = R[(R.scenario == s) & (R.policy == b)]
                est, lo, hi = boot_diff(ga, gb, col)
                rows.append({"scenario": s, "metric": col, "description": lab, "model_a": a, "model_b": b,
                             "diff_a_minus_b": est, "ci_lo": lo, "ci_hi": hi,
                             "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def dev_paths(D):
    rows = []
    for (s, pol, y), g in D.groupby(["scenario", "policy", "year"]):
        est, lo, hi = boot_mean(g, "dev_pct")
        rows.append({"scenario": s, "policy": pol, "year": y, "dev_mean": est, "ci_lo": lo, "ci_hi": hi,
                     "p10": g.dev_pct.quantile(.1), "p90": g.dev_pct.quantile(.9)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    sns.set_theme(context="paper", style="ticks", font="serif",
                  rc={"pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8,
                      "xtick.major.width": 0.8, "ytick.major.width": 0.8})


def figure1(Pth, path, years, legend_loc="upper left"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    dx = {"Rule-based": -0.08, "Q-learning": 0.0, "DQN": 0.08}
    for ax, (s, title) in zip(axes, SCEN.items()):
        ax.axvspan(SHOCK[0] - 0.5, SHOCK[1] + 0.5, color="0.92", lw=0, zorder=0)
        ax.axhline(0, color="0.35", lw=0.7, zorder=1)
        for pol in MODELS:
            g = Pth[(Pth.scenario == s) & (Pth.policy == pol) & Pth.year.between(*years)]
            x = g.year + dx[pol]
            ax.fill_between(x, g.ci_lo, g.ci_hi, color=COLORS[pol], alpha=0.22, lw=0, zorder=2)
            ax.plot(x, g.dev_mean, color=COLORS[pol], ls=LINES[pol], lw=1.5, marker=MARKERS[pol], ms=4.5,
                    mec="white", mew=0.5, label=pol, zorder=3)
        ax.set_title(title.replace("$-$", "−"), fontsize=9, loc="left")
        ax.set_xticks(range(years[0], years[1] + 1))
        ax.tick_params(labelsize=8, axis="x", rotation=45 if years[1] - years[0] > 6 else 0)
        ax.tick_params(labelsize=8, axis="y")
        ax.set_xlabel("Year", fontsize=9)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Quality-adjusted median price,\n% deviation from no-shock run", fontsize=9)
    ymax = max(a.get_ylim()[1] for a in axes)
    for ax in axes:
        ax.set_ylim(top=ymax * 1.08)
        ax.text(np.mean(SHOCK), ymax * 1.02, "Shock 2022–2023", ha="center", va="top", fontsize=7.5, color="0.3")
    axes[0].legend(loc=legend_loc, fontsize=7.5, frameon=False, handlelength=3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_single(Pth, years, stem):
    """One untitled figure per shock (same y-range for both), e.g. <stem>-Supply-2019-2028.pdf."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    _style()
    dx = {"Rule-based": -0.08, "Q-learning": 0.0, "DQN": 0.08}
    sel = Pth[Pth.year.between(*years)].copy()
    sel["x"] = sel.year + sel.policy.map(dx)          # small horizontal dodge so markers don't overlap
    sel = sel.rename(columns={"policy": "Model"})
    ylo = min(0.0, sel.ci_lo.min()) - 0.5
    yhi = sel.ci_hi.max() * 1.1
    dashes = {"Rule-based": (5, 2), "Q-learning": (5, 1.5, 1, 1.5), "DQN": ""}
    for s in SCEN:
        g = sel[sel.scenario == s]
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.axvspan(SHOCK[0] - 0.5, SHOCK[1] + 0.5, color="0.92", lw=0, zorder=0)
        ax.axhline(0, color="0.35", lw=0.7, zorder=1)
        # 95% two-way bootstrap intervals (training x market seeds), same as Table 1
        for pol in MODELS:
            h = g[g.Model == pol]
            ax.fill_between(h.x, h.ci_lo, h.ci_hi, color=COLORS[pol], alpha=0.22, lw=0, zorder=2)
        sns.lineplot(data=g, x="x", y="dev_mean", hue="Model", style="Model", hue_order=MODELS,
                     style_order=MODELS, palette=COLORS, dashes=dashes, markers=MARKERS, markersize=5,
                     markeredgecolor="white", markeredgewidth=0.5, linewidth=1.5, errorbar=None,
                     ax=ax, zorder=3)
        ax.set_ylim(ylo, yhi)
        ax.text(np.mean(SHOCK), yhi * 0.97, "Shock", ha="center", va="top", fontsize=7.5, color="0.3")
        ax.set_xticks(range(years[0], years[1] + 1))
        ax.tick_params(labelsize=8, axis="x", rotation=45 if years[1] - years[0] > 6 else 0)
        ax.tick_params(labelsize=8, axis="y")
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Price deviation (%)", fontsize=9)
        ax.legend(title=None, loc="lower right", fontsize=7.5, frameon=False, handlelength=3)
        sns.despine(ax=ax)
        fig.tight_layout()
        path = GRAPHS / f"{stem}-{s.capitalize()}-{years[0]}-{years[1]}.pdf"
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def figure2(R, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    _style()
    d = R[["scenario", "policy", "vol_shock_2022_24"]].copy()
    d["Scenario"] = d.scenario.map({"none": "No shock", "supply": "Supply shock\n(listings −50%)",
                                    "demand": "Demand shock\n(buyers +50%)"})
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    sns.boxplot(data=d, x="Scenario", y="vol_shock_2022_24", hue="policy", hue_order=MODELS,
                order=["No shock", "Supply shock\n(listings −50%)", "Demand shock\n(buyers +50%)"],
                palette=COLORS, width=0.7, gap=0.15, linewidth=0.8, fliersize=2,
                medianprops={"color": "white", "lw": 1.4}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("SD of annual quality-adjusted\nprice growth, 2022–2024 (pp)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(title=None, fontsize=7.5, frameon=False, loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- table
def table1(S):
    f = lambda r, c, d=1: f"{r[c]:.{d}f} [{r[c + '_lo']:.{d}f}, {r[c + '_hi']:.{d}f}]"
    rows = []
    for s in SCEN:
        for pol in MODELS:
            r = S[(S.scenario == s) & (S.policy == pol)].iloc[0]
            rows.append({"Model": pol, "Shock": s.capitalize(),
                         "Expected sign 2022-23 (% runs)": f"{r.share_expected_sign_pct:.0f}",
                         "Peak deviation (%)": f(r, "peak_dev"),
                         "Deviation 2024 (%)": f(r, "dev_2024"),
                         "Reduction in |dev.| 2023-24 (pp)": f(r, "post_reduction_pp", 2),
                         "Volatility before (pp)": f(r, "vol_pre_2019_21", 2),
                         "Volatility 2022-24 (pp)": f(r, "vol_shock_2022_24", 2)})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table1_shock_response.csv", index=False)
    tex = [r"\begin{tabular}{llrlllll}", r"\toprule",
           r"Shock & Model & Exp. sign & Peak dev. (\%) & Dev. 2024 (\%) & $\Delta|\mathrm{dev}|$ 2023--24 (pp) & Vol. 2019--21 (pp) & Vol. 2022--24 (pp) \\",
           r"\midrule"]
    for i, r in T.iterrows():
        tex.append(" & ".join([r.Shock if i % 3 == 0 else "", r.Model,
                               r["Expected sign 2022-23 (% runs)"] + r"\%", r["Peak deviation (%)"],
                               r["Deviation 2024 (%)"], r["Reduction in |dev.| 2023-24 (pp)"],
                               r["Volatility before (pp)"], r["Volatility 2022-24 (pp)"]]) + r" \\")
        if i == 2:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table1_shock_response.tex").write_text("\n".join(tex) + "\n")
    return T


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GRAPHS.mkdir(parents=True, exist_ok=True)
    P = load_paths()
    D = paired(P)
    n_pairs = D.groupby(["scenario", "policy"]).apply(lambda g: g[["train_seed", "market_seed"]]
                                                      .drop_duplicates().shape[0], include_groups=False)
    print("paired runs:\n", n_pairs.to_string())
    pre = D[D.year < SHOCK[0]].dev_pct.abs().max()
    print(f"max |deviation| before 2022 over all pairs: {pre:.3e}%")
    if pre > 1e-9:
        raise SystemExit("Shocked and no-shock runs differ before the shock; stopping.")

    D.to_csv(OUT / "shock_response_paired_paths.csv", index=False)
    R = run_metrics(D, P)
    R.to_csv(OUT / "shock_response_runs.csv", index=False)
    S = summary(R)
    S.to_csv(OUT / "shock_response_summary.csv", index=False)
    C = comparisons(R)
    C.to_csv(OUT / "shock_response_model_differences.csv", index=False)
    Pth = dev_paths(D)
    Pth.to_csv(OUT / "shock_response_deviation_by_year.csv", index=False)
    T = table1(S)

    figure1(Pth, GRAPHS / "Shock-Response-Recovery-2019-2024.pdf", (2019, 2024))
    figure1(Pth, GRAPHS / "Shock-Response-Recovery-Extended-2019-2028.pdf", (2019, 2028),
            legend_loc="lower right")
    figure_single(Pth, (2019, 2028), "Shock-Response")
    figure2(R, GRAPHS / "Shock-Volatility-Comparison-2022-2024.pdf")

    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    cols = ["scenario", "policy", "response_2022_23", "response_2022_23_lo", "response_2022_23_hi",
            "share_expected_sign_pct", "response_cv_across_runs_pct", "share_deviation_shrinks_2024_pct",
            "share_peak_in_2024_pct", "spearman_initial_vs_later", "sd_across_training_seeds",
            "sd_across_market_seeds", "dev_2028", "post_reduction_2028_pp"]
    print(S[[c for c in cols if c in S]].round(3).to_string(index=False))
    print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
