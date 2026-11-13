"""Annual shock-response figures and numbers for the revised shock design (2021-2028).

Supply shock: new listings +50% in 2022-2023 (step-10 scenario ``supply_up``); theory: lower prices.
Demand shock: new buyers   +50% in 2022-2023 (step-10 scenario ``demand``);    theory: higher prices.

Input: results/results-scenario/shocks/shock_paths.csv (step 10,
already run: shared calibrated market, frozen policies, 20 market seeds, 10 training seeds per
learner). No simulation is run here. Pairing, price measure (constant-quality median), two-way
bootstrap and figure style are those of src/src-scenario/shock_response_analysis.py, which is
imported unchanged.

Writes graphs/shocks/Shock-Response-{Supply,Demand}-2021-2028.{pdf,png} (Figures 7 and 8
of Chapter 2) and results/results-scenario/annual_deviation_by_year.csv, annual_summary.csv,
annual_model_differences.csv.

Usage (from the package root): python src/src-scenario/annual_shocks.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
OUT = PKG / "results" / "results-scenario"
GRAPHS = PKG / "graphs" / "shocks"

_s = importlib.util.spec_from_file_location("sra", PKG / "src" / "src-scenario" / "shock_response_analysis.py")
sra = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sra)

STEP10 = {"supply_up": "supply", "demand": "demand"}   # step-10 name -> name used here
EXPECTED = {"supply": -1, "demand": 1}


def figure(Pth, years):
    """Same visual style as the quarterly figures (plot_quarterly_shocks.py), one value per year."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from matplotlib.lines import Line2D
    sra._style()
    for s, name in [("supply", "Supply"), ("demand", "Demand")]:
        g = Pth[(Pth.scenario == s) & Pth.year.between(*years)]
        lo, hi = min(0.0, g.ci_lo.min()), max(0.0, g.ci_hi.max())
        pad = 0.12 * (hi - lo)
        ylo, yhi = np.floor(lo - pad), np.ceil(hi + pad)
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        ax.axvspan(2021.5, 2023.5, color="0.93", lw=0, zorder=0)
        ax.yaxis.grid(True, color="0.88", lw=0.6, zorder=0.4)
        ax.axhline(0, color="0.4", lw=0.8, zorder=1)
        for pol in sra.MODELS:
            h = g[g.policy == pol]
            ax.fill_between(h.year, h.ci_lo, h.ci_hi, color=sra.COLORS[pol], alpha=0.14, lw=0, zorder=2)
            ax.plot(h.year, h.dev_mean, color=sra.COLORS[pol], ls=sra.LINES[pol], lw=1.7, zorder=3)
            ax.plot(h.year, h.dev_mean, ls="none", marker=sra.MARKERS[pol], ms=4.8, color=sra.COLORS[pol],
                    mec="white", mew=0.7, zorder=4)
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(years[0] - 0.4, years[1] + 0.4)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.text(max(2021.58, years[0] - 0.33), yhi - 0.02 * (yhi - ylo), "Shock period", ha="left", va="top", fontsize=7.5,
                color="0.35", style="italic")
        ax.set_xticks(range(years[0], years[1] + 1))
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Price deviation (%)", fontsize=9)
        handles = [Line2D([], [], color=sra.COLORS[p], ls=sra.LINES[p], lw=1.7, marker=sra.MARKERS[p], ms=4.8,
                          mec="white", mew=0.7) for p in sra.MODELS]
        ax.legend(handles, sra.MODELS, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8,
                  frameon=False, handlelength=3.2, columnspacing=1.6)
        sns.despine(ax=ax)
        fig.tight_layout()
        p = GRAPHS / f"Shock-Response-{name}-{years[0]}-{years[1]}"
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GRAPHS.mkdir(parents=True, exist_ok=True)
    sra.SCEN = {k: v for k, v in zip(STEP10, ["Supply shock (new listings +50%)", "Demand shock (new buyers +50%)"])}
    sra.GRAPHS = GRAPHS
    P = sra.load_paths()
    D = sra.paired(P)
    pre = D[D.year < 2022].dev_pct.abs().max()
    assert pre < 1e-9, f"shocked and no-shock runs differ before 2022: {pre}"
    D["scenario"] = D.scenario.map(STEP10)
    P["scenario"] = P.scenario.map(lambda s: STEP10.get(s, s))
    sra.SCEN = {"supply": "Supply shock (new listings +50%)", "demand": "Demand shock (new buyers +50%)"}

    Pth = sra.dev_paths(D)
    Pth.to_csv(OUT / "annual_deviation_by_year.csv", index=False)
    figure(Pth, (2021, 2028))

    R = sra.run_metrics(D, P)
    R = R[R.scenario != "none"].copy()
    # expected direction depends on the shock (the original helper assumes a price increase)
    R["expected_sign_both_years"] = [
        bool((np.sign(r[["dev_2022", "dev_2023"]].astype(float)) == EXPECTED[r.scenario]).all())
        for _, r in R.iterrows()]
    rows = []
    for s in STEP10.values():
        for pol in sra.MODELS:
            g = R[(R.scenario == s) & (R.policy == pol)]
            row = {"scenario": s, "policy": pol, "n_runs": len(g)}
            for c in ["dev_2022", "dev_2023", "response_2022_23", "dev_2024", "dev_2028"]:
                est, lo, hi = sra.boot_mean(g, c)
                row.update({c: est, f"{c}_lo": lo, f"{c}_hi": hi})
            row["share_expected_sign_pct"] = 100 * g.expected_sign_both_years.mean()
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "annual_summary.csv", index=False)

    rows = []
    for s in STEP10.values():
        for a, b in [("DQN", "Q-learning"), ("DQN", "Rule-based"), ("Q-learning", "Rule-based")]:
            for c in ["response_2022_23", "dev_2024", "dev_2028"]:
                ga = R[(R.scenario == s) & (R.policy == a)]
                gb = R[(R.scenario == s) & (R.policy == b)]
                est, lo, hi = sra.boot_diff(ga, gb, c)
                rows.append({"scenario": s, "metric": c, "model_a": a, "model_b": b, "diff": est,
                             "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "annual_model_differences.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["scenario", "policy", "response_2022_23", "response_2022_23_lo", "response_2022_23_hi",
            "dev_2022", "dev_2023", "dev_2024", "dev_2028", "share_expected_sign_pct"]
    print(S[cols].round(2).to_string(index=False))
    print(C.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
