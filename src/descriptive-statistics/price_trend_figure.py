"""Figure 2 of Chapter 2: mean assessed value and mean sale price in HRM, 2010-2024.

Same steps as cells 20-28 of ``validation/descriptive-analysis.ipynb`` in the main repository,
which produced the figure used in the chapter (``assessed_value_sale_price_per_year.png``):

1. Read the yearly HRM sales (``HRM_sale_data_yearly.csv``) and the assessment history
   (``Assessed_Value_and_Taxable_Assessed_Value_History.csv``).
2. Keep assessment account numbers present in both files.
3. Attach assessed values to sales by account number (no tax-year match, as in the notebook).
4. Mean assessed value and mean sale price per sale year, years before 2025.

Writes graphs/assessed_value_sale_price_per_year.png and
results/descriptive-statistics/assessed_value_sale_price_per_year.csv.

Usage (from the package root):
    python src/descriptive-statistics/price_trend_figure.py [--raw-dir ../validation]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT.parent / "validation"
ACC = "Assessment Account Number"


def yearly_means(raw_dir: Path) -> pd.DataFrame:
    df_hrm = pd.read_csv(raw_dir / "HRM_sale_data_yearly.csv", low_memory=False)
    assessed_df = pd.read_csv(raw_dir / "Assessed_Value_and_Taxable_Assessed_Value_History.csv",
                              usecols=[ACC, "Assessed Value"], low_memory=False)
    common = set(df_hrm[ACC]).intersection(set(assessed_df[ACC]))
    df_hrm_common = df_hrm[df_hrm[ACC].isin(common)]
    assessed_common = assessed_df[assessed_df[ACC].isin(common)]
    merged = df_hrm_common.merge(assessed_common[[ACC, "Assessed Value"]], on=ACC, how="left")
    out = (merged.groupby("Year").agg({"Assessed Value": "mean", "Sale Price": "mean"}).reset_index()
           .rename(columns={"Assessed Value": "Mean Assessed Value", "Sale Price": "Mean Sale Price"}))
    return out[out["Year"] < 2025]


def plot(t: pd.DataFrame, path: Path) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("husl", 2)
    ax = sns.lineplot(
        data=t.melt(id_vars="Year", value_vars=["Mean Assessed Value", "Mean Sale Price"],
                    var_name="Metric", value_name="Value"),
        x="Year", y="Value", hue="Metric", style="Metric", markers=["o", "s"], dashes=[(2, 2), ""],
        linewidth=2.5, markersize=8, palette=palette)
    plt.title("Assessed Price and Sale Prices in HRM (2010-2024)", fontsize=15, pad=20, fontweight="bold")
    plt.xticks(rotation=45)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    last_year = t["Year"].iloc[-1]
    for metric in ["Mean Assessed Value", "Mean Sale Price"]:
        value = t[t["Year"] == last_year][metric].values[0]
        ax.text(last_year, value, f"${value:,.0f}",
                color=palette[0] if metric == "Mean Assessed Value" else palette[1],
                fontsize=10, ha="left", va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))
    plt.fill_between(t["Year"], t["Mean Assessed Value"], t["Mean Sale Price"],
                     color="gray", alpha=0.1, label="Assessment Gap")
    plt.legend(title="Metrics", fontsize=10, title_fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Price (CAD $)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    args = ap.parse_args()
    t = yearly_means(args.raw_dir)
    (ROOT / "results" / "descriptive-statistics").mkdir(parents=True, exist_ok=True)
    (ROOT / "graphs").mkdir(parents=True, exist_ok=True)
    t.to_csv(ROOT / "results" / "descriptive-statistics" / "assessed_value_sale_price_per_year.csv", index=False)
    plot(t, ROOT / "graphs" / "assessed_value_sale_price_per_year.png")
    print(t.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
