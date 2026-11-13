"""Build the observed targets and model inputs used by the convergence experiments.

Reads the raw Halifax files in the main repository's ``validation/`` folder and the HMTS
survey worksheet, and writes small derived files to ``inputs/derived/``:

* ``observed_annual.csv``   cleaned annual sale-price statistics, 2010-2024
* ``observed_ratio.csv``    assessed value / sale price ratio, same parcel and tax year, 2019-2023
* ``listing_pool.csv``      dwellings that can be listed in the simulation, with relative quality
* ``buyers.csv``            survey households classified as movers, with WTP in dollars
* ``record_counts.csv``     records kept at each cleaning step

Cleaning rules for sales (each one fixes a problem in the submitted analysis):

* Sale price below $10,000 removed. Zero prices are non-market transfers (29% of raw rows).
* Multi-parcel sales removed. Their full price is repeated on every parcel.
* Exact duplicate (account, date, price) rows removed.

Usage (from the package root)::

    python src/data-preparation/prepare_observed_data.py [--raw-dir ../validation]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT.parent / "validation"
DEFAULT_SURVEY = ROOT / "inputs" / "HMTS Data_WTP.xlsx"
OUT = ROOT / "inputs" / "derived"
YEARS = range(2010, 2025)
AV_YEARS = [2019, 2020, 2021, 2022, 2023]
MIN_PRICE = 10_000


def clean_sales(raw_dir: Path, counts: list) -> pd.DataFrame:
    s = pd.read_csv(raw_dir / "HRM_sale_data_yearly.csv", low_memory=False,
                    usecols=["Assessment Account Number", "Sale Price", "Sale Date",
                             "Parcels In Sale", "Year"])
    counts.append(("Raw sale records", len(s)))
    s = s[s["Year"].isin(YEARS)]
    counts.append(("Sale year 2010-2024", len(s)))
    s = s[s["Sale Price"] >= MIN_PRICE]
    counts.append((f"Sale price >= ${MIN_PRICE:,}", len(s)))
    s = s[s["Parcels In Sale"] == 1]
    counts.append(("Single-parcel sales", len(s)))
    s = s.drop_duplicates(["Assessment Account Number", "Sale Date", "Sale Price"])
    counts.append(("After removing duplicates", len(s)))
    return s.rename(columns={"Assessment Account Number": "account", "Sale Price": "price",
                             "Year": "year"})[["account", "price", "year"]]


def annual(s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby("year")["price"]
    out = pd.DataFrame({
        "n_sales": g.size(),
        "p25": g.quantile(0.25),
        "median": g.median(),
        "p75": g.quantile(0.75),
        "mean": g.mean(),
    })
    return out.reset_index()


def assessments(raw_dir: Path) -> pd.DataFrame:
    a = pd.read_csv(raw_dir / "Assessed_Value_and_Taxable_Assessed_Value_History.csv",
                    usecols=["Assessment Account Number", "Tax Year", "Assessed Value"])
    a = a.rename(columns={"Assessment Account Number": "account", "Tax Year": "tax_year",
                          "Assessed Value": "av"})
    a = a.drop_duplicates(["account", "tax_year"])
    wide = a.pivot(index="account", columns="tax_year", values="av")
    wide = wide[AV_YEARS]
    wide.columns = [f"av_{y}" for y in AV_YEARS]
    return wide


def ratio(s: pd.DataFrame, av: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in AV_YEARS:
        sy = s[s["year"] == y].merge(av[[f"av_{y}"]], left_on="account", right_index=True)
        sy = sy[sy[f"av_{y}"] > 0]
        r = sy[f"av_{y}"] / sy["price"]
        rows.append({"year": y, "n_matched": len(r), "median_ratio": r.median(),
                     "mean_ratio": r.mean()})
    return pd.DataFrame(rows)


def listing_pool(s: pd.DataFrame, av: pd.DataFrame, counts: list) -> pd.DataFrame:
    accounts = s["account"].unique()
    pool = av.loc[av.index.intersection(accounts)]
    counts.append(("Parcels with a clean sale and assessments", len(pool)))
    pool = pool[(pool > 0).all(axis=1)]
    counts.append(("... with five positive assessments", len(pool)))
    lo, hi = pool["av_2019"].quantile([0.01, 0.99])
    pool = pool[(pool["av_2019"] >= lo) & (pool["av_2019"] <= hi)]
    counts.append(("... after trimming 1st/99th percentile of 2019 assessment", len(pool)))
    pool = pool.copy()
    pool["quality"] = pool["av_2019"] / pool["av_2019"].median()
    return pool.reset_index()


def buyers(survey: Path) -> pd.DataFrame:
    df = pd.read_excel(survey, sheet_name="Copy_Of_Query1")
    df.columns = [c.strip() for c in df.columns]
    wtp = df["WTP (in 1000)"]
    # Worksheet rule: mover when WTP > 1. Values are in dollars despite the column label
    # (median of movers is about $168,669).
    movers = df.loc[wtp > 1, ["ID Number"]].copy()
    movers["wtp"] = wtp[wtp > 1].values
    return movers.rename(columns={"ID Number": "survey_id"}).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--survey", type=Path, default=DEFAULT_SURVEY)
    ap.add_argument("--outdir", type=Path, default=OUT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    counts: list = []
    s = clean_sales(args.raw_dir, counts)
    av = assessments(args.raw_dir)
    obs = annual(s)
    rat = ratio(s, av)
    pool = listing_pool(s, av, counts)
    b = buyers(args.survey)
    counts.append(("Survey households", 313))
    counts.append(("Survey households classified as movers (buyers)", len(b)))

    obs.to_csv(args.outdir / "observed_annual.csv", index=False)
    rat.to_csv(args.outdir / "observed_ratio.csv", index=False)
    pool.to_csv(args.outdir / "listing_pool.csv", index=False)
    b.to_csv(args.outdir / "buyers.csv", index=False)
    pd.DataFrame(counts, columns=["step", "records"]).to_csv(
        args.outdir / "record_counts.csv", index=False)

    print(obs.round(0).to_string(index=False))
    print(rat.round(3).to_string(index=False))
    print(pd.DataFrame(counts, columns=["step", "records"]).to_string(index=False))
    print(f"Buyer WTP: median {b['wtp'].median():,.0f}, "
          f"P10 {b['wtp'].quantile(.1):,.0f}, P90 {b['wtp'].quantile(.9):,.0f}")
    print(f"Pool quality: P10 {pool['quality'].quantile(.1):.2f}, "
          f"P90 {pool['quality'].quantile(.9):.2f}")


if __name__ == "__main__":
    main()
