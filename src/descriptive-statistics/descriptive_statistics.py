"""Descriptive statistics of the HMTS household and parcel variables used in the WTP model.

Reproduces Table "Descriptive Statistics of Household and Parcel Characteristics
Retained in Modelling" from the raw survey extract ``inputs/HMTS Data_WTP.xlsx``.

Corrections relative to the earlier table (which averaged the raw worksheet columns):

* The HMTS special code -9999 (and -9.999 after the m -> km conversion) is treated
  as missing. Statistics use valid responses only, and the valid n is reported.
* "Income" previously showed the mean of the ``WTP (in 1000)`` column with its sign
  dropped (-216,910.86). Household income (``hinc``) is an ordinal code 1-7, so it
  is reported as a category mean together with the count per code.
* "Owned property" (89) and "population density" (7) are constants that are the
  same for every household. They are not household proportions.
* "Distance to CBD" is ``CurrWork_dist / 1000``, i.e. the home-to-work distance, so
  it is reported once.
* "Distance to nearest park" and "distance to nearest school" are both
  ``Busstop_dist / 1000`` (the GIS distance to the nearest bus stop), so they are
  reported once.
* "Parcel size" is derived as rooms x 240 sq ft / 43,560 and is not a measured lot area.

Usage (from the package root)::

    python src/descriptive-statistics/descriptive_statistics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "inputs" / "HMTS Data_WTP.xlsx"
DEFAULT_OUTDIR = ROOT / "results" / "descriptive-statistics"
SHEET = "Copy_Of_Query1"
N_TOTAL_EXPECTED = 313


def load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET)
    df.columns = [c.strip() for c in df.columns]
    df["hinc"] = pd.to_numeric(df["hinc"], errors="coerce")
    return df


def missing(s: pd.Series) -> pd.Series:
    """HMTS special codes: -9999 in raw units, -9.999 after dividing metres by 1000."""
    return s.isin([-9999, -9.999]) | s.isna()


def derived(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild every modelled variable from its raw source column, with missing as NaN.

    The worksheet formulas are reproduced exactly; only the missing-code handling differs.
    """
    out = pd.DataFrame(index=df.index)
    raw = {c: df[c].mask(missing(df[c])) for c in
           ["child", "cars", "h2tsd", "hinc", "hc1_r", "room", "dweltyp",
            "CurrWork_dist", "Busstop_dist", "Shop_dist", "value_cur"]}

    # Worksheet: =IF(AD>1,1,0), i.e. a household is a mover when WTP > $1.
    out["mover"] = (df["WTP (in 1000)"] > 1).astype(float)
    out["owned_property"] = df["% owned property (x10)"].astype(float)
    out["population_density"] = df["population density (x9)"].astype(float)
    out["children"] = (raw["child"] > 0).astype(float).mask(raw["child"].isna())
    out["cars"] = raw["cars"]
    out["no_car"] = (raw["cars"] == 0).astype(float).mask(raw["cars"].isna())
    out["birth"] = (raw["hc1_r"] == 1).astype(float).mask(raw["hc1_r"].isna())
    out["income_code"] = raw["hinc"]
    out["single_detached"] = (raw["dweltyp"] == 1).astype(float).mask(raw["dweltyp"].isna())
    out["rooms"] = raw["room"]
    out["parcel_acres"] = raw["room"] * 240 / 43560
    out["bus_stop_self_km"] = raw["h2tsd"] / 1000
    out["work_km"] = raw["CurrWork_dist"] / 1000
    out["bus_stop_gis_km"] = raw["Busstop_dist"] / 1000
    out["business_km"] = raw["Shop_dist"] / 1000
    wtp = df["WTP (in 1000)"]
    out["wtp_movers"] = wtp.where(wtp > 1)
    return out


def check_against_worksheet(df: pd.DataFrame, d: pd.DataFrame) -> None:
    """Confirm the rebuilt variables equal the worksheet columns wherever data are valid."""
    pairs = {
        "single_detached": "single-detached house (x5)",
        "parcel_acres": "parcel size (x1) (acres)",
        "children": "presence of children (x6)",
        "no_car": "no car ownership (x3)",
        "birth": "birth of a child (x1)",
        "bus_stop_self_km": "distacne to nearest bus stop(km) (x7)",
        "work_km": "distance to CBD (x3) (km)",
        "bus_stop_gis_km": "distance to nearest park area (x8) (km)",
        "business_km": "distance to nearest business center (x5) (km)",
        "mover": "move (1), stay (0)",
    }
    for mine, theirs in pairs.items():
        ok = d[mine].notna()
        if not np.allclose(d.loc[ok, mine], df.loc[ok, theirs]):
            raise AssertionError(f"{mine} does not match worksheet column {theirs!r}")
    # Documented duplicates in the worksheet.
    assert np.allclose(df["distance to nearest school (x6) (km)"],
                       df["distance to nearest park area (x8) (km)"])
    assert np.allclose(df["distance to CBD (x3) (km)"], df["CurrWork_dist"] / 1000)


# (key, group, label, description, kind) where kind is "share", "mean" or "constant"
ROWS = [
    ("mover", "Household", "Mover (1), non-mover (0)",
     "Classified as mover when computed WTP $>$ 0 = 1, 0 otherwise", "share"),
    ("children", "Household", "Presence of children",
     "Household has at least one child = 1, 0 otherwise", "share"),
    ("birth", "Household", "Birth of a child",
     "Birth of a child expected in the household = 1, 0 otherwise", "share"),
    ("cars", "Household", "Number of cars",
     "Total number of vehicles owned by the household", "mean"),
    ("no_car", "Household", "No car ownership",
     "Household owns no cars = 1, 0 otherwise", "share"),
    ("income_code", "Household", "Income category",
     "Annual household income, ordinal code 1 (lowest) to 7 (highest)", "mean"),
    ("wtp_movers", "Household", "Willingness to pay",
     "Computed WTP of movers, before the 1.8 calibration scaling (CAD)", "mean"),
    ("owned_property", "Household", "Owned property (\\%)",
     "Share of owner-occupied dwellings; same value for every household", "constant"),
    ("population_density", "Household", "Population density",
     "Population density index; same value for every household", "constant"),
    ("single_detached", "Parcel", "Dwelling type (single detached)",
     "Single-detached dwelling = 1, 0 otherwise", "share"),
    ("rooms", "Parcel", "Number of rooms",
     "Total number of rooms in the dwelling", "mean"),
    ("parcel_acres", "Parcel", "Parcel size",
     "Derived as rooms $\\times$ 240 sq ft (acres)", "mean"),
    ("bus_stop_self_km", "Parcel", "Distance to nearest bus stop",
     "Self-reported home to transit stop distance (km)", "mean"),
    ("work_km", "Parcel", "Distance to work location / CBD",
     "Home to work distance (km); also used as the CBD distance term", "mean"),
    ("bus_stop_gis_km", "Parcel", "Distance to nearest park / school",
     "GIS distance to nearest bus stop (km); used for both the park and school terms", "mean"),
    ("business_km", "Parcel", "Distance to nearest business centre",
     "GIS distance to nearest shopping centre (km)", "mean"),
]


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for key, group, label, desc, kind in ROWS:
        s = d[key].dropna()
        recs.append({
            "group": group, "variable": label, "description": desc, "kind": kind,
            "n": int(s.size), "mean": s.mean(), "sd": s.std(ddof=1),
            "min": s.min(), "max": s.max(),
        })
    return pd.DataFrame(recs)


def fmt_value(r: pd.Series) -> tuple[str, str]:
    if r.kind == "share":
        return f"{100 * r['mean']:.2f}\\%", "--"
    if r.kind == "constant":
        return f"{r['mean']:.0f}", "0"
    if r.variable == "Willingness to pay":
        return f"{r['mean']:,.0f}", f"{r['sd']:,.0f}"
    return f"{r['mean']:.4f}", f"{r['sd']:.4f}"


def to_latex(t: pd.DataFrame, income_counts: pd.Series, n_income_missing: int) -> str:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Descriptive Statistics of Household and Parcel Characteristics Retained in Modelling ($N=" + str(N_TOTAL_EXPECTED) + r"$)}",
        r"\label{tab:descriptive_statistics}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}Xrrr@{}}",
        r"\toprule",
        r"\textbf{Variable} & \textbf{Description} & \textbf{Mean/Prop.} & \textbf{SD} & \textbf{$n$} \\",
        r"\midrule",
    ]
    for group in ["Household", "Parcel"]:
        lines += [r"\multicolumn{5}{@{}l}{\textbf{" + group + r" Characteristics}} \\", r"\midrule"]
        for _, r in t[t.group == group].iterrows():
            m, sd = fmt_value(r)
            lines.append(f"{r.variable} & {r.description} & {m} & {sd} & {r.n} \\\\")
        lines.append(r"\midrule")
    counts = ", ".join(f"{int(k)}: {v}" for k, v in income_counts.items())
    lines[-1] = r"\bottomrule"
    lines += [
        r"\multicolumn{5}{@{}p{\textwidth}@{}}{\footnotesize \textit{Notes:} HMTS missing codes ($-9999$) are excluded, so valid $n$ varies. "
        r"Income is an ordinal category, not a dollar amount; counts by code: " + counts + f"; {n_income_missing} missing. "
        r"Willingness to pay is reported for movers only. "
        r"Distance to work and distance to CBD are the same variable, as are distance to the nearest park and to the nearest school (both the GIS bus stop distance). "
        r"In the WTP equation, missing children and birth responses enter as 0 and missing bus stop distances as $-9.999$ km.} \\",
        r"\end{tabularx}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    df = load(args.input)
    assert len(df) == N_TOTAL_EXPECTED, f"expected {N_TOTAL_EXPECTED} households, got {len(df)}"
    d = derived(df)
    check_against_worksheet(df, d)
    table = summarise(d)

    income = d["income_code"]
    income_counts = income.dropna().astype(int).value_counts().sort_index()
    n_income_missing = int(income.isna().sum())

    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "descriptive_statistics.csv", index=False, float_format="%.6f")
    income_counts.rename_axis("income_code").rename("count").to_csv(args.outdir / "income_categories.csv")
    (args.outdir / "descriptive_statistics.tex").write_text(to_latex(table, income_counts, n_income_missing))

    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(table[["group", "variable", "n", "mean", "sd", "min", "max"]].round(4).to_string(index=False))
    print("\nIncome counts:", income_counts.to_dict(), "| missing:", n_income_missing)
    print(f"\nWrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
