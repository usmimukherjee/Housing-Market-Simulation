"""Shared helpers for the numbered experiment scripts."""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import pandas as pd

from .config import RESULTS, Bundle, MarketParams, RewardParams, Spec


def load_bundle(reward: str = "realism", state: str = "rich", actions: str = "A3") -> Bundle:
    """Shared environment: market parameters selected by 1_calibrate_market.py."""
    p = RESULTS / "calibration" / "market_params.json"
    mp = MarketParams(**json.loads(p.read_text()))
    return Bundle(market=mp, reward=RewardParams(kind=reward), spec=Spec(state=state, actions=actions))


def save_run(outdir: Path, tag: str, ep_df: pd.DataFrame, ck_df: pd.DataFrame, best_state, final_state,
             meta: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    ep_df.to_csv(outdir / f"{tag}_episodes.csv", index=False)
    ck_df.to_csv(outdir / f"{tag}_checkpoints.csv", index=False)
    with open(outdir / f"{tag}_best.pkl", "wb") as f:
        pickle.dump(best_state, f)
    with open(outdir / f"{tag}_final.pkl", "wb") as f:
        pickle.dump(final_state, f)
    (outdir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=float))


def load_state(outdir: Path, tag: str, which: str = "best"):
    with open(outdir / f"{tag}_{which}.pkl", "rb") as f:
        return pickle.load(f)


def asdict_all(*objs) -> dict:
    return {type(o).__name__: dataclasses.asdict(o) for o in objs}
