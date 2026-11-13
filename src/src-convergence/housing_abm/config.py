"""Every setting used by the convergence experiments.

Values marked "calibrated" are overwritten by the market calibration step
(``1_calibrate_market.py``) and saved in ``results/results-convergence/config_final.json``.
Values marked "tuned" are overwritten by hyperparameter tuning (``2_tune_hyperparameters.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DERIVED = PACKAGE_ROOT / "inputs" / "derived"
RESULTS = PACKAGE_ROOT / "results" / "results-convergence"
GRAPHS = PACKAGE_ROOT / "graphs"
LOGS = PACKAGE_ROOT / "logs" / "convergence"

# ---------------------------------------------------------------- temporal protocol
FIRST_YEAR = 2010                      # initial condition: ask index starts at the 2010 median
TRAIN_YEARS = list(range(2010, 2019))  # simulated during RL training; reward targets available
TRAIN_FIT_YEARS = list(range(2011, 2019))  # years scored for calibration fit (2010 is the anchor)
VAL_YEARS = [2019, 2020, 2021]         # market calibration selection, RL hyperparameters, checkpoints
TEST_YEARS = [2022, 2023, 2024]        # reported once, never used for any choice
ALL_YEARS = list(range(2010, 2025))
RATIO_YEARS = [2019, 2020, 2021, 2022, 2023]  # assessments exist for these tax years only

# ---------------------------------------------------------------- seeds
TRAIN_SEEDS = list(range(10))               # RL / DQN training seeds for final runs
TUNE_SEEDS = [100, 101]                     # seeds used during hyperparameter tuning
CHECKPOINT_EVAL_SEEDS = [3000, 3001, 3002, 3003, 3004]  # market seeds for checkpoint validation during training
FINAL_EVAL_SEEDS = list(range(1000, 1020))  # market seeds shared by every model in final evaluation
CALIB_SEEDS = [500, 501, 502, 503, 504]               # market seeds for calibrating the environment


@dataclass
class MarketParams:
    n_listings: int = 300            # new listings per simulated year (scaled sample)
    buyers_per_listing: float = 1.2  # calibrated: new buyers per new listing
    wtp_scale: float = 1.8           # calibrated: multiplies survey WTP (dollars)
    income_growth: float = 0.02      # calibrated: annual WTP growth
    feedback_sellthrough: float = 0.1  # calibrated: ask index response to sell-through
    feedback_sale_to_ask: float = 1.0  # ask index moves with the realised sale-to-ask ratio
    target_sellthrough: float = 0.6
    search_floor: float = 0.7        # lowest ask considered, as a share of WTP
    cap: float = 1.2                 # bids can never exceed cap x WTP
    open_wtp: float = 0.8            # opening bid = min(open_wtp x WTP, open_ask x ask)
    open_ask: float = 0.92
    ask_noise: float = 0.05
    wtp_noise: float = 0.10
    reservation_lo: float = 0.90     # seller reservation = U(lo, hi) x ask, private to seller
    reservation_hi: float = 0.98
    concession: float = 0.35         # seller closes this share of the gap per round, / (1 + rivals)
    escalation: float = 0.01         # counter starts at ask x (1 + escalation x rivals)
    max_rounds: int = 8
    cycles: int = 3                  # sub-market cycles per year
    listing_carry: float = 0.5       # probability an unsold listing stays on the market
    listing_cut: float = 0.03        # ask cut for a carried listing
    buyer_carry: float = 0.6         # probability an unsuccessful buyer stays in the market


@dataclass
class RewardParams:
    kind: str = "realism"        # "realism" (proposed), "deal" (original-style), "surplus"
    deal: float = 1.0
    price_error: float = 2.0     # x |P - P*| / P*, capped at 1; P* = quality x observed median
    overpay: float = 1.0         # x max(0, P - WTP) / WTP
    step: float = 0.02           # per negotiation round
    fail: float = 0.5            # timeout or impasse
    lost: float = 0.25           # reached a deal but a rival paid more
    surplus_weight: float = 2.0  # "surplus" reward: deal + weight x (cap - P) / cap


@dataclass
class QParams:
    alpha: float = 0.1     # tuned
    alpha_c: float = 0.0   # tuned; > 0 decays the step size as alpha / (1 + visits(s, a) / alpha_c)
    gamma: float = 0.95    # tuned
    eps_start: float = 1.0  # tuned
    eps_min: float = 0.05   # tuned
    explore_fraction: float = 0.5  # epsilon decays linearly over this share of episodes


@dataclass
class DQNParams:
    hidden: tuple = (64, 64)  # tuned
    lr: float = 5e-4          # tuned
    gamma: float = 0.95       # tuned
    batch_size: int = 64      # tuned
    buffer_size: int = 50_000  # tuned
    target_update: int = 500  # tuned, in gradient steps
    warmup: int = 2_000       # transitions collected before the first gradient step
    train_every: int = 4      # one gradient step per this many transitions
    grad_clip: float = 10.0
    double: bool = False      # Double DQN only if overestimation is demonstrated
    lr_decay: bool = False    # linear decay of the learning rate to lr_final_frac x lr over training
    lr_anneal: bool = False   # constant during exploration, then exponential decay to lr_anneal_frac x lr
    lr_final_frac: float = 0.02
    lr_anneal_frac: float = 1e-3
    polyak_eval: float = 0.0  # > 0: greedy policy uses an EMA of the online weights (tau per gradient step)
    eps_start: float = 1.0
    eps_min: float = 0.05
    explore_fraction: float = 0.5


@dataclass
class TrainParams:
    episodes: int = 300
    eval_every: int = 10
    ma_window: int = 20
    # convergence criterion, checked at each checkpoint after exploration has finished
    conv_reward_rel: float = 0.05   # |change in greedy-reward MA (5 checkpoints)| / max(|MA|, 0.1)
    conv_policy_change: float = 0.05  # share of probe states whose greedy action changed
    conv_val_improve: float = 0.005   # validation score improvement that counts as meaningful
    conv_k: int = 3                   # consecutive checkpoints required
    patience: int = 10                # checkpoints without validation improvement before stopping


@dataclass
class Spec:
    state: str = "rich"      # "rich" or "minimal" (bid/ask and counter/ask only)
    actions: str = "A3"      # "A3": +1/3/6% of ask (selected: validation ablation); "A5" adds hold, match


@dataclass
class ValScore:
    ratio_weight: float = 1.0   # lambda_1
    volume_weight: float = 0.0  # lambda_2: simulated flows are fixed, volume not comparable


def to_dict(*objs) -> dict:
    return {type(o).__name__: asdict(o) for o in objs}


@dataclass
class Bundle:
    market: MarketParams = field(default_factory=MarketParams)
    reward: RewardParams = field(default_factory=RewardParams)
    spec: Spec = field(default_factory=Spec)
