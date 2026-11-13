"""Bidding policies. All of them act inside the same ``Market``.

* ``RulePolicy``      the published rule: raise the bid by U(1%, 5%) of the bid each round
* ``RandomPolicy``    uniform over the discrete actions
* ``FixedPolicy``     always the same discrete action (reference points)
* ``QLearningPolicy`` tabular Q-learning, one table shared by all buyers
* ``DQNPolicy``       DQN with replay, warm-up, target network, Huber loss, gradient clipping
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

MINIMAL_COLS = [0, 1]


class RulePolicy:
    kind = "rule"
    name = "Rule-based"

    def act(self, f, eps):  # never called; the market applies the rule directly
        raise NotImplementedError


class RandomPolicy:
    kind = "discrete"

    def __init__(self, n_actions, seed):
        self.n = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, f, eps):
        return self.rng.integers(0, self.n, len(f))


class FixedPolicy:
    kind = "discrete"

    def __init__(self, action):
        self.a = action

    def act(self, f, eps):
        return np.full(len(f), self.a)


# ---------------------------------------------------------------------- tabular
class Discretizer:
    """Rich: gap x room x round x rivals x last year's sale-to-ask = 6 x 4 x 4 x 3 x 3 = 864.
    Minimal: bid/ask x counter/ask = 8 x 6 = 48 (closest to the submitted state)."""

    def __init__(self, state: str, max_rounds: int):
        self.state = state
        self.K = max_rounds
        if state == "rich":
            self.edges = [
                (2, np.array([0.005, 0.02, 0.04, 0.07, 0.12])),
                (3, np.array([0.02, 0.06, 0.15])),
                (7, np.array([0.5, 1.5])),
                (8, np.array([-0.02, 0.01])),
            ]
            self.dims = [6, 4, 4, 3, 3]
        else:
            self.edges = [
                (0, np.array([0.80, 0.85, 0.88, 0.91, 0.94, 0.97, 1.0])),
                (1, np.array([0.92, 0.95, 0.98, 1.0, 1.02])),
            ]
            self.dims = [8, 6]
        self.n_states = int(np.prod(self.dims))

    def __call__(self, f):
        if self.state == "rich":
            g = np.digitize(f[:, 2], self.edges[0][1])
            r = np.digitize(f[:, 3], self.edges[1][1])
            k = np.rint(f[:, 4] * self.K).astype(int)
            kb = np.select([k == 0, k == 1, k <= 3], [0, 1, 2], 3)
            v = np.digitize(f[:, 7], self.edges[2][1])
            m = np.digitize(f[:, 8], self.edges[3][1])
            parts = [g, r, kb, v, m]
        else:
            parts = [np.digitize(f[:, c], e) for c, e in self.edges]
        return np.ravel_multi_index(parts, self.dims)


class QLearningPolicy:
    kind = "discrete"
    name = "Q-learning"

    def __init__(self, n_actions, state, max_rounds, alpha, gamma, seed, alpha_c=0.0):
        self.disc = Discretizer(state, max_rounds)
        self.alpha_c = alpha_c  # > 0: alpha / (1 + visits(s, a) / alpha_c), a decaying step size
        self.Q = np.zeros((self.disc.n_states, n_actions))
        self.visits = np.zeros((self.disc.n_states, n_actions), dtype=np.int64)
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.n = n_actions
        self.reset_metrics()

    def reset_metrics(self):
        self._td = []
        self.updates = 0

    def greedy(self, f, tie_break=True):
        s = self.disc(f)
        q = self.Q[s]
        if tie_break:
            q = q + 1e-9 * self.rng.random(q.shape)
        return q.argmax(1)

    def act(self, f, eps):
        a = self.greedy(f)
        if eps > 0:
            m = self.rng.random(len(f)) < eps
            a[m] = self.rng.integers(0, self.n, m.sum())
        return a

    def learn(self, S, A, R, S2, D, K):
        s = self.disc(S)
        s2 = self.disc(S2)
        Q, al, ga, V, c = self.Q, self.alpha, self.gamma, self.visits, self.alpha_c
        td_abs = np.empty(len(s))
        for i in range(len(s)):
            si, ai = s[i], A[i]
            target = R[i] if D[i] else R[i] + ga * Q[s2[i]].max()
            td = target - Q[si, ai]
            step = al / (1.0 + V[si, ai] / c) if c > 0 else al
            Q[si, ai] += step * td
            V[si, ai] += 1
            td_abs[i] = abs(td)
        self._td.append(td_abs)
        self.updates += len(s)

    def metrics(self):
        td = np.concatenate(self._td) if self._td else np.array([np.nan])
        vis = self.visits > 0
        qv = self.Q[vis] if vis.any() else np.array([0.0])
        return {"td_error": float(np.mean(td)), "q_mean": float(qv.mean()),
                "q_max": float(qv.max()), "states_visited": int(vis.any(1).sum()),
                "updates": self.updates}

    def snapshot(self):
        return self.Q.copy()

    def state_dict(self):
        return {"Q": self.Q.copy(), "visits": self.visits.copy()}

    def load_state_dict(self, sd):
        self.Q = sd["Q"].copy()
        self.visits = sd["visits"].copy()


# ---------------------------------------------------------------------- DQN
class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNPolicy:
    kind = "discrete"
    name = "DQN"

    def __init__(self, n_actions, state, p, seed, n_features=10):
        self.cols = list(range(n_features)) if state == "rich" else MINIMAL_COLS
        self.p = p
        self.n = n_actions
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.online = MLP(len(self.cols), p.hidden, n_actions)
        self.target = copy.deepcopy(self.online)
        self.ema = copy.deepcopy(self.online) if p.polyak_eval > 0 else None
        self.opt = torch.optim.Adam(self.online.parameters(), lr=p.lr)
        self.mu = np.zeros(len(self.cols))
        self.sd = np.ones(len(self.cols))
        cap = p.buffer_size
        self.buf_s = np.zeros((cap, len(self.cols)), np.float32)
        self.buf_s2 = np.zeros((cap, len(self.cols)), np.float32)
        self.buf_a = np.zeros(cap, np.int64)
        self.buf_r = np.zeros(cap, np.float32)
        self.buf_d = np.zeros(cap, np.float32)
        self.ptr = 0
        self.size = 0
        self.pending = 0
        self.grad_steps = 0
        self.reset_metrics()

    def set_lr_factor(self, factor):
        for g in self.opt.param_groups:
            g["lr"] = self.p.lr * factor

    def set_normalization(self, F):
        """Fixed scaling estimated once from a random-policy rollout on training years."""
        X = F[:, self.cols]
        self.mu = X.mean(0)
        self.sd = np.where(X.std(0) > 1e-6, X.std(0), 1.0)

    def _x(self, f):
        return torch.as_tensor(((f[:, self.cols] - self.mu) / self.sd).astype(np.float32))

    def reset_metrics(self):
        self._loss, self._qm, self._qx = [], [], []

    def q_values(self, f, net=None):
        net = net or (self.ema if self.ema is not None else self.online)
        with torch.no_grad():
            return net(self._x(f)).numpy()

    def greedy(self, f, tie_break=True):
        """Greedy action of the evaluated network (EMA weights when polyak_eval > 0)."""
        return self.q_values(f).argmax(1)

    def act(self, f, eps):
        # behaviour policy during training uses the online network; frozen evaluation (eps = 0,
        # no learning) uses the evaluated network
        a = self.q_values(f, self.online if eps > 0 else None).argmax(1)
        if eps > 0:
            m = self.rng.random(len(f)) < eps
            a[m] = self.rng.integers(0, self.n, m.sum())
        return a

    def learn(self, S, A, R, S2, D, K):
        n = len(A)
        cap = self.p.buffer_size
        idx = (self.ptr + np.arange(n)) % cap
        self.buf_s[idx] = ((S[:, self.cols] - self.mu) / self.sd)
        self.buf_s2[idx] = ((S2[:, self.cols] - self.mu) / self.sd)
        self.buf_a[idx], self.buf_r[idx], self.buf_d[idx] = A, R, D
        self.ptr = (self.ptr + n) % cap
        self.size = min(self.size + n, cap)
        if self.size < self.p.warmup:
            return
        self.pending += n
        steps, self.pending = divmod(self.pending, self.p.train_every)
        for _ in range(steps):
            self._step()

    def _step(self):
        p = self.p
        j = self.rng.integers(0, self.size, p.batch_size)
        s = torch.from_numpy(self.buf_s[j])
        s2 = torch.from_numpy(self.buf_s2[j])
        a = torch.from_numpy(self.buf_a[j])
        r = torch.from_numpy(self.buf_r[j])
        d = torch.from_numpy(self.buf_d[j])
        q_all = self.online(s)
        q = q_all.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            if p.double:
                a2 = self.online(s2).argmax(1, keepdim=True)
                q2 = self.target(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.target(s2).max(1).values
            y = r + p.gamma * (1 - d) * q2
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), p.grad_clip)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % p.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        if self.ema is not None:
            with torch.no_grad():
                for pe, po in zip(self.ema.parameters(), self.online.parameters()):
                    pe.mul_(p.polyak_eval).add_(po, alpha=1 - p.polyak_eval)
        self._loss.append(loss.item())
        qd = q_all.detach()
        self._qm.append(qd.mean().item())
        self._qx.append(qd.max().item())

    def metrics(self):
        f = lambda v: float(np.mean(v)) if v else np.nan
        return {"td_error": f(self._loss), "q_mean": f(self._qm),
                "q_max": float(np.max(self._qx)) if self._qx else np.nan,
                "grad_steps": self.grad_steps, "buffer": self.size}

    def snapshot(self):
        return None

    def state_dict(self):
        d = {"online": copy.deepcopy(self.online.state_dict()), "mu": self.mu.copy(), "sd": self.sd.copy()}
        if self.ema is not None:
            d["ema"] = copy.deepcopy(self.ema.state_dict())
        return d

    def load_state_dict(self, sd):
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["online"])
        if self.ema is not None:
            self.ema.load_state_dict(sd.get("ema", sd["online"]))
        self.mu, self.sd = sd["mu"].copy(), sd["sd"].copy()


class MixedPolicy:
    """Diagnostic: a share of buyers follows ``minority``, the rest ``majority``. Group membership is
    fixed per negotiation (hash of WTP / ask, constant within a negotiation)."""
    kind = "discrete"

    def __init__(self, majority, minority, share=0.1):
        self.maj, self.mino, self.share = majority, minority, share

    def group(self, f):
        h = (np.floor(f[:, 5] * 1e6) * 2654435761) % 1000 / 1000.0
        return (h < self.share).astype(int)

    def act(self, f, eps):
        g = self.group(f)
        a = self.maj.act(f, 0.0)
        if g.any():
            a[g == 1] = self.mino.act(f[g == 1], 0.0)
        return a
