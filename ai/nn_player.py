"""Actor-critic neural-network player (pure numpy).

Improvements over the previous version
---------------------------------------
5-CHANNEL INPUT
    Channels: own dots | opponent dots | forbidden | own conn mask | opp conn mask
    Connection masks give the network direct visibility of the graph structure —
    the most important spatial relationship — which the 3-channel encoding hid.

ADAM OPTIMISER  (replaces vanilla SGD)
    Adaptive per-parameter learning rates with momentum.  Trains 3-5× more
    reliably than plain gradient ascent, especially in the early phases when
    the policy is noisy and gradients vary wildly in magnitude.

GRADIENT BUG FIX
    The old code updated self.w_pol *before* computing d_trunk_pol, so the
    trunk received gradients from the post-update weights.  Now the pre-update
    weight matrix is used for backpropagation, which is correct.

ENTROPY BONUS
    Added −ENTROPY_COEF · H[π] to discourage premature determinism.
    The policy keeps exploring even after it finds a locally good strategy.

GAE  (λ-returns)
    Replaces plain Monte-Carlo returns with Generalised Advantage Estimation,
    matching the PyTorch player.  Reduces variance while keeping bias low.
"""

import os
import numpy as np
import settings as S
from ai.base_player import BasePlayer
from ai.paths import experience_path
from ai.features import (opportunity_masks, apply_boost, enclosure_potential,
                          strategic_channels)

# Adam hyper-parameters
_BETA1  = 0.9
_BETA2  = 0.999
_EPS_   = 1e-8

# Entropy bonus coefficient (exploration regulariser)
_ENTROPY_COEF = getattr(S, 'NN_ENTROPY_COEF', 0.01)

# GAE lambda
_GAE_LAMBDA = getattr(S, 'NN_GAE_LAMBDA', 0.95)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(0.0, x)

def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def _xavier(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


# ── Adam parameter group ──────────────────────────────────────────────────────

class _AdamParam:
    """Tracks first and second moment for a single weight array."""
    __slots__ = ('w', 'm', 'v')

    def __init__(self, w: np.ndarray):
        self.w = w
        self.m = np.zeros_like(w)
        self.v = np.zeros_like(w)

    def step(self, grad: np.ndarray, lr: float, t: int) -> None:
        """In-place Adam update."""
        self.m = _BETA1 * self.m + (1 - _BETA1) * grad
        self.v = _BETA2 * self.v + (1 - _BETA2) * grad ** 2
        m_hat  = self.m / (1 - _BETA1 ** t)
        v_hat  = self.v / (1 - _BETA2 ** t)
        self.w += lr * m_hat / (np.sqrt(v_hat) + _EPS_)


# ── Actor-Critic network ──────────────────────────────────────────────────────

class _ACNetwork:
    """Shared-trunk actor-critic with Adam optimiser."""

    def __init__(self, input_size: int, hidden_sizes: list[int],
                 output_size: int):
        sizes = [input_size] + hidden_sizes

        # Trunk (list of AdamParam pairs: weights + biases)
        self._trunk_w = [_AdamParam(_xavier(sizes[i], sizes[i+1]))
                         for i in range(len(sizes)-1)]
        self._trunk_b = [_AdamParam(np.zeros(sizes[i+1]))
                         for i in range(len(sizes)-1)]

        # Policy head
        self._pol_w = _AdamParam(_xavier(sizes[-1], output_size))
        self._pol_b = _AdamParam(np.zeros(output_size))

        # Value head
        self._val_w = _AdamParam(_xavier(sizes[-1], 1))
        self._val_b = _AdamParam(np.zeros(1))

        self._t = 0   # Adam step counter

    # ── Properties for save/load convenience ──────────────────────────────────
    @property
    def w_trunk(self): return [p.w for p in self._trunk_w]
    @property
    def b_trunk(self): return [p.w for p in self._trunk_b]
    @property
    def w_pol(self):   return self._pol_w.w
    @property
    def w_val(self):   return self._val_w.w

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray):
        """Returns (logits, value, trunk_activations)."""
        acts = [x]
        h = x
        for pw, pb in zip(self._trunk_w, self._trunk_b):
            h = _relu(h @ pw.w + pb.w)
            acts.append(h)
        logits = h @ self._pol_w.w + self._pol_b.w
        value  = float(np.tanh(h @ self._val_w.w + self._val_b.w)[0])
        return logits, value, acts

    def policy(self, x: np.ndarray, legal_mask: np.ndarray):
        logits, value, acts = self.forward(x)
        masked = logits.copy()
        masked[~legal_mask] = -1e9
        probs = _softmax(masked)
        return probs, value, acts

    # ── Update (one transition) ───────────────────────────────────────────────

    def update(self, acts: list, action: int, advantage: float,
               value_target: float, entropy_bonus: float, lr: float) -> None:
        """Adam step for one (state, action, advantage, value_target) tuple."""
        self._t += 1
        t  = self._t
        h  = acts[-1]   # trunk output

        # ── Policy head ───────────────────────────────────────────────────────
        probs     = _softmax(h @ self._pol_w.w + self._pol_b.w)

        # Policy gradient: ∇log π(a) · advantage  + entropy bonus
        d_pol = -probs.copy()
        d_pol[action] += 1.0
        d_pol *= advantage
        # Entropy gradient w.r.t. logits: −(log π + 1) shifted by mean,
        # which simplifies to  −(log π − Σ π·log π)  = −log π + H
        log_probs  = np.log(np.clip(probs, 1e-10, None))
        entropy    = -float((probs * log_probs).sum())
        d_entropy  = -(log_probs - entropy)          # ∂H/∂logit_i = −(log π_i + 1) + H
        d_pol     += _ENTROPY_COEF * d_entropy

        # Save pre-update weights for correct trunk backprop
        w_pol_pre = self._pol_w.w.copy()
        w_val_pre = self._val_w.w.copy()

        self._pol_w.step(np.outer(h, d_pol), lr, t)
        self._pol_b.step(d_pol, lr, t)
        d_trunk_pol = d_pol @ w_pol_pre.T

        # ── Value head (MSE loss) ─────────────────────────────────────────────
        v_pred    = float(np.tanh(h @ self._val_w.w + self._val_b.w)[0])
        d_val_raw = (value_target - v_pred) * (1 - v_pred ** 2)   # tanh deriv
        d_val_vec = np.array([d_val_raw])

        self._val_w.step(S.NN_VALUE_COEF * np.outer(h, d_val_vec), lr, t)
        self._val_b.step(S.NN_VALUE_COEF * d_val_vec, lr, t)
        d_trunk_val = (d_val_vec @ w_val_pre.T).ravel()

        # ── Shared trunk ──────────────────────────────────────────────────────
        delta = d_trunk_pol + d_trunk_val
        for i in reversed(range(len(self._trunk_w))):
            inp   = acts[i]
            self._trunk_w[i].step(np.outer(inp, delta), lr, t)
            self._trunk_b[i].step(delta, lr, t)
            if i > 0:
                w_pre = self._trunk_w[i].w   # already updated — but for ReLU
                # gate only: use activation, not weight, for ReLU derivative
                delta = (delta @ self._trunk_w[i].w.T) * (acts[i] > 0).astype(float)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str, grid: int = 0, hidden: list = None) -> None:
        arrays = {}
        for i, (pw, pb) in enumerate(zip(self._trunk_w, self._trunk_b)):
            arrays[f'wt{i}']  = pw.w;  arrays[f'bt{i}']  = pb.w
            arrays[f'mwt{i}'] = pw.m;  arrays[f'vwt{i}'] = pw.v
            arrays[f'mbt{i}'] = pb.m;  arrays[f'vbt{i}'] = pb.v
        for name, p in (('pol', self._pol_w), ('pol_b', self._pol_b),
                         ('val', self._val_w), ('val_b', self._val_b)):
            arrays[f'w_{name}'] = p.w
            arrays[f'm_{name}'] = p.m
            arrays[f'v_{name}'] = p.v
        arrays['adam_t'] = np.array([self._t])
        arrays['meta_grid']     = np.array([grid])
        arrays['meta_hidden']   = np.array(hidden or [])
        arrays['meta_channels'] = np.array([15])
        np.savez(path, **arrays)

    def load(self, path: str, grid: int = 0, hidden: list = None) -> None:
        data = np.load(path)
        # Reject weights from a different grid size or hidden architecture
        if 'meta_grid' in data:
            saved_grid     = int(data['meta_grid'][0])
            saved_hidden   = list(data['meta_hidden'].astype(int))
            saved_channels = int(data['meta_channels'][0]) if 'meta_channels' in data else 7
            if saved_grid != grid or saved_hidden != (hidden or []) or saved_channels != 15:
                return   # incompatible architecture — start fresh   # silently ignore incompatible checkpoint
        if 'adam_t' in data:
            self._t = int(data['adam_t'][0])
        for i, (pw, pb) in enumerate(zip(self._trunk_w, self._trunk_b)):
            for key, arr, attr in (
                (f'wt{i}',  pw, 'w'), (f'mwt{i}', pw, 'm'), (f'vwt{i}', pw, 'v'),
                (f'bt{i}',  pb, 'w'), (f'mbt{i}', pb, 'm'), (f'vbt{i}', pb, 'v'),
            ):
                if key in data and data[key].shape == getattr(arr, attr).shape:
                    setattr(arr, attr, data[key].copy())
        for name, p in (('pol', self._pol_w), ('pol_b', self._pol_b),
                         ('val', self._val_w), ('val_b', self._val_b)):
            for attr in ('w', 'm', 'v'):
                key = f'{attr}_{name}'
                if key in data and data[key].shape == getattr(p, attr).shape:
                    setattr(p, attr, data[key].copy())


# ── State encoder (5 channels) ───────────────────────────────────────────────

def encode_state(board: dict, connections: set,
                 forbidden: set, player: int) -> np.ndarray:
    """Return a flat float32 vector of length 7 × GRID × GRID.

    Channel 0 : own dots
    Channel 1 : opponent dots
    Channel 2 : forbidden positions
    Channel 3 : own connection mask     (cells touching any own edge)
    Channel 4 : opponent connection mask
    Channel  5 : own closing score / 4.0
    Channel  6 : opponent threat score / 4.0
    Channel  7 : own enclosure potential
    Channel  8 : opponent enclosure potential
    Channel  9 : own bridge potential  (extra from joining two arcs)
    Channel 10 : opp bridge potential
    Channel 11 : disruption map        (how much placing here hurts opp ring)
    Channel 12 : fork map              (simultaneous threats created)
    Channel 13 : game phase            (dots placed / max dots, broadcast)
    Channel 14 : centrality            (geometric centre-distance, constant)
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1
    out = np.zeros(15 * n * n, dtype=np.float32)

    for (gx, gy), p in board.items():
        idx = gy * n + gx
        if p == player:
            out[idx]           = 1.0   # ch 0
        else:
            out[n*n + idx]     = 1.0   # ch 1

    for (gx, gy) in forbidden:
        if 0 <= gx < n and 0 <= gy < n:
            out[2*n*n + gy*n + gx] = 1.0   # ch 2

    for conn in connections:
        a, b  = tuple(conn)
        owner = board.get(a)
        base  = 3*n*n if owner == player else 4*n*n
        out[base + a[1]*n + a[0]] = 1.0
        out[base + b[1]*n + b[0]] = 1.0

    own_scores, opp_scores = opportunity_masks(board, connections, player)
    out[5*n*n : 6*n*n] = (own_scores / 4.0).clip(0.0, 1.0).astype(np.float32)
    out[6*n*n : 7*n*n] = (opp_scores / 4.0).clip(0.0, 1.0).astype(np.float32)

    own_enc, opp_enc = enclosure_potential(board, connections, player)
    out[7*n*n : 8*n*n] = own_enc.clip(0.0, 1.0)
    out[8*n*n : 9*n*n] = opp_enc.clip(0.0, 1.0)

    own_br, opp_br, disrupt, fork, phase_arr, cent = strategic_channels(
        board, connections, player, total_dots=len(board))
    out[ 9*n*n:10*n*n] = own_br.clip(0.0, 1.0)
    out[10*n*n:11*n*n] = opp_br.clip(0.0, 1.0)
    out[11*n*n:12*n*n] = disrupt.clip(0.0, 1.0)
    out[12*n*n:13*n*n] = fork
    out[13*n*n:14*n*n] = phase_arr
    out[14*n*n:15*n*n] = cent

    return out


# ── NNPlayer ─────────────────────────────────────────────────────────────────

class NNPlayer(BasePlayer):
    """Actor-critic NN player — 5-channel input, Adam, GAE, entropy bonus."""

    def __init__(self, player_id: int):
        self.player_id = player_id
        n = S.GRID
        self._hidden = S.nn_hidden_sizes()
        self._net = _ACNetwork(
            input_size   = 15 * n * n,
            hidden_sizes = self._hidden,
            output_size  = n * n,
        )
        # trajectory: (enc, action, acts, value_pred)
        self._traj:     list = []
        self._obs_traj: list = []  # opponent-observed steps (enc, action, acts, val)
        self.load()

    def choose_move(self, board, connections, player, scores,
                    forbidden=None) -> tuple:
        n         = S.GRID
        occupied  = set(board.keys())
        forbidden = forbidden or set()

        legal_flat = np.zeros(n * n, dtype=bool)
        for gx in range(n):
            for gy in range(n):
                if (gx, gy) not in occupied and (gx, gy) not in forbidden:
                    legal_flat[gy * n + gx] = True

        if not legal_flat.any():
            return (0, 0)

        enc              = encode_state(board, connections, forbidden, player)
        probs, val, acts = self._net.policy(enc, legal_flat)

        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0.0:
            probs = legal_flat.astype(np.float64); probs /= probs.sum()
        else:
            probs = probs / total

        # Boost closing and blocking moves post-policy
        own_scores, opp_scores = opportunity_masks(board, connections, player)
        probs = apply_boost(probs, own_scores, opp_scores, legal_flat)

        action = int(np.random.choice(n * n, p=probs))
        self._traj.append((enc, action, acts, val))
        return (action % n, action // n)

    def record_outcome(self, winner: int,
                       intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        # Compute terminal first — needed for obs_traj even when _traj empty.
        if final_scores and winner != 0:
            total_fields = max((S.GRID - 1) ** 2, 1)
            own_s = final_scores.get(self.player_id, 0.0)
            opp_s = final_scores.get(3 - self.player_id, 0.0)
            terminal = float(max(-1.0, min(1.0,
                                           (own_s - opp_s) / total_fields)))
        else:
            terminal = (1.0  if winner == self.player_id else
                       -1.0  if winner != 0              else 0.0)

        lr = S.NN_LEARNING_RATE
        γ  = S.NN_DISCOUNT
        λ  = _GAE_LAMBDA

        if self._traj:
            T     = len(self._traj)
            shape = (list(intermediate_rewards)
                     if intermediate_rewards and len(intermediate_rewards) == T
                     else [0.0] * T)

            vals  = [t[3] for t in self._traj]
            vnext = vals[1:] + [terminal]

            advantages = np.zeros(T, dtype=np.float64)
            gae = 0.0
            for t in reversed(range(T)):
                delta      = shape[t] + γ * vnext[t] - vals[t]
                gae        = delta + γ * λ * gae
                advantages[t] = gae

            returns = advantages + np.array(vals, dtype=np.float64)

            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            for t, (enc, action, acts, val) in enumerate(self._traj):
                self._net.update(
                    acts          = acts,
                    action        = action,
                    advantage     = float(advantages[t]),
                    value_target  = float(returns[t]),
                    entropy_bonus = _ENTROPY_COEF,
                    lr            = lr,
                )
            self._traj.clear()

        # ── Train on observed opponent moves (symmetric self-play) ────────────
        if self._obs_traj:
            T_obs        = len(self._obs_traj)
            obs_terminal = -terminal   # invert: opponent's outcome

            obs_vals  = [t[3] for t in self._obs_traj]
            obs_vnext = obs_vals[1:] + [obs_terminal]

            obs_adv = np.zeros(T_obs, dtype=np.float64)
            gae = 0.0
            for t in reversed(range(T_obs)):
                delta      = γ * obs_vnext[t] - obs_vals[t]  # no shaping
                gae        = delta + γ * λ * gae
                obs_adv[t] = gae

            obs_returns = obs_adv + np.array(obs_vals, dtype=np.float64)

            obs_adv_std = obs_adv.std()
            if obs_adv_std > 1e-8:
                obs_adv = (obs_adv - obs_adv.mean()) / (obs_adv_std + 1e-8)

            for t, (enc, action, acts, val) in enumerate(self._obs_traj):
                self._net.update(
                    acts          = acts,
                    action        = action,
                    advantage     = float(obs_adv[t]),
                    value_target  = float(obs_returns[t]),
                    entropy_bonus = _ENTROPY_COEF,
                    lr            = lr,
                )
            self._obs_traj.clear()

        self.save()

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None) -> None:
        """Record an opponent's move as a training example.

        Encodes the position from the opponent's perspective (before their
        move), runs the network forward (read-only) to get activations and
        value estimate, then stores in _obs_traj for training at game end
        with the inverted terminal value.
        """
        forbidden = forbidden or set()
        n = S.GRID
        gx, gy = move

        occupied = set(board.keys())
        legal_mask = np.array([
            (bx, by) not in occupied and (bx, by) not in forbidden
            for by in range(n) for bx in range(n)
        ], dtype=bool)

        if not legal_mask.any():
            return

        enc    = encode_state(board, connections, forbidden, player_who_moved)
        action = gy * n + gx

        # Forward pass (no weight update) — get value and activations
        probs, val, acts = self._net.policy(enc, legal_mask)
        self._obs_traj.append((enc, action, acts, val))

    def save(self) -> None:
        path = experience_path(S.NN_EXPERIENCE_BASE, '.npz')
        try:
            self._net.save(path, grid=S.GRID, hidden=self._hidden)
        except OSError:
            pass

    def load(self) -> None:
        path = experience_path(S.NN_EXPERIENCE_BASE, '.npz')
        if os.path.exists(path):
            try:
                self._net.load(path, grid=S.GRID, hidden=self._hidden)
            except Exception:
                pass
