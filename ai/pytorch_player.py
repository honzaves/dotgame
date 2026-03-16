"""PyTorch actor-critic player with PPO + residual CNN.

Architecture
------------
Input   : (5, GRID, GRID) spatial tensor
  ch 0  : own dots
  ch 1  : opponent dots
  ch 2  : forbidden positions
  ch 3  : own connections mask   (1.0 at every cell touched by an own edge)
  ch 4  : opponent connections mask

Residual trunk:
  stem  : Conv2d(5 → 64, 3×3, padding=1) → BN → ReLU
  blocks: N × ResBlock(64 channels)
    each block: Conv(64→64,3×3,p=1)→BN→ReLU→Conv(64→64,3×3,p=1)→BN + skip→ReLU

Policy head : Conv(64→2,1×1) → flatten → Linear(2·G² → G²)
Value head  : Conv(64→1,1×1) → flatten → Linear(G² → 64) → ReLU → Linear(64→1)

Training — PPO with GAE  (same algorithm as before, improved network)
"""

import os
import numpy as np
import settings as S
from ai.base_player import BasePlayer
from ai.paths import experience_path
from ai.features import (opportunity_masks, apply_boost, enclosure_potential,
                          strategic_channels)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Architecture resolved at runtime from S.pt_arch() preset
_ARCH = None   # populated lazily in PyTorchPlayer.__init__


# ── Input encoder (5 channels) ────────────────────────────────────────────────

def encode_state(board: dict, connections: set,
                 forbidden: set, player: int) -> np.ndarray:
    """Return a (15, GRID, GRID) float32 array.

    Channel 0 : own dots
    Channel 1 : opponent dots
    Channel 2 : forbidden positions
    Channel 3 : own connection mask     (cells touching any own edge)
    Channel 4 : opponent connection mask
    Channel 5 : own closing score / 4.0  (0.25 per triangle, 0.5 per full sq, up to 1.0)
    Channel 6 : opponent threat score / 4.0  (same scale — preserves magnitude)
    """
    n   = S.GRID
    opp = 2 if player == 1 else 1
    out = np.zeros((15, n, n), dtype=np.float32)

    for (gx, gy), p in board.items():
        if p == player:
            out[0, gy, gx] = 1.0
        else:
            out[1, gy, gx] = 1.0

    for (gx, gy) in forbidden:
        if 0 <= gx < n and 0 <= gy < n:
            out[2, gy, gx] = 1.0

    for conn in connections:
        a, b = tuple(conn)
        owner = board.get(a)
        ch = 3 if owner == player else 4
        out[ch, a[1], a[0]] = 1.0
        out[ch, b[1], b[0]] = 1.0

    own_scores, opp_scores = opportunity_masks(board, connections, player)
    out[5] = (own_scores / 4.0).clip(0.0, 1.0).reshape(n, n).astype(np.float32)
    out[6] = (opp_scores / 4.0).clip(0.0, 1.0).reshape(n, n).astype(np.float32)

    own_enc, opp_enc = enclosure_potential(board, connections, player)
    out[7] = own_enc.clip(0.0, 1.0).reshape(n, n)
    out[8] = opp_enc.clip(0.0, 1.0).reshape(n, n)

    own_br, opp_br, disrupt, fork, phase_arr, cent = strategic_channels(
        board, connections, player, total_dots=len(board))
    out[ 9] = own_br.clip(0.0, 1.0).reshape(n, n)
    out[10] = opp_br.clip(0.0, 1.0).reshape(n, n)
    out[11] = disrupt.clip(0.0, 1.0).reshape(n, n)
    out[12] = fork.reshape(n, n)
    out[13] = phase_arr.reshape(n, n)
    out[14] = cent.reshape(n, n)

    return out


# ── Residual block ────────────────────────────────────────────────────────────

def _build_net(grid: int, n_blocks: int, channels: int = 64):
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(ch)

        def forward(self, x):
            h = F.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return F.relu(h + x)          # skip connection

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            # Stem
            self.stem = nn.Sequential(
                nn.Conv2d(15, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )
            # Residual tower
            self.tower = nn.Sequential(
                *[ResBlock(channels) for _ in range(n_blocks)]
            )
            # Policy head
            self.pol_conv = nn.Conv2d(channels, 2, 1)
            self.pol_fc   = nn.Linear(2 * grid * grid, grid * grid)
            # Value head (two FC layers)
            self.val_conv = nn.Conv2d(channels, 1, 1)
            self.val_fc1  = nn.Linear(grid * grid, 64)
            self.val_fc2  = nn.Linear(64, 1)

        def forward(self, x):
            """x: (B, 5, G, G) → (logits (B, G²), values (B,))"""
            h = self.tower(self.stem(x))

            # Policy
            p = F.relu(self.pol_conv(h)).flatten(1)
            logits = self.pol_fc(p)

            # Value
            v = F.relu(self.val_conv(h)).flatten(1)
            v = F.relu(self.val_fc1(v))
            values = self.val_fc2(v).squeeze(-1)

            return logits, values

    return _Net()


# ── PyTorchPlayer ─────────────────────────────────────────────────────────────

class PyTorchPlayer(BasePlayer):

    def __init__(self, player_id: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.  Run:  pip install torch")
        self.player_id = player_id
        self._device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch           = S.pt_arch()
        self._arch     = {**arch, 'in_channels': 15}  # includes channel count for compat check
        self._net      = _build_net(
            S.GRID, arch['blocks'], arch['channels']
        ).to(self._device)
        self._opt      = torch.optim.Adam(
            self._net.parameters(), lr=S.PT_LEARNING_RATE
        )
        # Trajectory: (enc_np, mask_np, action, log_prob_old, value_old)
        self._traj:     list[tuple] = []
        self._obs_traj: list[tuple] = []  # opponent-observed (board, action, log_p, val)
        self.load()

    # ── BasePlayer interface ──────────────────────────────────────────────────

    def choose_move(self, board, connections, player, scores,
                    forbidden=None) -> tuple[int, int]:
        n        = S.GRID
        occupied = set(board.keys())
        forbidden = forbidden or set()

        mask_np = np.zeros(n * n, dtype=bool)
        for gx in range(n):
            for gy in range(n):
                if (gx, gy) not in occupied and (gx, gy) not in forbidden:
                    mask_np[gy * n + gx] = True

        if not mask_np.any():
            return (0, 0)

        enc_np = encode_state(board, connections, forbidden, player)

        with torch.no_grad():
            x    = torch.tensor(enc_np, dtype=torch.float32,
                                device=self._device).unsqueeze(0)   # (1,7,G,G)
            mask = torch.tensor(mask_np, dtype=torch.bool,
                                device=self._device)

            logits, val = self._net(x)
            logits = logits[0].clone()
            logits[~mask] = float('-inf')
            log_p  = F.log_softmax(logits, dim=0)
            probs  = log_p.exp()

            p_np = probs.cpu().numpy().astype(np.float64)
            p_np = np.where(np.isfinite(p_np), np.clip(p_np, 0.0, None), 0.0)
            total = p_np.sum()
            if total <= 0.0:
                p_np = mask_np.astype(np.float64); p_np /= p_np.sum()
            else:
                p_np /= total

            # Boost closing and blocking moves post-policy
            own_scores, opp_scores = opportunity_masks(board, connections, player)
            p_np = apply_boost(p_np, own_scores, opp_scores, mask_np)

            action       = int(np.random.choice(n * n, p=p_np))
            log_prob_old = float(log_p[action].item())
            value_old    = float(val[0].item())

        self._traj.append((enc_np, mask_np, action, log_prob_old, value_old))
        return (action % n, action // n)

    def record_outcome(self, winner: int,
                       intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        # Compute terminal value first — needed for obs_traj even when _traj
        # is empty (e.g. player never moved in a very short game).
        if final_scores and winner != 0:
            total_fields = max((S.GRID - 1) ** 2, 1)
            own_s  = final_scores.get(self.player_id, 0.0)
            opp_s  = final_scores.get(3 - self.player_id, 0.0)
            terminal = float(max(-1.0, min(1.0,
                                           (own_s - opp_s) / total_fields)))
        else:
            terminal = (1.0  if winner == self.player_id else
                       -1.0  if winner != 0              else 0.0)

        if self._traj:
            T = len(self._traj)

            shape = (list(intermediate_rewards)
                     if intermediate_rewards and len(intermediate_rewards) == T
                     else [0.0] * T)

            # GAE
            γ, λ   = S.PT_DISCOUNT, S.PT_GAE_LAMBDA
            vals   = [t[4] for t in self._traj]
            vnext  = vals[1:] + [terminal]

            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta      = shape[t] + γ * vnext[t] - vals[t]
                gae        = delta + γ * λ * gae
                advantages[t] = gae

            returns_np = advantages + np.array(vals, dtype=np.float32)

            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            n = S.GRID
            states_t  = torch.tensor(
                np.stack([t[0] for t in self._traj]),
                dtype=torch.float32, device=self._device)          # (T,5,G,G)
            masks_t   = torch.tensor(
                np.stack([t[1] for t in self._traj]),
                dtype=torch.bool, device=self._device)
            actions_t = torch.tensor(
                [t[2] for t in self._traj],
                dtype=torch.long, device=self._device)
            lp_old_t  = torch.tensor(
                [t[3] for t in self._traj],
                dtype=torch.float32, device=self._device)
            adv_t     = torch.tensor(advantages, dtype=torch.float32,
                                     device=self._device)
            ret_t     = torch.tensor(returns_np, dtype=torch.float32,
                                     device=self._device)

            # PPO epochs
            idx  = np.arange(T)
            mb   = min(S.PT_PPO_MINIBATCH, T)
            clip = S.PT_PPO_CLIP

            for _ in range(S.PT_PPO_EPOCHS):
                np.random.shuffle(idx)
                for start in range(0, T, mb):
                    batch = idx[start:start + mb]
                    if len(batch) == 0:
                        continue

                    b_states  = states_t[batch]
                    b_masks   = masks_t[batch]
                    b_actions = actions_t[batch]
                    b_lp_old  = lp_old_t[batch]
                    b_adv     = adv_t[batch]
                    b_ret     = ret_t[batch]

                    logits, vals_pred = self._net(b_states)

                    logits = logits.clone()
                    logits[~b_masks] = float('-inf')
                    log_p_all = F.log_softmax(logits, dim=1)
                    log_p     = log_p_all.gather(
                        1, b_actions.unsqueeze(1)).squeeze(1)

                    ratio  = torch.exp(log_p - b_lp_old)
                    surr1  = ratio * b_adv
                    surr2  = torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv
                    pol_loss = -torch.min(surr1, surr2).mean()

                    val_loss = F.mse_loss(vals_pred, b_ret)

                    probs_all = log_p_all.exp()
                    entropy   = -(probs_all * log_p_all).sum(dim=1).mean()

                    loss = (pol_loss
                            + S.PT_VALUE_COEF   * val_loss
                            - S.PT_ENTROPY_COEF * entropy)

                    self._opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self._net.parameters(), S.PT_MAX_GRAD_NORM)
                    self._opt.step()

            self._traj.clear()

        # ── Train on observed opponent moves (symmetric self-play) ────────────
        if self._obs_traj:
            T_obs = len(self._obs_traj)
            obs_terminal = -terminal   # opponent's perspective: inverted

            γ, λ = S.PT_DISCOUNT, S.PT_GAE_LAMBDA
            obs_vals  = [t[4] for t in self._obs_traj]
            obs_vnext = obs_vals[1:] + [obs_terminal]

            obs_adv = np.zeros(T_obs, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T_obs)):
                # No shaping for observations — only the terminal value matters
                delta      = γ * obs_vnext[t] - obs_vals[t]
                gae        = delta + γ * λ * gae
                obs_adv[t] = gae

            obs_ret = obs_adv + np.array(obs_vals, dtype=np.float32)

            obs_adv_std = obs_adv.std()
            if obs_adv_std > 1e-8:
                obs_adv = (obs_adv - obs_adv.mean()) / (obs_adv_std + 1e-8)

            n = S.GRID
            obs_states_t  = torch.tensor(
                np.stack([t[0] for t in self._obs_traj]),
                dtype=torch.float32, device=self._device)
            obs_masks_t   = torch.tensor(
                np.stack([t[1] for t in self._obs_traj]),
                dtype=torch.bool, device=self._device)
            obs_actions_t = torch.tensor(
                [t[2] for t in self._obs_traj],
                dtype=torch.long, device=self._device)
            obs_lp_old_t  = torch.tensor(
                [t[3] for t in self._obs_traj],
                dtype=torch.float32, device=self._device)
            obs_adv_t     = torch.tensor(obs_adv, dtype=torch.float32,
                                         device=self._device)
            obs_ret_t     = torch.tensor(obs_ret, dtype=torch.float32,
                                         device=self._device)

            # One PPO pass over observed data
            idx_obs  = np.arange(T_obs)
            mb_obs   = min(S.PT_PPO_MINIBATCH, T_obs)
            clip     = S.PT_PPO_CLIP
            np.random.shuffle(idx_obs)
            for start in range(0, T_obs, mb_obs):
                batch = idx_obs[start:start + mb_obs]
                if len(batch) == 0:
                    continue

                b_states  = obs_states_t[batch]
                b_masks   = obs_masks_t[batch]
                b_actions = obs_actions_t[batch]
                b_lp_old  = obs_lp_old_t[batch]
                b_adv     = obs_adv_t[batch]
                b_ret     = obs_ret_t[batch]

                logits, vals_pred = self._net(b_states)
                logits = logits.clone()
                logits[~b_masks] = float('-inf')
                log_p_all = F.log_softmax(logits, dim=1)
                log_p     = log_p_all.gather(
                    1, b_actions.unsqueeze(1)).squeeze(1)

                ratio    = torch.exp(log_p - b_lp_old)
                surr1    = ratio * b_adv
                surr2    = torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(vals_pred, b_ret)
                probs_all = log_p_all.exp()
                entropy   = -(probs_all * log_p_all).sum(dim=1).mean()

                loss = (pol_loss
                        + S.PT_VALUE_COEF   * val_loss
                        - S.PT_ENTROPY_COEF * entropy)

                self._opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._net.parameters(), S.PT_MAX_GRAD_NORM)
                self._opt.step()

            self._obs_traj.clear()

        self.save()

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None) -> None:
        """Record an opponent's move as a training example.

        Encodes the position from the opponent's perspective (before their
        move), runs a forward pass to get the log-probability of their action
        and their value estimate, then stores the tuple in _obs_traj.
        At record_outcome these steps are trained with the inverted terminal.
        """
        forbidden = forbidden or set()
        n = S.GRID
        gx, gy = move

        occupied = set(board.keys())
        mask_np  = np.zeros(n * n, dtype=bool)
        for bx in range(n):
            for by in range(n):
                if (bx, by) not in occupied and (bx, by) not in forbidden:
                    mask_np[by * n + bx] = True

        if not mask_np.any():
            return

        enc = encode_state(board, connections, forbidden, player_who_moved)
        action = gy * n + gx

        with torch.no_grad():
            x      = torch.tensor(enc, dtype=torch.float32,
                                  device=self._device).unsqueeze(0)
            mask_t = torch.tensor(mask_np, dtype=torch.bool,
                                  device=self._device)
            logits, val = self._net(x)
            logits = logits[0].clone()
            logits[~mask_t] = float('-inf')
            log_p  = F.log_softmax(logits, dim=0)
            log_prob_old = float(log_p[action].item())
            value_old    = float(val[0].item())

        self._obs_traj.append((enc, mask_np, action, log_prob_old, value_old))

    def save(self) -> None:
        path = experience_path(S.PT_EXPERIENCE_BASE, '.pt')
        try:
            torch.save({
                'model':  self._net.state_dict(),
                'optim':  self._opt.state_dict(),
                'grid':   S.GRID,
                'arch':   self._arch,
            }, path)
        except OSError:
            pass

    def load(self) -> None:
        path = experience_path(S.PT_EXPERIENCE_BASE, '.pt')
        if not os.path.exists(path):
            return
        try:
            ck = torch.load(path, map_location=self._device,
                            weights_only=False)
            if (ck.get('grid') == S.GRID
                    and ck.get('arch') == self._arch):
                self._net.load_state_dict(ck['model'])
                self._opt.load_state_dict(ck['optim'])
        except Exception:
            pass
