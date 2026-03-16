"""Neural MCTS player — AlphaZero-style search guided by a residual CNN.

How it differs from the three other AI types
---------------------------------------------
Pure MCTS (mcts.py)          — hand-crafted priors, random rollouts, no training
Neural Network (nn_player)   — learned policy + value, no lookahead at all
PyTorch Net (pytorch_player) — learned policy + value (CNN), no lookahead at all
Neural MCTS (this file)      — learned policy + value (CNN) + MCTS tree search

The CNN guides the tree in two ways:
  policy head  → PUCT prior for each child   (replaces the hand-crafted raw_priors formula)
  value head   → leaf-node evaluation        (replaces _eval_fast + random rollouts)

This creates the AlphaZero self-improvement loop:
  stronger network  →  better priors + leaf values
  better search     →  better training data (visit distributions, outcomes)
  better data       →  stronger network   →  ...

Training signal per game
  policy target  : MCTS visit distribution (not the raw network output)
                   The tree's opinion of the best move is more accurate than the network's
                   raw prior, especially with many simulations.  Training on visit counts
                   directly teaches the network to replicate what the tree found.
  value target   : GAE returns from territory-delta + arc-potential-delta shaping rewards
                   (same shaping as PyTorchPlayer — teaches ring-building from early training)

Architecture
  Identical residual CNN to PyTorchPlayer (shared presets, separate weight file).
  Training loss:  L = policy_coef · CE(π_mcts, π_net)
                    + value_coef  · MSE(v_pred, v_target)
                    − entropy_coef· H(π_net)
  No PPO needed — the policy target is a stable supervised signal from the tree,
  not a self-bootstrapped value estimate.

Requirements: pip install torch
"""

import math
import os
import time

import numpy as np

import settings as S
from ai.base_player import BasePlayer
from ai.paths import experience_path
from ai.features import opportunity_masks, arc_potential_map

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# ── Tree node ─────────────────────────────────────────────────────────────────

class _NMNode:
    """Tree node for Neural MCTS.

    children = None   →  never visited; will be expanded on first visit
    children = []     →  expanded (may be empty only at true terminal states)
    """

    __slots__ = ('move', 'parent', 'children', 'visits', 'total_value',
                 'player_who_moved', 'prior')

    def __init__(self, move, parent, children, player_who_moved, prior=0.0):
        self.move             = move
        self.parent           = parent
        self.children         = children  # None | list[_NMNode]
        self.visits           = 0
        self.total_value      = 0.0       # sum of backpropagated values
        self.player_who_moved = player_who_moved
        self.prior            = prior     # P(s,a) from parent's network evaluation

    def puct(self, c_puct: float) -> float:
        """AlphaZero PUCT:  Q(s,a)  +  c_puct · P(s,a) · √N(s) / (1 + N(s,a))

        Q is the average value from this node's perspective (fraction of visits
        that were wins for the AI player who owns this search tree).
        U is the exploration bonus — high for unvisited children with a strong prior.
        """
        parent_n = self.parent.visits if self.parent else 1
        q = self.total_value / self.visits if self.visits > 0 else 0.0
        u = c_puct * self.prior * math.sqrt(max(parent_n, 1)) / (1 + self.visits)
        return q + u


# ── Helpers that reuse PyTorchPlayer internals ────────────────────────────────

def _build_nm_net(grid: int, n_blocks: int, channels: int):
    """Same residual CNN architecture as PyTorchPlayer — no code duplication."""
    from ai.pytorch_player import _build_net
    return _build_net(grid, n_blocks, channels)


def _encode(board: dict, connections: set, forbidden: set, player: int):
    """15-channel spatial encoding — identical to pytorch_player.encode_state."""
    from ai.pytorch_player import encode_state
    return encode_state(board, connections, forbidden, player)


# ── NeuralMCTSPlayer ──────────────────────────────────────────────────────────

class NeuralMCTSPlayer(BasePlayer):
    """AlphaZero-style MCTS guided by a residual CNN.

    Starts weak (untrained network ≈ random priors) and improves continuously
    through self-play or curriculum training.  Given enough training games it
    surpasses pure MCTS at the same time budget:

      - The policy head focuses the search on promising moves instead of
        spreading simulations across all legal moves equally.
      - The value head replaces imprecise short rollouts with a direct
        position evaluation, allowing far more tree nodes per second.
      - Training on MCTS visit distributions causes the network to learn
        *what the tree found*, not just what the network originally guessed.
    """

    def __init__(self, player_id: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.  Run:  pip install torch")

        self.player_id = player_id
        self._device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch           = S.nm_arch()
        self._arch     = {**arch, 'in_channels': 15}  # stored for load-compat check
        self._net      = _build_nm_net(
            S.GRID, arch['blocks'], arch['channels']
        ).to(self._device)
        self._opt      = torch.optim.Adam(
            self._net.parameters(), lr=S.NM_LEARNING_RATE
        )

        # Episode buffer — one entry per move this player made this game:
        #   (enc_np, legal_mask_flat, visit_dist_flat, root_value_pred)
        self._episode:     list[tuple] = []
        # Observed opponent moves — same tuple format but encoded from the
        # opponent's perspective; trained with inverted terminal value.
        self._obs_episode: list[tuple] = []
        self.load()

    # ── BasePlayer interface ──────────────────────────────────────────────────

    def choose_move(self, board: dict, connections: set, player: int,
                    scores: dict, forbidden=None) -> tuple[int, int]:
        from ai.mcts import FastBoard

        forbidden = forbidden or set()
        fb        = FastBoard(board, connections, player, scores, forbidden)
        legal     = fb.legal_moves()
        n         = S.GRID

        if not legal:
            return (0, 0)

        # ── Encode board and get network output at root ───────────────────────
        enc          = _encode(board, connections, forbidden, player)
        legal_mask   = np.zeros(n * n, dtype=bool)
        for gx, gy in legal:
            legal_mask[gy * n + gx] = True

        with torch.no_grad():
            x      = torch.tensor(enc, dtype=torch.float32,
                                  device=self._device).unsqueeze(0)
            rl, rv = self._net(x)
            rl     = rl[0].clone()
            mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=self._device)
            rl[~mask_t] = float('-inf')
            root_priors = F.softmax(rl, dim=0).cpu().numpy()   # (G²,) ≈1 over legal

        root_val = float(rv[0].item())   # in [-1,1], from player's perspective

        if len(legal) == 1:
            # Only one move — record and return immediately (no search needed)
            visit_dist = np.zeros(n * n, dtype=np.float32)
            gx, gy = legal[0]
            visit_dist[gy * n + gx] = 1.0
            self._episode.append((enc, legal_mask, visit_dist, root_val))
            return legal[0]

        # ── Tactical boost on root priors (safety net for untrained network) ──
        # Exponential formula: exp(k × score) where k = 2·ln(close_boost).
        # This gives close_boost^(2×score), so:
        #   score 0.5  →  close_boost^1  ≈   4×
        #   score 1.0  →  close_boost^2  ≈  16×
        #   score 2.5  →  close_boost^5  ≈ 1024×
        # Even a network prior 100× biased toward a triangle cannot override a
        # 2.5-point square opportunity (1/100 × 1024 > 1 × 4).
        # Also suppresses dead spots — positions with zero immediate or ring
        # potential — but only once productive moves exist on the board.
        root_priors = self._tactical_priors(
            root_priors, board, connections, player, legal_mask,
            suppress_dead=True)

        # ── Dirichlet noise for exploration ───────────────────────────────────
        # Prevents deterministic play and ensures diversity in training data.
        # Adaptive alpha keeps spikiness consistent regardless of board size.
        if S.NM_NOISE_FRAC > 0 and len(legal) > 1:
            alpha = max(0.01, S.NM_NOISE_ALPHA * 10.0 / len(legal))
            eta   = np.random.dirichlet([alpha] * len(legal))
            for i, (gx, gy) in enumerate(legal):
                idx              = gy * n + gx
                root_priors[idx] = ((1.0 - S.NM_NOISE_FRAC) * root_priors[idx]
                                    + S.NM_NOISE_FRAC * float(eta[i]))

        # ── Build root with all direct children pre-populated ─────────────────
        # AlphaZero-style: children already exist with priors from root network
        # call.  On the first simulation each child immediately gets expanded
        # (another network call) to populate its own children.
        root = _NMNode(move=None, parent=None, children=[],
                       player_who_moved=3 - player, prior=0.0)
        for gx, gy in legal:
            child = _NMNode(
                move=(gx, gy), parent=root, children=None,
                player_who_moved=player,              # player moved to reach child
                prior=float(root_priors[gy * n + gx])
            )
            root.children.append(child)

        # ── MCTS simulations ──────────────────────────────────────────────────
        deadline = time.time() + S.AI_THINK_MS / 1000.0

        while time.time() < deadline:
            node, sim_fb = self._select(root, fb.copy())

            if sim_fb.done:
                # Terminal state: use actual score ratio
                s1, s2 = sim_fb.scores.get(1, 0.0), sim_fb.scores.get(2, 0.0)
                total  = (s1 + s2) or 1.0
                val01  = (s1 if player == 1 else s2) / total
            elif node.children is None:
                # First visit to this node → expand and evaluate with network
                raw_v = self._expand_node(node, sim_fb)
                # raw_v ∈ [-1,1] from sim_fb.player's perspective;
                # convert to [0,1] from ai_player (= player) perspective
                if sim_fb.player == player:
                    val01 = (raw_v + 1.0) / 2.0
                else:
                    val01 = (1.0 - raw_v) / 2.0
            else:
                # All children visited but no legal moves: shouldn't happen
                val01 = 0.5

            self._backprop(node, val01, player)

        # ── Visit distribution for policy training ────────────────────────────
        visit_dist = np.zeros(n * n, dtype=np.float32)
        total_vis  = max(sum(c.visits for c in root.children), 1)
        for child in root.children:
            gx, gy = child.move
            visit_dist[gy * n + gx] = child.visits / total_vis

        self._episode.append((enc, legal_mask, visit_dist, root_val))

        # ── Final move: most-visited child ────────────────────────────────────
        # After many simulations visit counts are much more reliable than
        # a single network query, so we always use argmax(visits).
        best = max(root.children, key=lambda c: c.visits)
        return best.move

    def record_outcome(self, winner: int,
                       intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        if not self._episode:
            return

        T = len(self._episode)

        # ── Terminal value in [-1, 1] ─────────────────────────────────────────
        if final_scores and winner != 0:
            total_fields = max((S.GRID - 1) ** 2, 1)
            own_s  = final_scores.get(self.player_id, 0.0)
            opp_s  = final_scores.get(3 - self.player_id, 0.0)
            terminal = float(max(-1.0, min(1.0, (own_s - opp_s) / total_fields)))
        else:
            terminal = (1.0 if winner == self.player_id else
                       -1.0 if winner != 0               else 0.0)

        shape = (list(intermediate_rewards)
                 if intermediate_rewards and len(intermediate_rewards) == T
                 else [0.0] * T)

        # ── GAE (same formula as PyTorchPlayer) ───────────────────────────────
        γ, λ  = S.NM_DISCOUNT, S.NM_GAE_LAMBDA
        vals  = [t[3] for t in self._episode]          # root value predictions
        vnext = vals[1:] + [terminal]

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta         = shape[t] + γ * vnext[t] - vals[t]
            gae           = delta + γ * λ * gae
            advantages[t] = gae

        returns_np = advantages + np.array(vals, dtype=np.float32)

        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)  # noqa (unused after this)

        # ── Build tensors ─────────────────────────────────────────────────────
        states_t  = torch.tensor(
            np.stack([t[0] for t in self._episode]),
            dtype=torch.float32, device=self._device)           # (T, 15, G, G)
        masks_t   = torch.tensor(
            np.stack([t[1] for t in self._episode]),
            dtype=torch.bool, device=self._device)              # (T, G²)
        pi_mcts_t = torch.tensor(
            np.stack([t[2] for t in self._episode]),
            dtype=torch.float32, device=self._device)           # (T, G²)  visit dist
        ret_t     = torch.tensor(returns_np,
                                 dtype=torch.float32, device=self._device)  # (T,)

        # ── Single gradient update ────────────────────────────────────────────
        # AlphaZero does not use PPO: the policy target (MCTS visit distribution)
        # is an external supervised signal, not a self-bootstrapped estimate.
        # One Adam step per game is stable and correct.
        logits, vals_pred = self._net(states_t)
        logits_m = logits.clone()
        logits_m[~masks_t] = float('-inf')
        log_probs = F.log_softmax(logits_m, dim=1)      # (T, G²)
        probs     = log_probs.exp()

        # Cross-entropy: teach the network to replicate what the tree found.
        # Illegal positions have log_probs = -inf and pi_mcts_t = 0.0, but
        # 0.0 * -inf = nan in IEEE 754, which would poison the entire loss.
        # Zero out log_probs at illegal positions before the dot product —
        # those positions contribute 0 to the sum anyway.
        safe_log_probs = log_probs.clone()
        safe_log_probs[~masks_t] = 0.0
        pol_loss = -(pi_mcts_t * safe_log_probs).sum(dim=1).mean()

        # MSE: teach the network to predict what the return will be
        val_loss = F.mse_loss(vals_pred, ret_t)

        # Entropy bonus: keep exploration alive, especially early in training.
        # Same 0 * -inf = nan hazard as pol_loss: illegal positions have
        # probs=0 and log_probs=-inf. Use safe_log_probs (zeroed at illegal
        # positions) so the product is always 0 * 0 = 0 there.
        entropy  = -(probs * safe_log_probs).sum(dim=1).mean()

        loss = (S.NM_POLICY_COEF  * pol_loss
                + S.NM_VALUE_COEF * val_loss
                - S.NM_ENTROPY_COEF * entropy)

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), S.PT_MAX_GRAD_NORM)
        self._opt.step()

        self._episode.clear()

        # ── Train on observed opponent moves (symmetric self-play) ────────────
        if self._obs_episode:
            T_obs = len(self._obs_episode)
            # Opponent's terminal is the inverse of ours.
            obs_terminal = -terminal

            # GAE with no shaping (we don't have opponent's intermediate rewards)
            obs_vals  = [t[3] for t in self._obs_episode]
            obs_vnext = obs_vals[1:] + [obs_terminal]

            obs_advantages = np.zeros(T_obs, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T_obs)):
                delta          = γ * obs_vnext[t] - obs_vals[t]
                gae            = delta + γ * λ * gae
                obs_advantages[t] = gae

            obs_returns = obs_advantages + np.array(obs_vals, dtype=np.float32)

            obs_adv_std = obs_advantages.std()
            if obs_adv_std > 1e-8:
                obs_advantages = ((obs_advantages - obs_advantages.mean())
                                  / (obs_adv_std + 1e-8))

            obs_states_t = torch.tensor(
                np.stack([t[0] for t in self._obs_episode]),
                dtype=torch.float32, device=self._device)
            obs_masks_t  = torch.tensor(
                np.stack([t[1] for t in self._obs_episode]),
                dtype=torch.bool, device=self._device)
            obs_pi_t     = torch.tensor(
                np.stack([t[2] for t in self._obs_episode]),
                dtype=torch.float32, device=self._device)
            obs_ret_t    = torch.tensor(
                obs_returns, dtype=torch.float32, device=self._device)

            obs_logits, obs_vals_pred = self._net(obs_states_t)
            obs_logits_m = obs_logits.clone()
            obs_logits_m[~obs_masks_t] = float('-inf')
            obs_log_probs = F.log_softmax(obs_logits_m, dim=1)
            obs_probs     = obs_log_probs.exp()

            obs_safe_lp = obs_log_probs.clone()
            obs_safe_lp[~obs_masks_t] = 0.0

            obs_pol_loss = -(obs_pi_t * obs_safe_lp).sum(dim=1).mean()
            obs_val_loss = F.mse_loss(obs_vals_pred, obs_ret_t)
            obs_entropy  = -(obs_probs * obs_safe_lp).sum(dim=1).mean()

            obs_loss = (S.NM_POLICY_COEF   * obs_pol_loss
                        + S.NM_VALUE_COEF  * obs_val_loss
                        - S.NM_ENTROPY_COEF * obs_entropy)

            self._opt.zero_grad()
            obs_loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), S.PT_MAX_GRAD_NORM)
            self._opt.step()

            self._obs_episode.clear()

        self.save()

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None) -> None:
        """Record an opponent's move as a training example.

        Encodes the position from the opponent's perspective (before their
        move), uses a one-hot policy target on their chosen action, and
        stores the network's own value estimate for GAE bootstrapping.
        At record_outcome these are trained with the inverted terminal value
        so the network learns to evaluate positions from both sides.
        """
        forbidden = forbidden or set()
        n = S.GRID
        gx, gy = move

        # Build legal mask from the position BEFORE the opponent's move,
        # encoded from the opponent's perspective.
        occupied = set(board.keys())
        mask_np  = np.zeros(n * n, dtype=bool)
        for bx in range(n):
            for by in range(n):
                if (bx, by) not in occupied and (bx, by) not in forbidden:
                    mask_np[by * n + bx] = True

        if not mask_np.any():
            return  # nothing legal — don't record

        enc = _encode(board, connections, forbidden, player_who_moved)

        with torch.no_grad():
            x      = torch.tensor(enc, dtype=torch.float32,
                                  device=self._device).unsqueeze(0)
            rl, rv = self._net(x)
            raw_v  = float(rv[0].item())  # [-1,1] from player_who_moved's POV

        # One-hot visit distribution: all weight on the move that was played
        visit_dist = np.zeros(n * n, dtype=np.float32)
        visit_dist[gy * n + gx] = 1.0

        self._obs_episode.append((enc, mask_np, visit_dist, raw_v))

    def save(self) -> None:
        path = experience_path(S.NM_EXPERIENCE_BASE, '.pt')
        try:
            torch.save({
                'model': self._net.state_dict(),
                'optim': self._opt.state_dict(),
                'grid':  S.GRID,
                'arch':  self._arch,
            }, path)
        except OSError:
            pass

    def load(self) -> None:
        path = experience_path(S.NM_EXPERIENCE_BASE, '.pt')
        if not os.path.exists(path):
            return
        try:
            ck = torch.load(path, map_location=self._device, weights_only=False)
            if (ck.get('grid') == S.GRID and ck.get('arch') == self._arch):
                self._net.load_state_dict(ck['model'])
                self._opt.load_state_dict(ck['optim'])
        except Exception:
            pass

    # ── Internal MCTS helpers ─────────────────────────────────────────────────

    def _select(self, root: _NMNode, fb) -> tuple:
        """Descend tree following PUCT until an unvisited (children=None) node."""
        node = root
        while not fb.done:
            if node.children is None:
                # Unvisited leaf — stop here for expansion
                return node, fb
            if not node.children:
                # Expanded but empty (terminal detected during expansion)
                return node, fb
            # Descend to highest-PUCT child
            best = max(node.children, key=lambda c: c.puct(S.NM_C_PUCT))
            fb.play(*best.move)
            node = best
        return node, fb

    def _expand_node(self, node: _NMNode, fb) -> float:
        """First visit to node: call network to get policy + value.

        Populates node.children with priors from the policy head.
        Returns raw value in [-1, 1] from fb.player's perspective.
        """
        legal = fb.legal_moves()
        if not legal:
            node.children = []
            return 0.0

        enc = _encode(fb.board, fb.connections, fb.forbidden, fb.player)
        with torch.no_grad():
            x  = torch.tensor(enc, dtype=torch.float32,
                               device=self._device).unsqueeze(0)
            lg, v = self._net(x)
            lg    = lg[0].clone()
            n     = S.GRID
            mask  = torch.zeros(n * n, dtype=torch.bool, device=self._device)
            for gx, gy in legal:
                mask[gy * n + gx] = True
            lg[~mask] = float('-inf')
            priors = F.softmax(lg, dim=0).cpu().numpy()

        # Apply tactical correction inside the tree too — otherwise every
        # simulation uses raw network priors that may strongly favour useless
        # moves over obvious immediate scores deep in the search.
        # Skip dead-spot suppression here for speed (arc_potential_map is O(G²)).
        legal_mask_np = mask.cpu().numpy()
        priors = self._tactical_priors(
            priors, fb.board, fb.connections, fb.player, legal_mask_np,
            suppress_dead=False)

        node.children = [
            _NMNode(move=(gx, gy), parent=node, children=None,
                    player_who_moved=fb.player,
                    prior=float(priors[gy * n + gx]))
            for gx, gy in legal
        ]
        return float(v[0].item())

    @staticmethod
    def _tactical_priors(
        probs: np.ndarray,
        board: dict,
        connections: set,
        player: int,
        legal_mask: np.ndarray,
        suppress_dead: bool = True,
    ) -> np.ndarray:
        """Tactically-corrected priors for MCTS nodes.

        Two improvements over the old linear apply_boost():

        1. Exponential scoring boost
           Old formula:  1 + (base−1) × score   (linear)
             score=0.5 → 2.5×,  score=2.5 → 8.5×
           New formula:  exp(2·ln(base) × score) = base^(2×score)
             score=0.5 → base^1 ≈  4×
             score=1.0 → base^2 ≈ 16×
             score=2.5 → base^5 ≈ 1024×
           A network prior 100× biased toward a 0.5-pt triangle cannot beat
           a 2.5-pt square: (1/100)×1024 = 10.2  vs  1.0×4 = 4.

        2. Dead-spot suppression (suppress_dead=True, root only)
           Positions with zero immediate scoring value AND zero ring-arc
           potential are divided by NM_DEAD_SPOT_PENALTY.  This is only
           applied when productive moves already exist on the board, so an
           empty board is not affected.
        """
        own_s, opp_s = opportunity_masks(board, connections, player)

        p        = probs.astype(np.float64)
        own_leg  = own_s.astype(np.float64) * legal_mask
        opp_leg  = opp_s.astype(np.float64) * legal_mask

        close_k = 2.0 * math.log(max(S.AI_CLOSE_BOOST, 2.0))
        block_k = 2.0 * math.log(max(S.AI_BLOCK_BOOST, 2.0))

        if own_leg.any():
            p *= np.exp(close_k * own_leg)
        if opp_leg.any():
            p *= np.exp(block_k * opp_leg)

        if suppress_dead:
            own_arc, opp_arc = arc_potential_map(board, player)
            # A position is "live" if it has direct scoring value OR is
            # adjacent to an existing ring arc (contributing to a future close).
            live = (
                (own_leg > 0) | (opp_leg > 0) |
                ((own_arc > 0) & legal_mask) |
                ((opp_arc > 0) & legal_mask)
            )
            # Only suppress when productive moves already exist — avoids
            # penalising every position on a nearly-empty board.
            if live.any():
                dead = legal_mask.astype(bool) & ~live
                if dead.any():
                    p[dead] /= S.NM_DEAD_SPOT_PENALTY

        total = p[legal_mask.astype(bool)].sum()
        if total > 0:
            p /= total
        else:
            p = legal_mask.astype(np.float64)
            p /= p.sum()

        return p.astype(np.float32)

    @staticmethod
    def _backprop(node: _NMNode, val01: float, ai_player: int) -> None:
        """Propagate val01 ∈ [0,1] (from ai_player's perspective) up the tree.

        Nodes where ai_player moved get +val01 (good position for ai_player).
        Nodes where the opponent moved get +(1−val01) (good for them = bad for us).
        """
        while node is not None:
            node.visits += 1
            if node.player_who_moved == ai_player:
                node.total_value += val01
            else:
                node.total_value += (1.0 - val01)
            node = node.parent
