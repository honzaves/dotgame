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
from ai.features import opportunity_masks, arc_potential_map, adaptive_dirichlet_alpha
from ai.symmetry import transform_planes, transform_flat, N_TRANSFORMS

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
        self._arch     = {**arch, 'in_channels': 15, 'norm': 'gn'}
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
        # Replay buffer of (enc, legal_mask, visit_dist, value_target) — the
        # AlphaZero training store.  In-memory only; cleared on restart.
        self._buffer: list[tuple] = []
        self._learning_shared: bool = False
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
        game_phase = len(board) / max(n * n, 1)
        root_priors = self._tactical_priors(
            root_priors, board, connections, player, legal_mask,
            suppress_dead=True, game_phase=game_phase)

        # ── Dirichlet noise for exploration ───────────────────────────────────
        # Prevents deterministic play and ensures diversity in training data.
        # Adaptive alpha keeps spikiness consistent regardless of board size;
        # floored at 0.1 to avoid single-spike noise on large open boards.
        if S.NM_NOISE_FRAC > 0 and len(legal) > 1:
            alpha = adaptive_dirichlet_alpha(S.NM_NOISE_ALPHA, len(legal))
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
        if not self._episode and not self._obs_episode:
            return

        # ── Terminal value in [-1, 1] ─────────────────────────────────────────
        if final_scores and winner != 0:
            total_fields = max((S.GRID - 1) ** 2, 1)
            own_s  = final_scores.get(self.player_id, 0.0)
            opp_s  = final_scores.get(3 - self.player_id, 0.0)
            terminal = float(max(-1.0, min(1.0, (own_s - opp_s) / total_fields)))
        else:
            terminal = (1.0 if winner == self.player_id else
                       -1.0 if winner != 0               else 0.0)

        # ── Append episodes to the replay buffer ─────────────────────────────
        if self._episode:
            targets = self._value_targets(
                [t[3] for t in self._episode], intermediate_rewards, terminal)
            for (enc, mask, dist, _), tgt in zip(self._episode, targets):
                self._buffer.append((enc, mask, dist, float(tgt)))
            self._episode.clear()

        if self._obs_episode:
            # Opponent's perspective: inverted terminal, no shaping rewards.
            obs_targets = self._value_targets(
                [t[3] for t in self._obs_episode], None, -terminal)
            for (enc, mask, dist, _), tgt in zip(self._obs_episode, obs_targets):
                self._buffer.append((enc, mask, dist, float(tgt)))
            self._obs_episode.clear()

        if len(self._buffer) > S.NM_BUFFER_MAX:
            del self._buffer[:len(self._buffer) - S.NM_BUFFER_MAX]

        self._train_from_buffer()
        self.save()

    def _value_targets(self, vals: list, shape, terminal: float) -> np.ndarray:
        """Per-step value targets: GAE returns blended with the final outcome.

        Pure GAE bootstraps through the network's own (initially wrong) value
        estimates; pure z (AlphaZero) ignores shaping.  The NM_Z_WEIGHT blend
        gives early ring-building moves direct credit for the final margin
        while keeping the dense shaping signal.
        """
        T = len(vals)
        shape = list(shape) if (shape and len(shape) == T) else [0.0] * T
        γ, λ  = S.NM_DISCOUNT, S.NM_GAE_LAMBDA
        vnext = vals[1:] + [terminal]
        adv   = np.zeros(T, dtype=np.float32)
        gae   = 0.0
        for t in reversed(range(T)):
            delta  = shape[t] + γ * vnext[t] - vals[t]
            gae    = delta + γ * λ * gae
            adv[t] = gae
        returns = adv + np.array(vals, dtype=np.float32)
        w = S.NM_Z_WEIGHT
        return np.clip((1.0 - w) * returns + w * float(terminal), -1.0, 1.0)

    def _train_from_buffer(self) -> None:
        """Sample NM_TRAIN_STEPS minibatches from the replay buffer and take
        one Adam step each.  Each sampled item gets a random D4 transform
        (policy target and mask transformed identically; value is invariant)."""
        if not self._buffer:
            return
        n_steps = (S.NM_TRAIN_STEPS
                   if len(self._buffer) >= S.NM_BATCH_SIZE else 1)
        for _ in range(n_steps):
            k   = min(S.NM_BATCH_SIZE, len(self._buffer))
            idx = np.random.choice(len(self._buffer), size=k, replace=False)
            encs, masks, dists, tgts = [], [], [], []
            for i in idx:
                enc, mask, dist, tgt = self._buffer[i]
                if S.NM_AUGMENT:
                    op   = int(np.random.randint(N_TRANSFORMS))
                    enc  = transform_planes(enc, op)
                    mask = transform_flat(
                        mask.astype(np.float32), op) > 0.5
                    dist = transform_flat(dist, op)
                encs.append(enc); masks.append(mask)
                dists.append(dist); tgts.append(tgt)

            states_t = torch.tensor(np.stack(encs), dtype=torch.float32,
                                    device=self._device)
            masks_t  = torch.tensor(np.stack(masks), dtype=torch.bool,
                                    device=self._device)
            pi_t     = torch.tensor(np.stack(dists), dtype=torch.float32,
                                    device=self._device)
            ret_t    = torch.tensor(np.array(tgts, dtype=np.float32),
                                    device=self._device)

            logits, vals_pred = self._net(states_t)
            logits_m = logits.clone()
            logits_m[~masks_t] = float('-inf')
            log_probs = F.log_softmax(logits_m, dim=1)
            probs     = log_probs.exp()

            # 0 * -inf = nan — zero illegal log-probs before the dot product.
            safe_lp = log_probs.clone()
            safe_lp[~masks_t] = 0.0
            pol_loss = -(pi_t * safe_lp).sum(dim=1).mean()
            val_loss = F.mse_loss(vals_pred, ret_t)
            legal_lp = torch.where(masks_t, log_probs,
                                   torch.zeros_like(log_probs))
            entropy  = -(probs * legal_lp).sum(dim=1).mean()

            loss = (S.NM_POLICY_COEF   * pol_loss
                    + S.NM_VALUE_COEF  * val_loss
                    - S.NM_ENTROPY_COEF * entropy)

            self._opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), S.PT_MAX_GRAD_NORM)
            self._opt.step()

    def share_learning_with(self, other: 'NeuralMCTSPlayer') -> None:
        """Make *other* train and act with this player's network/optimizer."""
        other._net    = self._net
        other._opt    = self._opt
        other._arch   = self._arch
        other._buffer = self._buffer
        self._learning_shared = other._learning_shared = True

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None) -> None:
        """Record an opponent's move as a training example.

        Encodes the position from the opponent's perspective (before their
        move), uses a one-hot policy target on their chosen action, and
        stores the network's own value estimate for GAE bootstrapping.
        At record_outcome these are trained with the inverted terminal value
        so the network learns to evaluate positions from both sides.
        """
        if self._learning_shared:
            return
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
        node_phase = len(fb.board) / max(n * n, 1)
        priors = self._tactical_priors(
            priors, fb.board, fb.connections, fb.player, legal_mask_np,
            suppress_dead=False, game_phase=node_phase)

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
        game_phase: float = 0.0,
    ) -> np.ndarray:
        """Tactically-corrected priors for MCTS nodes.

        Two improvements over the old linear apply_boost():

        1. Exponential scoring boost (phase-decayed)
           Old formula:  1 + (base−1) × score   (linear)
             score=0.5 → 2.5×,  score=2.5 → 8.5×
           New formula:  exp(2·ln(base) × score × decay) = base^(2×score×decay)
             score=0.5 → base^1 ≈  4×  (at game_phase=0)
             score=1.0 → base^2 ≈ 16×  (at game_phase=0)
             score=2.5 → base^5 ≈ 1024× (at game_phase=0)
           A network prior 100× biased toward a 0.5-pt triangle cannot beat
           a 2.5-pt square: (1/100)×1024 = 10.2  vs  1.0×4 = 4.
           By game_phase=1.0, decay=0.3, so boost is 30% of maximum —
           a well-trained network can override it with a ring strategy.

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

        # Decay tactical boost with game phase so a trained network can
        # override it late-game. Floor of 0.3 keeps a safety net alive.
        phase_decay = max(0.3, 1.0 - game_phase)
        close_k = 2.0 * math.log(max(S.AI_CLOSE_BOOST, 2.0)) * phase_decay
        block_k = 2.0 * math.log(max(S.AI_BLOCK_BOOST, 2.0)) * phase_decay

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
