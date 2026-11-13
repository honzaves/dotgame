"""Neural MCTS player — tree search guided by a residual CNN.

How it differs from the three other AI types
---------------------------------------------
Pure MCTS (mcts.py)          — heuristic priors + playouts, no network
Neural Network (nn_player)   — learned policy + value, no lookahead
PyTorch Net (pytorch_player) — learned policy + value (CNN), no lookahead
Neural MCTS (this file)      — the MCTS search core + a CNN that learns from it

Search
  The same exact-rules search as MCTSPlayer (ai/search.py: bitboard engine,
  ring-race priors, playouts).  The network contributes at the root:
    policy head → blended into the root priors
    value head  → rates the positions after the top candidate moves, seeded
                  into the tree as virtual visits
  Both contributions are scaled by a *trust* factor that grows from 0 to
  NM_MAX_TRUST over the first NM_TRUST_STEPS training steps, so an untrained
  network cannot mislead the search and a trained one increasingly steers it.

Training signal per game (AlphaZero-style)
  policy target  : the search's visit distribution — the network learns to
                   predict what the search found.
  value target   : GAE returns from the shaping rewards blended with the
                   final outcome z.
  Loss:  L = policy_coef · CE(π_search, π_net)
           + value_coef  · MSE(v_pred, v_target)
           − entropy_coef· H(π_net)

Requirements: pip install torch
"""

import os

import numpy as np

import settings as S
from ai.base_player import BasePlayer
from ai.engine import BitGame, bits
from ai.paths import experience_path
from ai.search import Searcher, SearchConfig
from ai.symmetry import transform_planes, transform_flat, N_TRANSFORMS

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass


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
    """Exact-rules tree search whose root is steered by a residual CNN.

    With a fresh network it plays like MCTSPlayer (trust = 0).  As training
    steps accumulate the policy head re-weights the root priors and the
    value head pre-rates the top candidates, so the search spends its
    simulations where the network has learned good moves are.
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
        # Training steps taken so far (persisted) — drives the trust factor.
        # A one-element list so self-play partners can share the counter.
        self._steps: list = [0]

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

        self._searcher = Searcher(SearchConfig(
            c_puct=S.NM_C_PUCT,
            noise_alpha=S.NM_NOISE_ALPHA,
            noise_frac=S.NM_NOISE_FRAC,
        ))
        self.last_search = None

        self.load()

    # ── BasePlayer interface ──────────────────────────────────────────────────

    def trust(self) -> float:
        """How much the network steers the search: 0 (untrained) rising
        linearly to NM_MAX_TRUST after NM_TRUST_STEPS training steps."""
        return S.NM_MAX_TRUST * min(1.0, self._steps[0] / max(S.NM_TRUST_STEPS, 1))

    def choose_move(self, board: dict, connections: set, player: int,
                    scores: dict, forbidden=None) -> tuple[int, int]:
        forbidden = forbidden or set()
        bg = BitGame.from_board(board, player, scores, forbidden)
        legal = bits(bg.legal())
        n, W = S.GRID, bg.g.W
        if bg.done or not legal:
            return (0, 0)

        # ── Network output at the root ────────────────────────────────────────
        enc        = _encode(board, connections, forbidden, player)
        legal_mask = np.zeros(n * n, dtype=bool)
        flat       = {i: (i // W) * n + (i % W) for i in legal}   # bit idx → flat idx
        for f in flat.values():
            legal_mask[f] = True

        with torch.no_grad():
            x      = torch.tensor(enc, dtype=torch.float32,
                                  device=self._device).unsqueeze(0)
            rl, rv = self._net(x)
            rl     = rl[0].clone()
            mask_t = torch.tensor(legal_mask, dtype=torch.bool, device=self._device)
            rl[~mask_t] = float('-inf')
            net_pol = F.softmax(rl, dim=0).cpu().numpy()   # (G²,) ≈1 over legal
        root_val = float(rv[0].item())   # in [-1,1], from player's perspective

        # ── Search ────────────────────────────────────────────────────────────
        trust = self.trust()
        res = self._searcher.search(
            bg, S.AI_THINK_MS / 1000.0,
            prior_mix={i: float(net_pol[f]) for i, f in flat.items()}
                      if trust > 0 else None,
            mix_weight=trust,
            seed_fn=self._value_seeds if trust > 0 else None)
        self.last_search = res

        # ── Visit distribution for policy training ────────────────────────────
        visit_dist = np.zeros(n * n, dtype=np.float32)
        total_vis  = sum(res.visits.values())
        if total_vis > 0:
            for i, v in res.visits.items():
                visit_dist[flat[i]] = v / total_vis
        else:
            visit_dist[flat[res.move]] = 1.0   # no simulations ran

        self._episode.append((enc, legal_mask, visit_dist, root_val))
        return bg.g.xy[res.move]

    def _value_seeds(self, bg: BitGame, scores: dict) -> dict:
        """Rate the positions after the top heuristic candidates with the
        value head → {idx: (virtual visits, value in [0,1] for the mover)}."""
        k = S.NM_VALUE_SEEDS
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        if not top:
            return {}
        encs = []
        for i in top:
            after = bg.copy()
            after.play(i)
            encs.append(_encode(after.board(), after.connections(),
                                after.forbidden(), after.player))
        with torch.no_grad():
            x = torch.tensor(np.stack(encs), dtype=torch.float32,
                             device=self._device)
            _, v = self._net(x)
        # v is from the opponent's perspective (they are to move).
        n_vis = S.NM_VALUE_VISITS * self.trust() / max(S.NM_MAX_TRUST, 1e-9)
        return {i: (n_vis, (1.0 - float(v[j].item())) / 2.0)
                for j, i in enumerate(top)}

    def record_outcome(self, winner: int,
                       intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        self._searcher.reset()   # game ended — saved subtree is stale
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
        backup = {k: v.clone() for k, v in self._net.state_dict().items()}
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
            self._steps[0] += 1

        if not all(torch.isfinite(q).all() for q in self._net.parameters()):
            # Roll back rather than keep (and save) a dead network.
            self._net.load_state_dict(backup)
            self._opt.state.clear()

    def share_learning_with(self, other: 'NeuralMCTSPlayer') -> None:
        """Make *other* train and act with this player's network/optimizer."""
        other._net    = self._net
        other._opt    = self._opt
        other._arch   = self._arch
        other._buffer = self._buffer
        other._steps  = self._steps
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

        if not mask_np[gy * n + gx]:
            return  # illegal move (bad replay data) — don't record

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
                'steps': self._steps[0],
            }, path)
        except OSError:
            pass

    def load(self) -> None:
        path = experience_path(S.NM_EXPERIENCE_BASE, '.pt')
        if not os.path.exists(path):
            return
        try:
            ck = torch.load(path, map_location=self._device, weights_only=False)
            if (ck.get('grid') == S.GRID and ck.get('arch') == self._arch
                    and all(torch.isfinite(t).all()
                            for t in ck['model'].values())):
                self._net.load_state_dict(ck['model'])
                self._opt.load_state_dict(ck['optim'])
                self._steps[0] = int(ck.get('steps', 0))
        except Exception:
            pass
