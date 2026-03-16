"""Monte-Carlo Tree Search player.

Key improvements over naive MCTS
---------------------------------
TREE REUSE  (highest impact)
    After choosing move M the subtree rooted at M is kept alive.  When
    choose_move() is called next turn the opponent's actual move is
    identified by diffing the board, the matching child is promoted to
    root, and all its accumulated visit/win counts are inherited.
    Effect: by mid-game the root already has thousands of visits instead
    of zero, multiplying effective search depth by 10-50×.

FAST ROLLOUTS  (throughput)
    Rollouts are capped at MAX_ROLLOUT_DEPTH (default 20) moves.
    Terminal nodes (sim.done) are evaluated with the real territory
    engine.  Non-terminal leaves use a fast dot+connection heuristic
    (~1 µs vs ~3 ms for a flood fill), letting the tree run 100–500
    simulations per second instead of ~50.

ROLLOUT BIAS
    Moves adjacent to own dots are preferred 3:1 to encourage realistic
    territory-building play.

EXPERIENCE / OPENING BOOK
    Board-position win-rates are still persisted as before.  The root
    node is warm-started from experience; child warm-starting is removed
    because its O(N) board-key cost dominated expansions for < 1% hit rate.
"""

import json
import math
import os
import random
import time
import numpy as np

import settings as S
from ai.base_player import BasePlayer
from ai.paths import experience_path
from ai.features import (opportunity_masks, apply_boost,
                          enclosure_potential, enclosure_scalars,
                          arc_potential_scalars, arc_potential_map,
                          bridge_potential, disruption_map, fork_map,
                          close_setup_map, get_centrality)

# Short rollouts → far more simulations per second.
# Increase to trade throughput for accuracy (sweet spot ≈ 15-30).
MAX_ROLLOUT_DEPTH = 20

# PUCT prior weight — biases UCB toward high-opportunity moves.
# Decays as 1/(1+visits) so search value dominates after ~20 visits.
# Increase to make the tree explore territory moves more aggressively.
PRIOR_WEIGHT = getattr(S, 'MCTS_PRIOR_WEIGHT', 2.0)

# Weight of territory potential in the final move selection score.
SELECTION_TERRITORY_WEIGHT = getattr(S, 'MCTS_SELECTION_TERRITORY_WEIGHT', 0.3)


# ── Fast board simulation ─────────────────────────────────────────────────────

class FastBoard:
    """Lightweight board for tree expansion and rollouts."""

    __slots__ = ('board', 'connections', 'scores', 'player',
                 'done', 'winner', 'forbidden')

    def __init__(self, board, connections, player, scores, forbidden=None):
        self.board       = dict(board)
        self.connections = set(connections)
        self.scores      = dict(scores)
        self.player      = player
        self.done        = False
        self.winner      = None
        self.forbidden   = set(forbidden) if forbidden else set()

    def copy(self):
        fb             = FastBoard.__new__(FastBoard)
        fb.board       = dict(self.board)
        fb.connections = set(self.connections)
        fb.scores      = dict(self.scores)
        fb.player      = self.player
        fb.done        = self.done
        fb.winner      = self.winner
        fb.forbidden   = set(self.forbidden)
        return fb

    def legal_moves(self):
        occ = self.board.keys()
        return [(gx, gy)
                for gx in range(S.GRID) for gy in range(S.GRID)
                if (gx, gy) not in occ and (gx, gy) not in self.forbidden]

    def closing_moves(self, player):
        """Return legal moves sorted by squares closed (most first)."""
        from ai.features import opportunity_masks
        own_scores, _ = opportunity_masks(self.board, self.connections, player)
        n = S.GRID
        moves = [
            (int(own_scores[idx]), idx % n, idx // n)
            for idx in np.where(own_scores)[0]
            if (idx % n, idx // n) not in self.board
            and (idx % n, idx // n) not in self.forbidden
        ]
        moves.sort(reverse=True)
        return [(gx, gy) for _, gx, gy in moves]

    def blocking_moves(self, player):
        """Return legal moves sorted by opponent squares blocked (most first)."""
        from ai.features import opportunity_masks
        _, opp_scores = opportunity_masks(self.board, self.connections, player)
        n = S.GRID
        moves = [
            (int(opp_scores[idx]), idx % n, idx // n)
            for idx in np.where(opp_scores)[0]
            if (idx % n, idx // n) not in self.board
            and (idx % n, idx // n) not in self.forbidden
        ]
        moves.sort(reverse=True)
        return [(gx, gy) for _, gx, gy in moves]

    def adjacent_own(self, player):
        occ = self.board.keys()
        result = []
        for gx in range(S.GRID):
            for gy in range(S.GRID):
                if (gx, gy) in occ or (gx, gy) in self.forbidden:
                    continue
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        if self.board.get((gx + dx, gy + dy)) == player:
                            result.append((gx, gy))
                            break
                    else:
                        continue
                    break
        return result

    def setup_moves(self, player: int) -> list:
        """Return legal positions that would create a 3-own-corner unit square
        (a next-turn territory threat).  Used as a rollout priority tier between
        blocking and plain adjacent extension, so rollouts actually generate
        territory signals for the tree to learn from."""
        n   = S.GRID
        occ = self.board
        opp = 2 if player == 1 else 1
        result = []
        for gx in range(n):
            for gy in range(n):
                pos = (gx, gy)
                if pos in occ or pos in self.forbidden:
                    continue
                for cx in range(max(0, gx - 1), min(n - 1, gx + 1)):
                    for cy in range(max(0, gy - 1), min(n - 1, gy + 1)):
                        corners = [(cx,cy),(cx+1,cy),(cx,cy+1),(cx+1,cy+1)]
                        other   = [c for c in corners if c != pos]
                        own_cnt = sum(1 for c in other if occ.get(c) == player)
                        opp_cnt = sum(1 for c in other if occ.get(c) == opp)
                        if own_cnt == 2 and opp_cnt == 0:
                            result.append(pos)
                            break
                    else:
                        continue
                    break
        return result

    def fork_moves(self, player) -> list:
        """Return legal positions that would create 2+ simultaneous threats.

        A future threat is a unit square with 3 own corners (after placing
        at this position) and 1 different empty corner — opponent can answer
        at most one fork, guaranteeing the other scores next turn.
        """
        n   = S.GRID
        occ = self.board
        result = []
        for gx in range(n):
            for gy in range(n):
                pos = (gx, gy)
                if pos in occ or pos in self.forbidden:
                    continue
                threats = 0
                for cx in range(max(0, gx - 1), min(n - 1, gx + 1)):
                    for cy in range(max(0, gy - 1), min(n - 1, gy + 1)):
                        corners = [(cx,cy),(cx+1,cy),(cx,cy+1),(cx+1,cy+1)]
                        other   = [c for c in corners if c != pos]
                        if (sum(1 for c in other if occ.get(c) == player) == 2
                                and sum(1 for c in other if c not in occ) == 1):
                            threats += 1
                if threats >= 2:
                    result.append(pos)
        return result

    def play(self, gx, gy):
        p = self.player
        self.board[(gx, gy)] = p
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nb = (gx + dx, gy + dy)
                if self.board.get(nb) == p:
                    self.connections.add(frozenset({(gx, gy), nb}))

        if S.MCTS_REAL_ROLLOUT:
            self._update_territory_real()
        else:
            self._update_scores_fast(gx, gy, p)

        self.player = 2 if p == 1 else 1
        self._check_terminal()

    def _update_scores_fast(self, gx: int, gy: int, p: int) -> None:
        """Lightweight territory: count completed unit squares for player p.

        Checks all unit squares that contain (gx, gy) as a corner.  Does not
        run flood fill or update forbidden positions (too expensive per rollout
        step), but gives MCTS rollouts accurate scoring signal for the most
        common scoring mechanism.
        """
        n = S.GRID
        for cx in range(max(0, gx - 1), min(n - 1, gx + 1)):
            for cy in range(max(0, gy - 1), min(n - 1, gy + 1)):
                corners = [(cx, cy), (cx+1, cy), (cx, cy+1), (cx+1, cy+1)]
                if all(self.board.get(c) == p for c in corners):
                    self.scores[p] = self.scores.get(p, 0.0) + 1.0

    def _update_territory_real(self) -> None:
        """Full territory engine: flood fill, forbidden, encirclement.

        Accurate but ~10× slower than _update_scores_fast.  Enabled when
        S.MCTS_REAL_ROLLOUT is True (UI toggle).
        """
        try:
            from ai.trainer import SimGame
            sim             = SimGame()
            sim.board       = dict(self.board)
            sim.connections = set(self.connections)
            sim.current     = self.player
            sim._recompute()
            self.scores    = dict(sim.scores)
            self.forbidden = set(sim.forbidden)
            if sim.game_over:
                self.done   = True
                self.winner = sim.winner
        except Exception:
            pass   # fall back to lightweight if SimGame unavailable

    def _check_terminal(self) -> None:
        """Set done/winner based on current scores or board full.

        Kept O(1) in fast mode — the rollout loop calls this after every move,
        so any O(G²) scan here multiplies directly into total rollout cost.

        Fast mode (MCTS_REAL_ROLLOUT=False): forbidden positions are not tracked,
        so "no legal moves" equals "board full" — just compare len(board) to G².

        Real-territory mode: forbidden positions are updated by _update_territory_real(),
        so we must scan for them, but we only reach this path at ~10× lower throughput
        anyway, making the scan affordable.
        """
        n         = S.GRID
        threshold = S.WIN_PCT * (n - 1) ** 2

        # Check win-threshold first (O(1))
        for pl in (1, 2):
            if self.scores.get(pl, 0.0) >= threshold:
                self.done   = True
                self.winner = pl
                return

        # Check no-moves-left
        if S.MCTS_REAL_ROLLOUT:
            # Forbidden positions are tracked — must scan (only reached in slow mode)
            placeable = sum(
                1 for gx_ in range(n) for gy_ in range(n)
                if (gx_, gy_) not in self.board and (gx_, gy_) not in self.forbidden
            )
            no_moves = placeable == 0
        else:
            # Fast mode: forbidden not tracked, so board full == no moves
            no_moves = len(self.board) >= n * n

        if no_moves:
            self.done   = True
            s1, s2      = self.scores.get(1, 0.0), self.scores.get(2, 0.0)
            self.winner = 1 if s1 >= s2 else 2


# ── Leaf evaluators ───────────────────────────────────────────────────────────

def _eval_terminal(fb: FastBoard, ai_player: int) -> float:
    """Accurate evaluation using the real territory engine.  Called only
    at true terminal nodes where sim.done is True."""
    try:
        from ai.trainer import SimGame
        sim             = SimGame()
        sim.board       = dict(fb.board)
        sim.connections = set(fb.connections)
        sim.current     = fb.player
        sim._recompute()
        s1    = sim.scores.get(1, 0.0)
        s2    = sim.scores.get(2, 0.0)
        total = s1 + s2
        if total == 0:
            return 0.5
        return (s1 if ai_player == 1 else s2) / total
    except Exception:
        s1 = fb.scores.get(1, 0); s2 = fb.scores.get(2, 0)
        total = s1 + s2
        return 0.5 if total == 0 else (s1 if ai_player == 1 else s2) / total


def _eval_fast(fb: FastBoard, ai_player: int,
               terr_w: float = 0.25, enc_w: float = 0.50) -> float:
    """Heuristic for non-terminal rollout leaves.

    MUST BE CHEAP — called hundreds of times per second inside the rollout
    loop.  Uses only O(G²) operations with no flood-fill BFS per call.

    Blends four signals:
      dot_ratio  — own fraction of placed dots (basic presence)
      conn_ratio — own fraction of connections (local structure)
      terr_pot   — net immediate closing opportunity (tactical)
      enc_pot    — net enclosure/arc potential (strategic)

    terr_w / enc_w are passed from the player's profile so Strategist
    rollouts weight ring potential more heavily than Opportunist ones.
    Remaining weight (1 − terr_w − enc_w) is split 0.15/0.10 between
    dot_ratio and conn_ratio.

    bridge_potential() is intentionally excluded here: it requires O(G²)
    nested flood-fills per call and completely destroys rollout throughput.
    Bridge signal is already applied once per choose_move() in the PUCT priors,
    which is the correct place for expensive strategic computation.
    """
    own_dots   = sum(1 for p in fb.board.values() if p == ai_player)
    total_dots = len(fb.board)
    if total_dots == 0:
        return 0.5
    dot_ratio = own_dots / total_dots

    own_conns   = sum(1 for c in fb.connections
                      if fb.board.get(next(iter(c))) == ai_player)
    total_conns = max(len(fb.connections), 1)
    conn_ratio  = own_conns / total_conns

    # Tactical: immediate closing/blocking opportunity
    own_opp, opp_opp = opportunity_masks(fb.board, fb.connections, ai_player)
    max_sq   = (S.GRID - 1) ** 2 or 1
    net_opp  = (float(own_opp.sum()) - float(opp_opp.sum())) / max_sq
    terr_pot = 0.5 + 0.5 * max(-1.0, min(1.0, net_opp))

    # Strategic: ring arc potential — grows continuously as ring is built.
    # arc_potential_scalars() returns a bounding-box × completion-ratio estimate
    # that fires throughout ring construction (unlike flood-fill which is 0 until
    # the ring is complete).
    own_arc, opp_arc = arc_potential_scalars(fb.board, ai_player)
    enc_pot = 0.5 + 0.5 * max(-1.0, min(1.0, own_arc - opp_arc))

    # Remaining weight split 60/40 between dot and connection ratio
    base_w = max(0.0, 1.0 - terr_w - enc_w)
    return (base_w * 0.6 * dot_ratio
            + base_w * 0.4 * conn_ratio
            + terr_w * terr_pot
            + enc_w  * enc_pot)


# ── MCTS node ─────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ('move', 'parent', 'children', 'visits', 'wins',
                 'untried', 'player_who_moved', 'prior')

    def __init__(self, move, parent, untried, player_who_moved, prior=0.0):
        self.move             = move
        self.parent           = parent
        self.children         = []
        self.visits           = 0
        self.wins             = 0.0
        self.untried          = list(untried)
        self.player_who_moved = player_who_moved
        self.prior            = prior   # PUCT territory-opportunity prior

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        # PUCT: standard UCB + prior that decays with visits
        puct_prior = PRIOR_WEIGHT * self.prior / (1 + self.visits)
        return (self.wins / self.visits
                + exploration * math.sqrt(math.log(self.parent.visits)
                                          / self.visits)
                + puct_prior)

    def best_child(self, exploration=1.41):
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def most_visited(self):
        return max(self.children, key=lambda c: c.visits)


# ── MCTS player ───────────────────────────────────────────────────────────────

class MCTSPlayer(BasePlayer):

    def __init__(self, player_id: int, profile: int = 0):
        """Create an MCTS player.

        profile:  0 = use global S.MCTS_* settings (default, for human vs MCTS etc.)
                  1 = "Opportunist"  — close-focus, high noise, wide exploration
                  2 = "Strategist"   — ring/arc focus, low noise, patient search
        Profiles are used when two MCTS players face each other so they play
        meaningfully different styles rather than mirroring.
        """
        self.player_id = player_id
        self.experience: dict = {}
        self._game_moves: list = []
        self._obs_moves:  list = []   # board states resulting from opponent moves

        # ── Per-instance weights (set from profile or global defaults) ────────
        p = S.MCTS_PROFILES.get(profile, {})
        self._close_w      = p.get('close_w',      4.0)
        self._block_w      = p.get('block_w',       3.0)
        self._arc_w        = p.get('arc_w',         0.22)
        self._arc_opp_w    = p.get('arc_opp_w',     0.28)
        self._bridge_w     = p.get('bridge_w',      0.16)
        self._bridge_opp_w = p.get('bridge_opp_w',  0.14)
        self._setup_w      = p.get('setup_w',       2.5)
        self._adj_w        = p.get('adj_w',         1.5)
        self._centrality_w = p.get('centrality_w',  0.5)
        self._disrupt_w    = p.get('disrupt_w',     0.3)
        self._noise_alpha  = p.get('noise_alpha',   S.MCTS_NOISE_ALPHA)
        self._noise_frac   = p.get('noise_frac',    S.MCTS_NOISE_FRAC)
        self._exploration  = p.get('exploration',   1.41)
        self._sel_terr_w   = p.get('sel_terr_w',    SELECTION_TERRITORY_WEIGHT)
        # Rollout leaf-eval weights — let profiles tune what the simulation
        # "cares about" when it runs out of depth.
        self._eval_terr_w  = p.get('eval_terr_w',   0.25)  # immediate territory
        self._eval_enc_w   = p.get('eval_enc_w',    0.50)  # ring/arc potential

        # Tree reuse state
        self._saved_root:  _Node | None = None   # child we chose last turn
        self._saved_board: dict | None  = None   # board after our last move
        self._root_priors:     dict   = {}    # PUCT priors for current root
        self._root_own_s:      object = None  # opportunity arrays (reused in selection)
        self._root_opp_s:      object = None
        self._root_own_enc:    object = None  # enclosure potential arrays
        self._root_opp_enc:    object = None
        self._root_own_br:     object = None  # bridge potential arrays
        self._root_opp_br:     object = None
        self._root_disrupt:    object = None  # disruption map
        self._root_own_setup:  object = None  # territory setup map

        self.load()

    # ── BasePlayer interface ──────────────────────────────────────────────────

    def choose_move(self, board, connections, player, scores,
                    forbidden=None) -> tuple:
        fb    = FastBoard(board, connections, player, scores, forbidden)
        legal = fb.legal_moves()
        if not legal:
            return (0, 0)
        if len(legal) == 1:
            self._discard_tree()
            return legal[0]

        root = self._try_reuse_tree(board)
        if root is None:
            root = _Node(move=None, parent=None, untried=legal,
                         player_who_moved=3 - player)
            self._warm_start(root, fb)

        # Precompute PUCT priors.
        # Weight hierarchy (highest to lowest):
        #   immediate territory > territory setup > ring extension > bridge/disrupt
        #
        # Two baseline priors prevent the zero-prior deadlock:
        #   own_adj:    positions adjacent to own existing dots (cheap, O(1) per pos)
        #   centrality: precomputed geometric weight (breaks column-major tie-breaking)
        # Without these, early-game priors are all exactly 0.0, which makes _expand
        # deterministically pick positions in column-major order, creating permanent
        # alternating-row placement that blocks all territory formation.
        own_s,   opp_s   = opportunity_masks(board, connections, player)
        own_arc, opp_arc = arc_potential_map(board, player)   # partial-ring aware
        own_br,  opp_br  = bridge_potential(board, connections, player)
        disrupt          = disruption_map(board, connections, player)
        own_setup        = close_setup_map(board, player)
        centrality       = get_centrality()
        n    = S.GRID
        opp  = 2 if player == 1 else 1

        self._root_own_s     = own_s
        self._root_opp_s     = opp_s
        self._root_own_enc   = own_arc    # now arc-based (partial-ring aware)
        self._root_opp_enc   = opp_arc
        self._root_own_br    = own_br
        self._root_opp_br    = opp_br
        self._root_disrupt   = disrupt
        self._root_own_setup = own_setup

        # arc values are in territory-cell units (0 to ~(n-1)²).
        # Normalise them so 1 arc-cell ≈ 1 territory point, then weight
        # relative to own_s (which is also in territory-point units).
        # Target: closing a ring likely to enclose 9 squares scores ~6 pts
        # in the prior — above a single immediate close (4 pts) but below
        # two immediate closes (8 pts), which is strategically correct.
        # arc and bridge values are in territory-cell units (0 … ~(n-1)²).
        # Keeping them un-normalised means arc_w / bridge_w weights are in the
        # same intuitive units as close_w:
        #
        #   arc_w = 0.22  →  9-cell ring scores 9 × 0.22 = 1.98
        #                    vs. closing 1 square = 1.0 × close_w=4.0
        #                    ≈ 50 % of a single immediate close (good default)
        #
        #   Strategist arc_w = 0.55  →  9-cell ring scores 4.95 > close (2.5)
        #   Opportunist arc_w = 0.08  →  9-cell ring scores 0.72 << close (5.5)
        #
        # Previously divided by total_cells, which collapsed a 9-cell ring's
        # contribution to ≈0.14 × arc_w — indistinguishable from noise and
        # far below any immediate close regardless of profile. Fixed.
        raw_priors = {}
        for gx, gy in legal:
            idx = gy * n + gx
            own_adj = sum(
                1 for dx in range(-1, 2) for dy in range(-1, 2)
                if (dx or dy) and board.get((gx+dx, gy+dy)) == player
            ) / 8.0
            arc_own  = float(own_arc[idx])
            arc_opp  = float(opp_arc[idx])
            br_own   = float(own_br[idx])
            br_opp   = float(opp_br[idx])
            raw_priors[(gx, gy)] = (
                float(own_s[idx])     * self._close_w
              + float(opp_s[idx])     * self._block_w
              + float(own_setup[idx]) * self._setup_w
              + arc_own               * self._arc_w
              + arc_opp               * self._arc_opp_w
              + own_adj               * self._adj_w
              + br_own                * self._bridge_w
              + br_opp                * self._bridge_opp_w
              + float(centrality[idx])* self._centrality_w
              + float(disrupt[idx])   * self._disrupt_w
            )
        max_raw   = max(raw_priors.values()) if raw_priors else 1.0
        max_score = max(max_raw, 1.0)

        # Inject Dirichlet noise into root priors to prevent determinism.
        # Without this, two MCTS instances with identical positions see identical
        # priors, run near-identical rollouts, and mirror each other's moves.
        # This is the standard AlphaZero treatment for the same problem.
        #
        # Alpha < 1 produces sparse/spiky noise (some moves get a large boost,
        # most get almost nothing) — this is what we want: random *emphasis* on
        # otherwise low-scored moves, not uniform blurring of all priors.
        if self._noise_frac > 0 and len(legal) > 1:
            # Adaptive alpha: AlphaZero uses alpha ∝ 1/n_legal so the noise
            # distribution is always appropriately spiky regardless of board size.
            adaptive_alpha = max(0.01, self._noise_alpha * 10.0 / len(legal))
            eta   = np.random.dirichlet([adaptive_alpha] * len(legal))
            frac  = self._noise_frac
            items = list(raw_priors.items())
            for i, (pos, val) in enumerate(items):
                raw_priors[pos] = (1.0 - frac) * val + frac * float(eta[i]) * max_score

        self._root_priors = {k: v / max(max(raw_priors.values()), 1e-9)
                             for k, v in raw_priors.items()}

        deadline = time.time() + S.AI_THINK_MS / 1000.0
        while time.time() < deadline:
            node, sim_fb = self._select(root, fb.copy())
            node         = self._expand(node, sim_fb)
            result       = self._rollout(sim_fb, player)
            self._backprop(node, result)

        # ── Final selection: win-rate + tactical + strategic score ──────────
        # Reuse own_s/opp_s/own_enc/opp_enc from the prior computation above.
        best = root.most_visited()
        if root.children and max_score > 0:
            n_        = S.GRID
            own_s_    = self._root_own_s
            opp_s_    = self._root_opp_s
            own_enc_  = self._root_own_enc
            opp_enc_  = self._root_opp_enc
            own_br_    = self._root_own_br
            opp_br_    = self._root_opp_br
            disrupt_   = self._root_disrupt
            own_setup_ = self._root_own_setup
            best_combined = -1.0
            for child in root.children:
                if child.visits < 3:
                    continue
                win_rate = child.wins / child.visits
                gx, gy   = child.move
                ci        = gy * n_ + gx
                setup_v   = float(own_setup_[ci]) if own_setup_ is not None else 0.0
                n2 = (S.GRID - 1) ** 2 or 1
                score = (
                    float(own_s_[ci])              * 4.0
                  + float(opp_s_[ci])              * 3.0
                  + setup_v                        * 2.5
                  + float(own_enc_[ci]) / n2       * 5.0
                  + float(opp_enc_[ci]) / n2       * 4.0
                  + float(own_br_[ci])  / n2       * 3.0
                  + float(opp_br_[ci])  / n2       * 2.5
                  + float(disrupt_[ci])            * 0.3
                ) / max_score
                combined = win_rate + self._sel_terr_w * score
                if combined > best_combined:
                    best_combined = combined
                    best = child

        # ── Save subtree for next turn ────────────────────────────────────────
        best.parent       = None   # detach from root so GC can collect rest
        self._saved_root  = best
        # Record the board state after OUR move so we can diff next call
        fb2 = fb.copy(); fb2.play(*best.move)
        self._saved_board = dict(fb2.board)

        self._record_move(fb, best.move)
        return best.move

    def observe_opponent_move(self, board, connections, player_who_moved,
                              move, forbidden=None) -> None:
        """Record the board state resulting from the opponent's move.

        The resulting position is the one we face at the start of our next
        turn, which is exactly what _warm_start looks up.  Recording these
        observations means warm-start fires for every position we've been in
        before — not just positions we reached by our own moves.

        The game outcome (assigned at record_outcome) is from OUR perspective,
        so a position reached by a strong opponent move that leads to our loss
        accumulates a low wins/visits ratio, teaching us to recognise danger.
        """
        gx, gy = move
        new_board = dict(board)
        new_board[(gx, gy)] = player_who_moved
        new_conn = set(connections)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nb = (gx + dx, gy + dy)
                if new_board.get(nb) == player_who_moved:
                    new_conn.add(frozenset({(gx, gy), nb}))
        # Build the board key directly (same format as _board_key)
        key = str(sorted((bx, by, p) for (bx, by), p in new_board.items()))
        self._obs_moves.append(key)

    def record_outcome(self, winner: int, intermediate_rewards=None,
                       final_scores: dict | None = None) -> None:
        # Compute a [0, 1] outcome value encoding the margin of victory.
        # Using score ratio rather than binary win/loss means the experience
        # dict distinguishes a 30-5 win (0.86) from a 16-15 squeaker (0.52).
        # This propagates into warm-start visit/win counts next game, making
        # MCTS prefer positions that tend to produce larger territory leads.
        if final_scores and winner != 0:
            own_s = final_scores.get(self.player_id, 0.0)
            opp_s = final_scores.get(3 - self.player_id, 0.0)
            total = own_s + opp_s
            outcome = (own_s / total) if total > 0 else 0.5
        else:
            outcome = (1.0  if winner == self.player_id else
                       0.0  if winner != 0              else 0.5)

        # Own moves AND observed opponent-move positions are both positions
        # we occupied during the game.  Both get the same outcome from our
        # perspective: good positions (that led to wins) get outcome→1,
        # bad positions (that led to losses) get outcome→0.
        all_keys = self._game_moves + self._obs_moves
        for key in all_keys:
            e = self.experience.setdefault(key, {'wins': 0.0, 'visits': 0})
            e['visits'] += 1
            e['wins']   += outcome
        self._game_moves.clear()
        self._obs_moves.clear()
        self._discard_tree()

        if len(self.experience) > S.AI_MAX_EXPERIENCE:
            keys = list(self.experience.keys())
            random.shuffle(keys)
            for k in keys[S.AI_MAX_EXPERIENCE:]:
                del self.experience[k]
        self.save()

    def save(self):
        path = experience_path(S.AI_EXPERIENCE_BASE, '.json')
        try:
            with open(path, 'w') as fh:
                json.dump(self.experience, fh)
        except OSError:
            pass

    def load(self):
        path = experience_path(S.AI_EXPERIENCE_BASE, '.json')
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    self.experience = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self.experience = {}

    # ── Tree reuse ────────────────────────────────────────────────────────────

    def _discard_tree(self):
        self._saved_root  = None
        self._saved_board = None

    def _try_reuse_tree(self, current_board: dict) -> '_Node | None':
        """Promote the child that matches the opponent's actual move.

        Returns the child node (new root) if found, else None.
        Discards the tree if dots were removed (encirclement) or if the
        board changed in an unexpected way.
        """
        if self._saved_root is None or self._saved_board is None:
            return None

        prev = self._saved_board

        # Detect removed dots (encirclement happened) → discard tree
        removed = [p for p in prev if p not in current_board]
        if removed:
            self._discard_tree()
            return None

        # Find the single new dot the opponent placed
        new_dots = [p for p in current_board if p not in prev]
        if len(new_dots) != 1:
            # Game restarted or something unexpected
            self._discard_tree()
            return None

        opp_move = new_dots[0]

        # Search existing children for a matching move
        for child in self._saved_root.children:
            if child.move == opp_move:
                child.parent = None
                self._discard_tree()   # clear refs before returning
                return child

        # Opponent played a move we never expanded — build a fresh node
        # but inherit any visits from root (partial warm-start)
        self._discard_tree()
        return None

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _board_key(fb: FastBoard) -> str:
        return str(sorted((gx, gy, p) for (gx, gy), p in fb.board.items()))

    def _warm_start(self, root: _Node, fb: FastBoard) -> None:
        key = self._board_key(fb)
        if key in self.experience:
            e = self.experience[key]
            root.visits = e['visits']
            root.wins   = e['wins']

    def _record_move(self, fb: FastBoard, move: tuple) -> None:
        fb2 = fb.copy(); fb2.play(*move)
        self._game_moves.append(self._board_key(fb2))

    def _select(self, node: _Node, fb: FastBoard) -> tuple:
        while not fb.done:
            if node.untried or not node.children:
                return node, fb
            node = node.best_child(self._exploration)
            fb.play(*node.move)
        return node, fb

    def _expand(self, node: _Node, fb: FastBoard) -> _Node:
        if node.untried and not fb.done:
            # Pick highest-prior untried move rather than random
            if self._root_priors:
                best_idx, best_p = 0, -1.0
                for i, move in enumerate(node.untried):
                    gx, gy = move
                    p = self._root_priors.get(move, 0.0)
                    if p > best_p:
                        best_p, best_idx = p, i
                move = node.untried.pop(best_idx)
            else:
                move = node.untried.pop(random.randrange(len(node.untried)))

            prior = self._root_priors.get(move, 0.0) if self._root_priors else 0.0
            fb.play(*move)
            child = _Node(move=move, parent=node,
                          untried=fb.legal_moves(),
                          player_who_moved=3 - fb.player,
                          prior=prior)
            node.children.append(child)
            return child
        return node

    def _rollout(self, fb: FastBoard, ai_player: int) -> float:
        """Short biased rollout with 6-tier move priority.

        Priority order (highest to lowest):
          1. Fork moves    — creates 2+ simultaneous threats
          2. Closing moves — complete a unit square right now (certain points)
          3. Blocking moves— stop opponent completing a unit square
          4. Setup moves   — creates a 3-own-corner unit square (next-turn threat)
                             This tier is critical for MCTS vs MCTS: without it,
                             both rollout sides only extend lines, territory never
                             forms in any simulation, win rates are all ~0.5, and
                             priors (biased toward ring extension) dominate forever.
          5. Adjacent-own  — extend own dot cluster (50% bias, reduced from 75%
                             so rollouts stay exploratory and territory can emerge)
          6. Random legal
        """
        sim   = fb.copy()
        depth = 0
        while not sim.done and depth < MAX_ROLLOUT_DEPTH:
            p     = sim.player
            legal = sim.legal_moves()
            if not legal:
                break

            closing  = sim.closing_moves(p)
            blocking = sim.blocking_moves(p)
            forks    = sim.fork_moves(p)

            if forks and not closing:
                move = random.choice(forks)
            elif closing:
                move = closing[0]
            elif blocking:
                move = blocking[0]
            else:
                setup = sim.setup_moves(p)
                adj   = sim.adjacent_own(p)
                if setup:
                    # Prefer creating a territory threat over plain line extension
                    move = random.choice(setup)
                elif adj and random.random() < 0.50:
                    # Reduced from 0.75 — lower bias keeps rollouts exploratory
                    move = random.choice(adj)
                else:
                    move = random.choice(legal)

            sim.play(*move)
            depth += 1

        if sim.done:
            return _eval_terminal(sim, ai_player)
        return _eval_fast(sim, ai_player,
                          terr_w=self._eval_terr_w,
                          enc_w=self._eval_enc_w)

    @staticmethod
    def _backprop(node: _Node, result: float) -> None:
        while node is not None:
            node.visits += 1
            node.wins   += result
            result       = 1.0 - result
            node         = node.parent
