"""Monte-Carlo tree search on the exact engine, guided by the ring race.

One search = PUCT tree + playouts:

  priors    strategy.move_scores() at the root — exact one-move tactics plus
            the ring race after each move — turned into a softmax prior.
            Deeper nodes reuse the root's score maps (one per player) as
            priors: stale, but they keep the tree on sensible moves.
  playouts  full games on BitGame with uniformly random moves — real
            territory, captures and both win conditions decide the result.

Measured, not assumed: seeding heuristic values as virtual visits, biasing
playouts toward ring points and re-analysing busy tree nodes were all tried
and each made the search weaker in head-to-head matches, so the heuristics
enter through the priors only and the playouts are left to judge.

On multi-core machines the simulations also run in worker processes
(ai/parallel.py): each worker searches its own tree from the same root and
the root statistics are added up ("root parallelisation").

The Searcher is stateless between calls except for the saved subtree (tree
reuse across turns).
"""

import math
import random
import time

import numpy as np

import settings as S
from ai.engine import BitGame, bits
from ai.features import adaptive_dirichlet_alpha
from ai import strategy
from ai import parallel


class SearchConfig:
    """Tunable search parameters (defaults from settings)."""

    def __init__(self, **kw):
        self.c_puct       = S.SEARCH_C_PUCT
        self.prior_temp   = S.SEARCH_PRIOR_TEMP
        self.noise_alpha  = 0.0
        self.noise_frac   = 0.0
        self.weights      = strategy.Weights()
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"unknown search option {k!r}")
            setattr(self, k, v)


class _Node:
    __slots__ = ('move', 'parent', 'children', 'N', 'V', 'P', 'mover',
                 'virtual')

    def __init__(self, move, parent, mover, prior=0.0):
        self.move = move
        self.parent = parent
        self.children = None      # None = not expanded
        self.N = 0.0              # visits (incl. virtual ones)
        self.V = 0.0              # summed value for `mover`, each in [0, 1]
        self.P = prior
        self.mover = mover        # player who played `move` to get here
        self.virtual = 0.0        # how many of N are virtual (seeded) visits


class SearchResult:
    """move: chosen point index;  visits: {idx: real (simulated) visits};
    value: mean outcome of the chosen move in [0, 1];  scores: the root's
    move_scores();  sims: simulations run."""
    __slots__ = ('move', 'visits', 'value', 'scores', 'sims')


class Searcher:

    def __init__(self, config: SearchConfig = None):
        self.cfg = config or SearchConfig()
        self._saved = None        # (subtree root, BitGame after our move)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget the saved subtree (new game)."""
        self._saved = None

    def search(self, bg: BitGame, think_s: float, prior_mix=None,
               mix_weight: float = 0.5, seed_visits=None,
               seed_fn=None) -> SearchResult:
        """Search *bg* (side to move = bg.player) for about *think_s* seconds.

        prior_mix:   optional {idx: probability} from a learned policy,
                     blended into the root priors with weight *mix_weight*.
        seed_visits: optional {idx: (n, value)} — virtual visits for root
                     moves (e.g. from an opening book), value in [0, 1]
                     from the mover's perspective.
        seed_fn:     optional callable(bg, scores) → same dict, called once
                     the root's heuristic scores are known (lets a value
                     network rate the top candidates).
        """
        t_end = time.time() + think_s
        me = bg.player
        res = SearchResult()
        res.sims = 0

        legal = bits(bg.legal())
        if len(legal) <= 1:
            self._saved = None
            res.move = legal[0] if legal else None
            res.visits = {res.move: 1} if legal else {}
            res.value = 0.5
            res.scores = {}
            return res

        # ── Root analysis ─────────────────────────────────────────────────────
        fast = strategy.fast_mode(think_s, bg.g.n)
        pa = strategy.analyse_position(bg, n_centres=1 if fast else None)
        scores = strategy.move_scores(bg, pa, self.cfg.weights,
                                      exact_blocks=not fast)
        res.scores = scores
        if max(scores.values()) == math.inf:       # immediate win
            self._saved = None
            res.move = next(i for i, v in scores.items() if v == math.inf)
            res.visits = {res.move: 1}
            res.value = 1.0
            return res

        root = self._reuse(bg)
        if root is None:
            root = _Node(None, None, 3 - me)
        self._set_root_children(root, me, scores, prior_mix, mix_weight)

        seeds = dict(seed_visits or {})
        if seed_fn is not None:
            for i, (n, v) in seed_fn(bg, scores).items():
                n0, v0 = seeds.get(i, (0.0, 0.0))
                seeds[i] = (n0 + n, (n0 * v0 + n * v) / (n0 + n))
        for c in root.children:
            if c.move in seeds:
                n, v = seeds[c.move]
                c.N += n
                c.V += n * v
                c.virtual += n

        # Score map for the opponent's turns deeper in the tree.
        opp_view = bg.copy()
        opp_view.player = 3 - me
        maps = {me: scores,
                3 - me: strategy.move_scores(opp_view, pa, self.cfg.weights,
                                             exact_blocks=False)}

        # ── Simulations (here and, if available, in worker processes) ────────
        pool = None
        remaining = t_end - time.time()
        if remaining >= S.SEARCH_PARALLEL_MIN_S:
            pool = parallel.pool()
        if pool is not None and pool.lock.acquire(blocking=False):
            try:
                ok = pool.submit({
                    'grid': bg.g.n, 'win_pct': S.WIN_PCT,
                    'state': (bg.d1, bg.d2, bg.forb, bg.s1, bg.s2,
                              bg.player, bg.dirty),
                    'priors': {c.move: c.P for c in root.children},
                    'maps': maps, 'think_s': remaining,
                    'c_puct': self.cfg.c_puct,
                    'prior_temp': self.cfg.prior_temp})
                res.sims += self._simulate(root, bg, maps, t_end)
                extra = pool.collect(timeout=1.0) if ok else []
            finally:
                pool.lock.release()
        else:
            res.sims += self._simulate(root, bg, maps, t_end)
            extra = []

        # ── Decision: own tree + worker statistics ───────────────────────────
        total = {c.move: [c.N, c.V] for c in root.children}
        for sims, stats in extra:
            res.sims += sims
            for move, (n, v) in stats.items():
                total[move][0] += n
                total[move][1] += v
        best = max(root.children, key=lambda c: (total[c.move][0], c.P))
        res.move = best.move
        res.visits = {c.move: total[c.move][0] - c.virtual
                      for c in root.children}
        n_best, v_best = total[best.move]
        res.value = v_best / n_best if n_best else 0.5

        after = bg.copy()
        after.play(best.move)
        best.parent = None
        self._saved = (best, after)
        return res

    def _simulate(self, root, bg: BitGame, maps: dict, t_end: float) -> int:
        """Run simulations from *root* until *t_end*; returns how many."""
        me = bg.player
        sims = 0
        while time.time() < t_end:
            node = root
            sim = bg.copy()
            while node.children and not sim.done:
                node = self._select(node)
                sim.play(node.move)
            if not sim.done and node.children is None:
                self._expand(node, sim, maps)
            value = self._playout(sim, me)
            while node is not None:
                node.N += 1
                node.V += value if node.mover == me else 1.0 - value
                node = node.parent
            sims += 1
        return sims

    # ── Tree ──────────────────────────────────────────────────────────────────

    def _reuse(self, bg: BitGame):
        """Return last turn's subtree if *bg* is the position after exactly
        one opponent reply to it, else None."""
        saved, self._saved = self._saved, None
        if saved is None:
            return None
        node, after = saved
        if not node.children or after.done:
            return None
        new_dots = (bg.d1 | bg.d2) & ~(after.d1 | after.d2)
        for child in node.children:
            if new_dots >> child.move & 1:
                probe = after.copy()
                probe.play(child.move)
                if (probe.d1, probe.d2, probe.forb) == (bg.d1, bg.d2, bg.forb):
                    child.parent = None
                    return child
                return None
        return None

    def _priors(self, scores: dict) -> dict:
        """Softmax of heuristic scores (a winning move, scored +inf, takes
        essentially all the mass)."""
        t = self.cfg.prior_temp
        m = max(scores.values())
        if m == math.inf:
            finite = [v for v in scores.values() if v != math.inf]
            m = (max(finite) if finite else 0.0) + 1.0
            scores = {i: min(v, m) for i, v in scores.items()}
        e = {i: math.exp((v - m) / t) for i, v in scores.items()}
        z = sum(e.values())
        return {i: v / z for i, v in e.items()}

    def _set_root_children(self, root, player, scores, prior_mix,
                           mix_weight) -> None:
        """Create the root's children — or, on a reused subtree, refresh
        their priors and add moves that were never expanded."""
        cfg = self.cfg
        pri = self._priors(scores)
        if prior_mix:
            z = sum(prior_mix.get(i, 0.0) for i in pri) or 1.0
            pri = {i: ((1.0 - mix_weight) * p
                       + mix_weight * prior_mix.get(i, 0.0) / z)
                   for i, p in pri.items()}
        if cfg.noise_frac > 0 and len(pri) > 1:
            alpha = adaptive_dirichlet_alpha(cfg.noise_alpha, len(pri))
            eta = np.random.dirichlet([alpha] * len(pri))
            pri = {i: (1 - cfg.noise_frac) * p + cfg.noise_frac * float(e)
                   for (i, p), e in zip(pri.items(), eta)}

        old = {c.move: c for c in (root.children or [])}
        children = []
        for i in scores:
            c = old.get(i) or _Node(i, root, player)
            c.P = pri[i]
            children.append(c)
        root.children = children

    def _expand(self, node, sim: BitGame, maps: dict) -> None:
        """Priors from the root's score map for the side to move."""
        player = sim.player
        smap = maps[player]
        legal = bits(sim.legal())
        if not legal:
            node.children = []
            return
        floor = min(smap.values()) if smap else 0.0
        pri = self._priors({i: smap.get(i, floor) for i in legal})
        node.children = [_Node(i, node, player, pri[i]) for i in legal]

    def _select(self, node):
        c_puct = self.cfg.c_puct
        sq = math.sqrt(node.N + 1.0)
        # First-play urgency: an unvisited child is assumed slightly worse
        # than the node's average outcome for the player choosing here.
        fpu = 1.0 - (node.V / node.N if node.N else 0.5) - 0.05
        best, best_u = None, -1e18
        for c in node.children:
            n = c.N
            u = (c.V / n if n else fpu) + c_puct * c.P * sq / (1.0 + n)
            if u > best_u:
                best, best_u = c, u
        return best

    # ── Playouts ──────────────────────────────────────────────────────────────

    @staticmethod
    def _result(sim: BitGame, me: int) -> float:
        """Game result in [0, 1] for *me*: win/loss with a small margin term
        so that bigger wins (and narrower losses) are preferred."""
        diff = (sim.s1 - sim.s2) if me == 1 else (sim.s2 - sim.s1)
        margin = max(-1.0, min(1.0, diff / sim.g.threshold))
        if sim.winner is None:
            return 0.5 + 0.1 * margin
        return (0.9 + 0.1 * margin) if sim.winner == me else (0.1 + 0.1 * margin)

    def _playout(self, sim: BitGame, me: int) -> float:
        """Play *sim* to the end with uniformly random legal moves."""
        order = sim.g.points[:]
        random.shuffle(order)
        play = sim.play
        while not sim.done:
            for i in order:
                if (sim.d1 | sim.d2 | sim.forb) >> i & 1:
                    continue
                play(i)
                if sim.done:
                    break
            else:
                # Pass complete but the game goes on: a capture freed points
                # that were skipped earlier — go round again.
                if not sim.legal():
                    break
        return self._result(sim, me)
