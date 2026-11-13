"""Strategic evaluation shared by the AI players.

Combines the exact engine (ai/engine.py) with the ring-race analysis
(ai/rings.py) into two things the players consume:

  PositionAnalysis  per player: moves-to-win W, ring potential U, and for
                    every point how much a ring through it would cost
  move_scores()     a heuristic value for every legal move of the side to
                    move — exact tactics (captures, closes, threats denied)
                    plus the ring race after the move

The race model: W_p = fewest additional dots player p needs to close a ring
that takes its score over the win threshold, if undisturbed.  With the
opponent to move after our move we are ahead iff W_me < W_opp.  A move is
good when it shortens our ring, lengthens theirs (only possible on points
ALL their cheapest rings need — the analysis is re-run with the point
blocked, so futile blocks that are trivially bypassed earn nothing), or both.
"""

import math

import numpy as np

import settings as S
from ai.engine import BitGame, bits
from ai import rings

INF_W = rings.NEVER


class PositionAnalysis:
    """Ring-race facts about one position, for both players."""

    __slots__ = ('W', 'U', 'tc', 'tu', 'target', 'per_centre', 'view')

    def __init__(self):
        self.W = {}       # player → moves to win (INF_W = no winning ring)
        self.U = {}       # player → discounted ring potential (cells)
        self.tc = {}      # player → {idx: cost of cheapest winning ring through idx}
        self.tu = {}      # player → {idx: potential of best ring through idx}
        self.target = {}  # player → new area still needed to win
        # player → [(centre geo, W, U, through-costs per rank, potentials per rank)]
        self.per_centre = {}
        # player → (own dots, blocked points, band-area table) for re-analysis
        self.view = {}


def _player_view(bg: BitGame, p: int):
    own, opp = (bg.d1, bg.d2) if p == 1 else (bg.d2, bg.d1)
    score = bg.s1 if p == 1 else bg.s2
    V = rings.band_areas(bg.g.n, bg.enclosed(p)) if score > 0 else None
    return own, opp | bg.forb, V, bg.g.threshold - score


def analyse_position(bg: BitGame, detail: bool = True,
                     n_centres: int = None) -> PositionAnalysis:
    """Analyse both players' rings.  detail=False skips the per-point maps
    (forward pass only — about 3× cheaper)."""
    gamma = S.RING_DISCOUNT
    pa = PositionAnalysis()
    cs = rings.centres(bg.g.n)[:n_centres]
    for p in (1, 2):
        own, blocked, V, target = _player_view(bg, p)
        pa.target[p] = target
        pa.view[p] = (own, blocked, V)
        W, U = INF_W, 0.0
        tc, tu = {}, {}
        per = []
        for cg in cs:
            info = rings.analyse(own, blocked, cg, backward=detail, V=V)
            w = info.moves_to(target)
            u = float(rings.discounted(info.profile, gamma))
            W = min(W, w)
            U = max(U, u)
            costs = pots = None
            if detail:
                best = rings.through_table(info)
                costs = rings.through_costs(best, target)
                pots = rings.discounted(best, gamma)
                order = cg.order
                for r in np.nonzero(costs < INF_W)[0]:
                    i = order[r]
                    c = int(costs[r])
                    if c < tc.get(i, INF_W):
                        tc[i] = c
                for r in np.nonzero(pots > 0)[0]:
                    i = order[r]
                    v = float(pots[r])
                    if v > tu.get(i, 0.0):
                        tu[i] = v
            per.append((cg, w, u, costs, pots))
        pa.W[p], pa.U[p], pa.tc[p], pa.tu[p] = W, U, tc, tu
        pa.per_centre[p] = per
    return pa


def _blocked_outlook(pa: PositionAnalysis, p: int, i: int) -> tuple:
    """(W, U) of player *p* if the point *i* were taken away from it.

    Exact, but lazy: a centre's ring DP is re-run only if the point lies on
    that centre's cheapest winning ring or its best-potential ring, and only
    while that centre can still decide the overall minimum / maximum."""
    own, blocked, V = pa.view[p]
    blocked |= 1 << i
    target = pa.target[p]
    gamma = S.RING_DISCOUNT
    memo = {}

    def redo(k, cg):
        if k not in memo:
            info = rings.analyse(own, blocked, cg, V=V)
            memo[k] = (info.moves_to(target),
                       float(rings.discounted(info.profile, gamma)))
        return memo[k]

    per = pa.per_centre[p]
    W = INF_W
    for k, (cg, w, u, costs, pots) in sorted(enumerate(per),
                                             key=lambda t: t[1][1]):
        if w >= W:
            break                      # cannot lower the minimum any more
        r = cg.rank_of.get(i)
        if r is not None and w < INF_W and costs[r] <= w:
            w = redo(k, cg)[0]
        W = min(W, w)
    U = 0.0
    for k, (cg, w, u, costs, pots) in sorted(enumerate(per),
                                             key=lambda t: -t[1][2]):
        if u <= U:
            break
        r = cg.rank_of.get(i)
        if r is not None and u > 0 and pots[r] >= u * 0.999:
            u = redo(k, cg)[1]
        U = max(U, u)
    return W, U


def fast_mode(think_s: float, n: int = None) -> bool:
    """True when the thinking time is too short for the full analysis
    (all centres + exact re-analysis of blocks).  The analysis cost grows
    roughly with the cube of the grid size, so the cut-off scales with it."""
    n = n or S.GRID
    return think_s < S.SEARCH_FAST_BELOW_S * (n / 10.0) ** 3


def race_value(w_me: int, w_opp: int, me_to_move: bool) -> float:
    """Ring-race outlook in (-1, 1) from 'me'.  The side to move needs only
    to tie the distance; the steepness grows as either ring nears closure."""
    if w_me >= INF_W and w_opp >= INF_W:
        return 0.0
    margin = (w_opp - w_me) + (0.5 if me_to_move else -0.5)
    steep = S.RACE_STEEPNESS / (1.0 + min(w_me, w_opp) / 3.0)
    return math.tanh(margin * steep)


def tactical_gains(bg: BitGame) -> tuple:
    """Exact one-move swings: ({idx: own gain}, {idx: opponent gain}) over
    the legal points where the gain is non-zero.  'Gain' is own score gained
    plus opponent score destroyed; math.inf marks an immediately winning
    move."""
    own_g, opp_g = {}, {}
    legal = bg.legal()
    for player, out in ((bg.player, own_g), (3 - bg.player, opp_g)):
        d = bg.d1 if player == 1 else bg.d2
        NB = bg.g.NB
        for i in bits(legal):
            if not bg.dirty and (d & NB[i]).bit_count() < 2:
                continue
            c = bg.copy()
            c.player = player
            c.play(i)
            if c.done and c.winner == player:
                out[i] = math.inf
                continue
            if player == 1:
                gsw = (c.s1 - bg.s1) + (bg.s2 - c.s2)
            else:
                gsw = (c.s2 - bg.s2) + (bg.s1 - c.s1)
            if gsw:
                out[i] = gsw
    return own_g, opp_g


class Weights:
    """Move-scoring weights (defaults from settings; MCTS profiles override)."""

    __slots__ = ('race', 'potential', 'gain', 'threat')

    def __init__(self, race=None, potential=None, gain=None, threat=None):
        self.race = S.STRAT_RACE_W if race is None else race
        self.potential = S.STRAT_POTENTIAL_W if potential is None else potential
        self.gain = S.STRAT_GAIN_W if gain is None else gain
        self.threat = S.STRAT_THREAT_W if threat is None else threat


def move_scores(bg: BitGame, pa: PositionAnalysis = None,
                w: Weights = None, exact_blocks: bool = True) -> dict:
    """Heuristic value of every legal move for the side to move.

    Returns {idx: score}; higher is better, differences of ~0.1 matter.
    A winning move scores +inf.

    exact_blocks=False skips re-analysing the opponent's rings for each
    candidate blocking point (the expensive part) and credits half a dot of
    delay instead — for very short thinking times.
    """
    if pa is None:
        pa = analyse_position(bg)
    if w is None:
        w = Weights()
    me, opp = bg.player, 3 - bg.player
    gamma = S.RING_DISCOUNT
    n2 = float((bg.g.n - 1) ** 2)
    own_g, opp_g = tactical_gains(bg)

    # Opponent's best and second-best immediate swing: taking the point of
    # the best one only helps by the difference to the runner-up.
    opp_sorted = sorted(opp_g.items(), key=lambda kv: -kv[1])
    top_i, top_g = (opp_sorted[0] if opp_sorted else (None, 0.0))
    second_g = opp_sorted[1][1] if len(opp_sorted) > 1 else 0.0

    Wm, Wo = pa.W[me], pa.W[opp]
    Um, Uo = pa.U[me], pa.U[opp]
    tc_m, tc_o = pa.tc[me], pa.tc[opp]
    tu_m, tu_o = pa.tu[me], pa.tu[opp]
    g = bg.g
    cx = cy = (g.n - 1) / 2.0
    own_d = bg.d1 if me == 1 else bg.d2

    scores = {}
    for i in bits(bg.legal()):
        go = own_g.get(i, 0.0)
        if go == math.inf:
            scores[i] = math.inf
            continue

        # Our ring after the move
        wm = min(Wm, tc_m[i] - 1) if i in tc_m else Wm
        um = max(Um, tu_m[i] / gamma) if i in tu_m else Um

        # Their ring after the move: only re-analyse points their best
        # rings actually use.
        on_win_ring = Wo < INF_W and tc_o.get(i, INF_W) <= Wo
        on_pot_ring = Uo > 0 and tu_o.get(i, 0.0) >= Uo * 0.999
        if not (on_win_ring or on_pot_ring):
            wo, uo = Wo, Uo
        elif exact_blocks:
            wo, uo = _blocked_outlook(pa, opp, i)
        else:
            wo, uo = Wo + (0.5 if on_win_ring else 0.0), Uo * 0.97

        threat_left = second_g if i == top_i else top_g
        if threat_left == math.inf:
            threat_left = n2          # they still win next move

        gx, gy = g.xy[i]
        scores[i] = (
            w.race * race_value(wm, wo, me_to_move=False)
            + w.potential * (um - uo) / n2
            + w.gain * go / n2
            - w.threat * threat_left / n2
            # tie-breaks: stay connected and central
            + 0.002 * (own_d & g.NB[i]).bit_count()
            - 0.0005 * (abs(gx - cx) + abs(gy - cy))
        )
    return scores


def guidance_offsets(board: dict, player: int, scores: dict = None,
                     forbidden=None, fast: bool = None) -> np.ndarray:
    """Log-domain policy guidance for the policy-network players.

    Returns float32 (G²,) offsets to ADD to the network's logits before the
    masked softmax:  (move score − best score) / AI_GUIDE_TEMP, clipped to
    [−AI_GUIDE_CLIP, 0], and 0 at illegal points.  An untrained network
    therefore plays close to the best heuristic move, and training shapes
    the (bounded) logits relative to it.  The players store the offsets with each
    trajectory step and replay them at training time, so sampling and
    training use the same distribution.

    scores=None derives scores/forbidden from the board (used for observed
    opponent moves, where only the board is known).  fast=None picks the
    analysis depth from the thinking time; observers pass fast=True because
    they run on the UI thread.
    """
    n = S.GRID
    out = np.zeros(n * n, dtype=np.float32)
    bg = BitGame.from_board(board, player, scores, forbidden)
    if bg.done or not bg.legal():
        return out
    if fast is None:
        fast = fast_mode(S.AI_THINK_MS / 1000.0, n)
    pa = analyse_position(bg, n_centres=1 if fast else None)
    sc = move_scores(bg, pa, exact_blocks=not fast)
    finite = [v for v in sc.values() if v != math.inf]
    if not finite:
        return out                    # every legal move wins — no preference
    best = (max(finite) + 1.0) if len(finite) < len(sc) else max(finite)
    xy = bg.g.xy
    for i, v in sc.items():
        gx, gy = xy[i]
        v = best if v == math.inf else v
        out[gy * n + gx] = max(-S.AI_GUIDE_CLIP, (v - best) / S.AI_GUIDE_TEMP)
    return out
